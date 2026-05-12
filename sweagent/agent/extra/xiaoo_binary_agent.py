from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import time
from pathlib import Path
from typing import Any, Self

from swerex.runtime.abstract import UploadRequest

from sweagent import __version__, get_agent_commit_hash, get_rex_commit_hash, get_rex_version
from sweagent.agent.agents import XiaoOBinaryAgentConfig
from sweagent.agent.hooks.abstract import AbstractAgentHook, CombinedAgentHook
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.environment.swe_env import SWEEnv
from sweagent.types import AgentInfo, AgentRunResult, Trajectory, TrajectoryStep
from sweagent.utils.log import get_logger


class XiaoOBinaryAgent:
    def __init__(self, config: XiaoOBinaryAgentConfig):
        self.config = config.model_copy(deep=True)
        self.replay_config = None
        self._chook = CombinedAgentHook()
        self.logger = get_logger("swea-xiaoo", emoji="x")
        self.trajectory: Trajectory = []
        self.info: AgentInfo = AgentInfo()
        self.traj_path: Path | None = None

    @classmethod
    def from_config(cls, config: XiaoOBinaryAgentConfig) -> Self:
        return cls(config)

    def add_hook(self, hook: AbstractAgentHook) -> None:
        self._chook.add_hook(hook)

    def _upload_file(self, env: SWEEnv, source_path: Path, target_path: str) -> None:
        if not source_path.exists():
            msg = f"Configured xiaoO file does not exist: {source_path}"
            raise FileNotFoundError(msg)
        env.communicate(f"mkdir -p {shlex.quote(str(Path(target_path).parent))}", timeout=30, check="raise")
        asyncio.run(env.deployment.runtime.upload(UploadRequest(source_path=str(source_path), target_path=target_path)))

    def _install_xiaoo(self, env: SWEEnv) -> None:
        self._upload_file(env, self.config.binary_path, self.config.container_binary_path)
        env.communicate(f"chmod +x {shlex.quote(self.config.container_binary_path)}", timeout=30, check="raise")

        if self.config.config_path is not None:
            self._upload_file(env, self.config.config_path, self.config.container_config_path)

        env_vars = {
            key: os.environ[key]
            for key in self.config.propagate_env_variables
            if os.environ.get(key) is not None
        }
        if self.config.config_path is not None:
            env_vars["XIAOO_CONFIG"] = self.config.container_config_path
        env.set_env_variables(env_vars)

        for command in self.config.setup_commands:
            env.communicate(command, timeout=300, check="raise")

        env.communicate(
            f"{shlex.quote(self.config.container_binary_path)} --help >/dev/null",
            timeout=30,
            check="raise",
            error_msg="xiaoO binary failed inside the SWE-bench container",
        )

    def _build_task(self, repo_root: str, problem_statement: ProblemStatement | ProblemStatementConfig) -> str:
        return f"""We need solve this SWE-bench issue.

Repository: {repo_root}

Problem statement:
{problem_statement.get_problem_statement_for_env()}

Edit the repository to fix the issue. Do not modify tests unless absolutely necessary.
Run relevant checks if practical. When finished, leave the working tree with only the intended source changes.
"""

    def _build_command(self, repo_root: str) -> str:
        args = [
            shlex.quote(self.config.container_binary_path),
            "run",
            "--max-turns",
            str(self.config.max_turns),
        ]
        if self.config.config_path is not None:
            args.extend(["--config", shlex.quote(self.config.container_config_path)])
        if self.config.provider:
            args.extend(["--provider", shlex.quote(self.config.provider)])
        if self.config.model:
            args.extend(["--model", shlex.quote(self.config.model)])
        if self.config.api_base:
            args.extend(["--api-base", shlex.quote(self.config.api_base)])
        if self.config.reasoning_effort:
            args.extend(["--reasoning-effort", shlex.quote(self.config.reasoning_effort)])
        if self.config.debug:
            args.append("--debug")
        args.extend(["--prompt", '"$(cat /root/xiaoo_task.md)"'])
        return f"cd {shlex.quote(repo_root)} && {' '.join(args)}"

    def _collect_submission(self, env: SWEEnv, repo_root: str) -> str | None:
        env.execute_command("git add -A && git diff --cached > /root/model.patch", check=False, cwd=repo_root)
        submission = env.read_file("/root/model.patch", encoding="utf-8", errors="backslashreplace")
        return submission if submission.strip() else None

    def _save_trace_db(self, env: SWEEnv, output_dir: Path) -> str | None:
        if not self.config.trace_db_path:
            return None

        trace_db_path = self.config.trace_db_path
        if trace_db_path.startswith("~/"):
            trace_db_path = f"/root/{trace_db_path[2:]}"

        encoded_path = "/root/xiaoo_trace.db.b64"
        command = (
            f"rm -f {shlex.quote(encoded_path)}; "
            f"if [ -s {shlex.quote(trace_db_path)} ]; then "
            f"base64 {shlex.quote(trace_db_path)} > {shlex.quote(encoded_path)}; "
            "fi"
        )
        output = env.communicate(command, timeout=120, check="ignore")
        try:
            encoded = env.read_file(encoded_path, encoding="utf-8", errors="strict")
        except Exception as e:
            self.logger.warning("Failed to read xiaoO trace db from %s: %s\n%s", trace_db_path, e, output)
            return None

        if not encoded.strip():
            self.logger.info("xiaoO trace db was not found or empty at %s", trace_db_path)
            return None

        trace_output_path = output_dir / self.config.trace_output_filename
        try:
            trace_output_path.write_bytes(base64.b64decode(encoded))
        except Exception as e:
            self.logger.warning("Failed to decode xiaoO trace db from %s: %s", trace_db_path, e)
            return None

        return str(trace_output_path)

    def get_trajectory_data(self) -> dict[str, Any]:
        return {
            "trajectory": self.trajectory,
            "info": self.info,
            "replay_config": None,
            "environment": "main",
        }

    def save_trajectory(self) -> None:
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(self.get_trajectory_data(), indent=2))

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.traj_path = output_dir / f"{problem_statement.id}.traj"
        repo_root = f"/{env.repo.repo_name}" if env.repo is not None else "/"

        self._chook.on_run_start()
        self._install_xiaoo(env)

        task = self._build_task(repo_root, problem_statement)
        env.write_file("/root/xiaoo_task.md", task)
        command = self._build_command(repo_root)

        t0 = time.perf_counter()
        observation = env.communicate(command, timeout=self.config.timeout, check="warn")
        execution_time = time.perf_counter() - t0

        submission = self._collect_submission(env, repo_root)
        trace_db_host_path = self._save_trace_db(env, output_dir)
        exit_status = "submitted" if submission else "no_submission"
        self.info = AgentInfo(
            submission=submission,
            exit_status=exit_status,
            model_stats={},
            swe_agent_hash=get_agent_commit_hash(),
            swe_agent_version=__version__,
            swe_rex_version=get_rex_version(),
            swe_rex_hash=get_rex_commit_hash(),
        )
        if trace_db_host_path is not None:
            self.info["xiaoo_trace_db"] = trace_db_host_path
        step: TrajectoryStep = {
            "action": command,
            "observation": observation,
            "response": observation,
            "state": {},
            "thought": "Ran xiaoO binary as an external SWE-bench agent.",
            "execution_time": execution_time,
            "query": [{"role": "user", "content": task}],
            "extra_info": {
                "binary_path": str(self.config.binary_path),
                "container_binary_path": self.config.container_binary_path,
                "xiaoo_trace_db": trace_db_host_path,
            },
        }
        self.trajectory = [step]
        self.save_trajectory()
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)
        return AgentRunResult(info=self.info, trajectory=self.trajectory)
