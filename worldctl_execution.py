"""Bounded trusted child execution in an owned Linux session; not a sandbox."""

from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any


def child_environment(profile: Path, root: Path) -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT", "WINDIR", "TMPDIR"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(
        HOME=str(profile),
        USERPROFILE=str(profile),
        PYTHONPATH=str(root),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        HF_HOME=str(profile / "hf"),
    )
    return env


def session_members(session: int) -> list[int]:
    members = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit():
            continue
        try:
            fields = (folder / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[3]) == session and fields[0] not in {"Z", "X"}:
                members.append(int(folder.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def stop_session(process: subprocess.Popen[bytes]) -> bool:
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 2
        kill_after = time.monotonic() + 0.25
        killed = False
        while session_members(process.pid):
            if time.monotonic() >= kill_after and not killed:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                killed = True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        process.wait(timeout=0.5)
        return True
    except (OSError, subprocess.TimeoutExpired):
        return False


def execute(
    root: Path,
    commands: list[list[str]],
    timeout: float,
    *,
    dry_run: bool = False,
    log_limit: int = 1_048_576,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    if (
        not math.isfinite(timeout)
        or not 0 < timeout <= 3600
        or not commands
        or log_limit <= 0
    ):
        raise ValueError("invalid execution budget or empty command set")
    result: dict[str, Any] = dict(
        status="PASS",
        exit_code=0,
        errors=[],
        artifacts=[],
        commands=commands,
        budget_seconds=timeout,
        cleanup_scope="owned Linux session only",
        steps=[],
        release_qualified=False,
    )
    if dry_run:
        return {**result, "dry_run": True, "execution_started": False}
    if not sys.platform.startswith("linux"):
        return {
            **result,
            "status": "BLOCKED",
            "exit_code": 3,
            "errors": [
                dict(
                    code="execution_platform_unsupported",
                    message="Bounded execution currently requires Linux/WSL.",
                )
            ],
        }
    start = time.monotonic()
    deadline = start + timeout
    logs = Path(tempfile.mkdtemp(prefix="worldctl-run-"))
    result["log_root"] = str(logs)
    profile = logs / "profile"
    profile.mkdir()
    env = child_environment(profile, root)
    for index, argv in enumerate(commands):
        if time.monotonic() >= deadline or (cancel is not None and cancel.is_set()):
            result.update(status="BLOCKED", exit_code=5)
            result["errors"] = [
                dict(
                    code="execution_not_started",
                    message="Budget exhausted or cancellation requested before the next child.",
                )
            ]
            break
        stdout, stderr = logs / f"{index}.stdout", logs / f"{index}.stderr"
        reason = ""
        cleanup = True
        process = None
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                process = subprocess.Popen(
                    argv,
                    cwd=root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                while True:
                    if stdout.stat().st_size + stderr.stat().st_size > log_limit:
                        reason = "log_budget"
                        break
                    if time.monotonic() >= deadline:
                        reason = "timeout"
                        break
                    if cancel is not None and cancel.is_set():
                        reason = "cancelled"
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(0.01)
                if not reason and time.monotonic() >= deadline:
                    reason = "timeout"
        except KeyboardInterrupt:
            reason = "cancelled"
        except OSError:
            reason = "execution_unavailable"
        finally:
            if process is not None:
                cleanup = stop_session(process)
        code = process.returncode if process is not None else None
        if (
            not reason
            and sum(p.stat().st_size for p in (stdout, stderr) if p.exists())
            > log_limit
        ):
            reason = "log_budget"
        result["steps"].append(
            dict(
                argv=argv,
                child_exit_code=code,
                attempted=process is not None,
                stop_reason=reason,
                cleanup_complete=cleanup,
            )
        )
        result["artifacts"].extend([str(stdout), str(stderr)])
        remaining = log_limit
        for path in (stdout, stderr):
            if path.exists():
                with path.open("rb") as handle:
                    raw = handle.read(remaining)
                remaining -= len(raw)
                sys.stderr.write(raw.decode("utf-8", errors="replace"))
        if reason or not cleanup or code != 0:
            result.update(
                status="BLOCKED" if reason or not cleanup else "FAIL",
                exit_code=5
                if reason in {"timeout", "cancelled", "log_budget"}
                else (3 if reason or not cleanup else 4),
            )
            result["errors"] = [
                dict(
                    code=reason
                    or ("cleanup_incomplete" if not cleanup else "child_failed"),
                    message="Child execution did not complete successfully; inspect the raw exit and logs.",
                )
            ]
            break
    result["duration_seconds"] = round(time.monotonic() - start, 3)
    result["execution_started"] = bool(result["steps"])
    return result
