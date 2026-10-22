"""Bounded trusted child execution in an owned Linux session; not a sandbox."""

from __future__ import annotations

import math
from collections.abc import Callable
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from worldctl_execution_record import Journal, snapshot, utc_now


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
    before_step: Callable[[list[str]], bool] | None = None,
    record: bool = False,
    record_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    if (
        not math.isfinite(timeout)
        or not 0 < timeout <= 3600
        or not commands
        or log_limit <= 0
    ):
        raise ValueError("invalid execution budget or empty command set")
    if record and (
        log_limit > 1_048_576
        or len(commands) > 128
        or any(
            not argv
            or len(argv) > 32
            or any(not isinstance(arg, str) or len(arg) > 4096 for arg in argv)
            for argv in commands
        )
        or sum(len(arg.encode()) for argv in commands for arg in argv) > 65536
        or (
            record_context is not None
            and (
                set(record_context) - {"suite", "execution_head_sha", "source_baseline"}
                or any(
                    not isinstance(value, str) or len(value) > 256
                    for value in record_context.values()
                )
            )
        )
    ):
        raise ValueError("execution record bounds exceeded")
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
    journal = Journal(logs) if record else None
    if journal is not None:
        result.update(
            run_id=journal.run_id,
            execution_record=str(journal.path),
            execution_record_complete=False,
            recording_origin_authenticated=False,
        )
        if not journal.emit(
            "run_started",
            cwd=str(root),
            commands=commands,
            budget_seconds=timeout,
            log_limit=log_limit,
            context=record_context or {},
            origin_authenticated=False,
            release_qualified=False,
        ):
            result.update(
                status="BLOCKED",
                exit_code=3,
                execution_started=False,
                errors=[
                    dict(
                        code="execution_record_unavailable",
                        message="Could not record execution start; no child was launched.",
                    )
                ],
                artifacts=[str(journal.path)],
            )
            return result
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
        if before_step is not None and not before_step(argv):
            result.update(status="BLOCKED", exit_code=3)
            result["errors"] = [
                dict(
                    code="execution_precondition_changed",
                    message="The next trusted child no longer satisfies its admission checks.",
                )
            ]
            break
        stdout, stderr = logs / f"{index}.stdout", logs / f"{index}.stderr"
        reason = ""
        cleanup = True
        process = None
        step_clock = time.monotonic()
        started_at = utc_now()
        if journal is not None and not journal.emit(
            "step_started", index=index, argv=argv
        ):
            result.update(
                status="BLOCKED",
                exit_code=3,
                errors=[
                    dict(
                        code="execution_record_unavailable",
                        message="Could not record the next step; no new child was launched.",
                    )
                ],
            )
            break
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                if journal is not None and time.monotonic() >= deadline:
                    raise TimeoutError
                process = subprocess.Popen(
                    argv,
                    cwd=root,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    start_new_session=True,
                )
                if journal is not None and not journal.emit(
                    "child_started", index=index, pid=process.pid, session=process.pid
                ):
                    raise OSError("record unavailable")
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
        except TimeoutError:
            reason = "timeout"
        except OSError:
            reason = (
                "execution_record_unavailable"
                if journal is not None and not journal.healthy
                else "execution_unavailable"
            )
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
        if journal is not None:
            step = result["steps"][-1]
            step.update(
                pid=process.pid if process is not None else None,
                started_at=started_at,
                ended_at=utc_now(),
                duration_seconds=round(time.monotonic() - step_clock, 6),
            )
            artifacts = [
                snapshot(path, log_limit)
                if cleanup
                else dict(name=path.name, status="unavailable", bytes=None, sha256=None)
                for path in (stdout, stderr)
            ]
            if not reason and any(item["status"] != "complete" for item in artifacts):
                reason = "artifact_record_unavailable"
                step["stop_reason"] = reason
            if not journal.emit(
                "step_finished", index=index, record=step, artifacts=artifacts
            ):
                reason = "execution_record_unavailable"
                step["stop_reason"] = reason
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
    result["execution_started"] = any(step["attempted"] for step in result["steps"])
    if journal is not None:
        result["artifacts"].append(str(journal.path))
        result["observed_execution_exit_code"] = result["exit_code"]
        result["execution_record_complete"] = journal.emit(
            "run_finished",
            status=result["status"],
            exit_code=result["exit_code"],
            step_count=len(result["steps"]),
            execution_started=result["execution_started"],
            stop_code=result["errors"][0]["code"] if result["errors"] else "",
        )
        if not result["execution_record_complete"]:
            result.update(
                status="BLOCKED",
                exit_code=3,
                errors=[
                    dict(
                        code="execution_record_unavailable",
                        message="Execution observations were not fully recorded; inspect retained child exits/logs.",
                    )
                ],
            )
    return result
