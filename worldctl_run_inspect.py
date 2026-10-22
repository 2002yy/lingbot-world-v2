"""Validate bounded local execution journals and their adjacent logs; never replay."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import stat
from typing import Any, NoReturn

from worldctl import CLIError
from worldctl_compare import hex_string, pairs
from worldctl_execution_record import (
    STOP_REASONS,
    TERMINAL_STOP_CODES,
    snapshot,
    step_outcome,
)
from worldctl_fingerprint import digest, identity
from worldctl_result import number


def invalid() -> NoReturn:
    raise CLIError(
        3,
        "execution_record_incomplete",
        "Execution record is missing, incomplete, changed or unsupported; no successful completion can be established.",
    )


def timestamp(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 64:
        return False
    try:
        return datetime.fromisoformat(value).utcoffset() == timedelta(0)
    except ValueError:
        return False


def read_events(path: Path) -> list[dict[str, Any]]:
    try:
        if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
            invalid()
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 4 * 1024**2:
            invalid()
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(fd, "rb") as handle:
            if identity(os.fstat(handle.fileno())) != identity(before):
                invalid()
            raw = handle.read(4 * 1024**2 + 1)
            if len(raw) > 4 * 1024**2 or identity(
                os.fstat(handle.fileno())
            ) != identity(before):
                invalid()
        if identity(path.lstat()) != identity(before) or not raw.endswith(b"\n"):
            invalid()
        lines = raw.splitlines()
        if not 2 <= len(lines) <= 512:
            invalid()
        events = []
        previous = None
        run_id = None
        elapsed = 0.0
        keys = {
            "schema_version",
            "run_id",
            "seq",
            "event",
            "recorded_at",
            "elapsed_seconds",
            "previous_sha256",
            "data",
            "sha256",
        }
        for index, line in enumerate(lines):
            item = json.loads(line, object_pairs_hook=pairs)
            if (
                not isinstance(item, dict)
                or set(item) != keys
                or type(item["schema_version"]) is not int
                or item["schema_version"] != 1
                or type(item["seq"]) is not int
                or item["seq"] != index
                or not hex_string(item["run_id"], 32)
                or not timestamp(item["recorded_at"])
                or not number(item["elapsed_seconds"], low=elapsed)
                or not isinstance(item["data"], dict)
                or not isinstance(item["event"], str)
                or item["previous_sha256"] != previous
            ):
                invalid()
            if index == 0:
                run_id = item["run_id"]
            if (
                item["run_id"] != run_id
                or not hex_string(item["sha256"], 64)
                or item["sha256"]
                != digest({k: v for k, v in item.items() if k != "sha256"})
            ):
                invalid()
            elapsed = item["elapsed_seconds"]
            previous = item["sha256"]
            events.append(item)
        return events
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        invalid()


def inspect_run(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise CLIError(
            2, "execution_record_required", "Provide test inspect-run --record FILE."
        )
    events = read_events(path)
    if events[0]["event"] != "run_started" or events[-1]["event"] != "run_finished":
        invalid()
    start, end = events[0]["data"], events[-1]["data"]
    commands = start.get("commands")
    limit = start.get("log_limit")
    context = start.get("context")
    if (
        not isinstance(commands, list)
        or not 1 <= len(commands) <= 128
        or any(
            not isinstance(a, list)
            or not 1 <= len(a) <= 32
            or any(not isinstance(v, str) or len(v) > 4096 for v in a)
            for a in commands
        )
        or type(limit) is not int
        or not 1 <= limit <= 1048576
        or not number(start.get("budget_seconds"), low=0.000000001, high=3600)
        or not isinstance(context, dict)
        or set(context) - {"suite", "execution_head_sha", "source_baseline"}
        or not all(isinstance(v, str) and len(v) <= 256 for v in context.values())
        or start.get("origin_authenticated") is not False
        or start.get("release_qualified") is not False
    ):
        invalid()
    index = 1
    steps: list[dict[str, Any]] = []
    outcome = 0
    derived_status, derived_code, derived_stop = "PASS", 0, ""
    logs_complete = True
    while index < len(events) - 1:
        if outcome != 0 or len(steps) >= len(commands):
            invalid()
        event = events[index]
        data = event["data"]
        step_index = len(steps)
        if (
            event["event"] != "step_started"
            or type(data.get("index")) is not int
            or data["index"] != step_index
            or data.get("argv") != commands[step_index]
        ):
            invalid()
        index += 1
        pid = None
        if index < len(events) - 1 and events[index]["event"] == "child_started":
            data = events[index]["data"]
            if (
                type(data.get("index")) is not int
                or data["index"] != step_index
                or type(data.get("pid")) is not int
                or data["pid"] <= 0
                or type(data.get("session")) is not int
                or data["session"] != data["pid"]
            ):
                invalid()
            pid = data["pid"]
            index += 1
        if index >= len(events) - 1 or events[index]["event"] != "step_finished":
            invalid()
        data = events[index]["data"]
        step = data.get("record")
        artifacts = data.get("artifacts")
        if (
            type(data.get("index")) is not int
            or data["index"] != step_index
            or not isinstance(step, dict)
            or step.get("argv") != commands[step_index]
            or step.get("attempted") is not (pid is not None)
            or step.get("pid") != pid
            or type(step.get("cleanup_complete")) is not bool
            or not timestamp(step.get("started_at"))
            or not timestamp(step.get("ended_at"))
            or not number(step.get("duration_seconds"))
            or not isinstance(artifacts, list)
            or len(artifacts) != 2
        ):
            invalid()
        code, reason = step.get("child_exit_code"), step.get("stop_reason")
        if (
            (pid is None and code is not None)
            or (
                pid is not None
                and not (
                    (type(code) is int and -255 <= code <= 255)
                    or (code is None and not step["cleanup_complete"])
                )
            )
            or not isinstance(reason, str)
            or reason not in STOP_REASONS
            or (pid is None and not reason)
            or (pid is not None and type(step.get("pid")) is not int)
        ):
            invalid()
        for suffix, artifact in zip(("stdout", "stderr"), artifacts):
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"name", "status", "bytes", "sha256"}
                or artifact["name"] != f"{step_index}.{suffix}"
                or not isinstance(artifact["status"], str)
                or artifact["status"] not in {"complete", "oversize", "unavailable"}
            ):
                invalid()
            if artifact["status"] == "complete":
                if (
                    type(artifact["bytes"]) is not int
                    or not 0 <= artifact["bytes"] <= limit
                    or not hex_string(artifact["sha256"], 64)
                ):
                    invalid()
                observed = snapshot(path.parent / artifact["name"], limit)
                if observed["status"] == "unavailable":
                    raise CLIError(
                        3,
                        "execution_logs_unavailable",
                        "An adjacent recorded log is unavailable or unsafe to read.",
                    )
                if observed != artifact:
                    raise CLIError(
                        4,
                        "execution_logs_changed",
                        "Recorded log bytes differ from the stored snapshot.",
                    )
            else:
                logs_complete = False
                if artifact["sha256"] is not None or (
                    artifact["status"] == "oversize"
                    and (
                        type(artifact["bytes"]) is not int or artifact["bytes"] <= limit
                    )
                ):
                    invalid()
                if not reason and step["cleanup_complete"]:
                    invalid()
        if (
            all(a["status"] == "complete" for a in artifacts)
            and sum(a["bytes"] for a in artifacts) > limit
            and reason != "log_budget"
        ):
            invalid()
        derived_status, derived_code, derived_stop = step_outcome(step)
        if derived_status != "PASS":
            outcome = derived_code
        steps.append(step)
        index += 1
    status = end.get("status")
    exit_code = end.get("exit_code")
    if (
        not isinstance(end.get("stop_code"), str)
        or not isinstance(status, str)
        or status not in {"PASS", "FAIL", "BLOCKED"}
        or type(exit_code) is not int
        or {0: "PASS", 3: "BLOCKED", 4: "FAIL", 5: "BLOCKED"}.get(exit_code) != status
        or type(end.get("step_count")) is not int
        or end["step_count"] != len(steps)
        or end.get("execution_started") is not any(s["attempted"] for s in steps)
    ):
        invalid()
    if outcome:
        if (
            status != derived_status
            or exit_code != derived_code
            or end.get("stop_code") != derived_stop
        ):
            invalid()
    elif len(steps) == len(commands):
        if status != "PASS" or exit_code != 0 or end.get("stop_code") != "":
            invalid()
    else:
        terminal = end.get("stop_code")
        if status != "BLOCKED" or (
            not isinstance(terminal, str)
            or TERMINAL_STOP_CODES.get(terminal) != exit_code
        ):
            invalid()
    return dict(
        status=status,
        exit_code=exit_code,
        errors=[]
        if exit_code == 0
        else [
            dict(
                code="recorded_execution_not_successful",
                message="Record is internally complete but the recorded execution did not succeed.",
            )
        ],
        run_id=events[0]["run_id"],
        record_complete=True,
        record_chain_valid=True,
        recorded_exit_code=exit_code,
        recorded_steps=len(steps),
        recorded_execution_started=any(step["attempted"] for step in steps),
        child_outcomes=[
            {
                key: step[key]
                for key in ("pid", "child_exit_code", "stop_reason", "cleanup_complete")
            }
            for step in steps
        ],
        recorded_context=context,
        started_at=events[0]["recorded_at"],
        ended_at=events[-1]["recorded_at"],
        duration_seconds=events[-1]["elapsed_seconds"],
        log_hashes_verified=logs_complete,
        execution_origin_authenticated=False,
        run_binding_verified=False,
        provenance_verified=False,
        release_qualified=False,
        execution_authorized=False,
        runtime_started=False,
        scope="local record/log consistency only; commands are never replayed and saved observations do not authenticate execution provenance",
        artifacts=[],
    )
