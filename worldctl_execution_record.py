"""Durable local observations, not signed execution provenance or a sandbox."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any
import uuid

from worldctl_fingerprint import digest, identity


STOP_REASONS = (
    "",
    "timeout",
    "cancelled",
    "log_budget",
    "execution_unavailable",
    "execution_record_unavailable",
    "artifact_record_unavailable",
)

TERMINAL_STOP_CODES = {
    "execution_not_started": 5,
    "execution_precondition_changed": 3,
    "execution_record_unavailable": 3,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sync_directory(folder: Path) -> None:
    """Persist the directory entry that names a new journal file.

    fsync on the file alone leaves the entry itself only in the page cache, so a
    crash can retain zero bytes while the run appears recorded. POSIX is the only
    platform where a directory can be opened and synced; bounded execution itself
    already refuses to start elsewhere.
    """
    if os.name != "posix":
        return
    fd = os.open(folder, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def step_outcome(step: dict[str, Any]) -> tuple[str, int, str]:
    """Single authority mapping one finished step to (status, exit_code, stop_code).

    Recording and inspection both call this so a recorded run cannot report a
    stop reason that disagrees with its own exit code.
    """
    reason = step["stop_reason"]
    if reason in {"timeout", "cancelled", "log_budget"}:
        return "BLOCKED", 5, reason
    if reason:
        return "BLOCKED", 3, reason
    if not step["cleanup_complete"]:
        return "BLOCKED", 3, "cleanup_incomplete"
    if step["child_exit_code"] != 0:
        return "FAIL", 4, "child_failed"
    return "PASS", 0, ""


class Journal:
    def __init__(self, folder: Path) -> None:
        self.path = folder / "execution.jsonl"
        self.run_id = uuid.uuid4().hex
        self.started_at = utc_now()
        self.clock = time.monotonic()
        self.previous: str | None = None
        self.sequence = 0
        self.healthy = True
        self.path.touch(exist_ok=False)
        try:
            sync_directory(folder)
        except OSError:
            self.healthy = False
            raise

    def emit(self, event: str, **data: Any) -> bool:
        if not self.healthy:
            return False
        record = dict(
            schema_version=1,
            run_id=self.run_id,
            seq=self.sequence,
            event=event,
            recorded_at=utc_now(),
            elapsed_seconds=round(time.monotonic() - self.clock, 6),
            previous_sha256=self.previous,
            data=data,
        )
        sha = digest(record)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {**record, "sha256": sha}, ensure_ascii=True, allow_nan=False
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            self.healthy = False
            return False
        self.previous = sha
        self.sequence += 1
        return True


def snapshot(path: Path, budget: int) -> dict[str, Any]:
    result: dict[str, Any] = dict(
        name=path.name, status="unavailable", bytes=None, sha256=None
    )
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            return result
        result["bytes"] = before.st_size
        if before.st_size > budget:
            return {**result, "status": "oversize"}
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(fd, "rb") as handle:
            if identity(os.fstat(handle.fileno())) != identity(before):
                return result
            raw = handle.read(budget + 1)
            if len(raw) > budget or identity(os.fstat(handle.fileno())) != identity(
                before
            ):
                return result
        if identity(path.lstat()) != identity(before):
            return result
        return {
            **result,
            "status": "complete",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    except OSError:
        return result
