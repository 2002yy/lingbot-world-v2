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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
