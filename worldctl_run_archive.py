"""Bounded, self-contained offline journal/log archives; no command replay."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, NoReturn

from worldctl import CLIError
from worldctl_compare import pairs
from worldctl_fingerprint import digest, identity
from worldctl_run_inspect import inspect_run
from worldctl_run_identity import match_identity, validate_expectations

MAX_BYTES = 5 * 1024**2
MAX_REPORT = 8 * 1024**2


def invalid() -> NoReturn:
    raise CLIError(
        3,
        "run_archive_invalid",
        "Archive inputs are unavailable, changed, incomplete or exceed the bounded archive format.",
    )


def read_blob(path: Path, limit: int) -> bytes:
    try:
        if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
            invalid()
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            invalid()
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(fd, "rb") as handle:
            if identity(os.fstat(handle.fileno())) != identity(before):
                invalid()
            raw = handle.read(limit + 1)
            if len(raw) > limit or identity(os.fstat(handle.fileno())) != identity(
                before
            ):
                invalid()
        if identity(path.lstat()) != identity(before):
            invalid()
        return raw
    except OSError:
        invalid()


def inspect_payload(archive: Any) -> dict[str, Any]:
    try:
        if (
            not isinstance(archive, dict)
            or set(archive) != {"schema_version", "kind", "files", "sha256"}
            or type(archive["schema_version"]) is not int
            or archive["schema_version"] != 1
            or archive["kind"] != "worldctl-run-archive"
            or not isinstance(archive["files"], list)
            or not 1 <= len(archive["files"]) <= 257
            or archive["sha256"]
            != digest({k: v for k, v in archive.items() if k != "sha256"})
        ):
            invalid()
        decoded: dict[str, bytes] = {}
        total = 0
        for item in archive["files"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"name", "bytes", "sha256", "base64"}
                or not isinstance(item["name"], str)
                or re.fullmatch(
                    r"execution\.jsonl|(?:0|[1-9][0-9]{0,2})\.(?:stdout|stderr)",
                    item["name"],
                )
                is None
                or item["name"] in decoded
                or type(item["bytes"]) is not int
                or not 0 <= item["bytes"] <= MAX_BYTES - total
                or not isinstance(item["base64"], str)
                or len(item["base64"]) > 4 * ((item["bytes"] + 2) // 3)
            ):
                invalid()
            raw = base64.b64decode(item["base64"], validate=True)
            if (
                len(raw) != item["bytes"]
                or hashlib.sha256(raw).hexdigest() != item["sha256"]
            ):
                invalid()
            decoded[item["name"]] = raw
            total += len(raw)
        if "execution.jsonl" not in decoded:
            invalid()
        with tempfile.TemporaryDirectory(
            prefix="worldctl-archive-inspect-"
        ) as directory:
            root = Path(directory).resolve()
            for name, raw in decoded.items():
                (root / name).write_bytes(raw)
            inspection = inspect_run(root / "execution.jsonl")
        expected = {"execution.jsonl"} | {
            f"{index}.{suffix}"
            for index in range(inspection["recorded_steps"])
            for suffix in ("stdout", "stderr")
        }
        if set(decoded) != expected or not inspection["log_hashes_verified"]:
            invalid()
        return dict(
            **inspection,
            archive_verified=True,
            archive_sha256=archive["sha256"],
            archived_files=len(decoded),
            archived_bytes=total,
        )
    except (KeyError, ValueError, TypeError, RecursionError, binascii.Error):
        invalid()


def archive_run(options: argparse.Namespace) -> dict[str, Any]:
    if options.record is None or (
        not getattr(options, "dry_run", False)
        and getattr(options, "output", None) is None
    ):
        raise CLIError(
            2,
            "run_archive_selection_required",
            "Provide --record and a new --output report, or use --dry-run.",
        )
    record = options.record
    first = read_blob(record, 4 * 1024**2)
    inspection = inspect_run(record)
    if not inspection["log_hashes_verified"]:
        invalid()
    files = []
    remaining = MAX_BYTES
    names = [
        "execution.jsonl",
        *(
            f"{index}.{suffix}"
            for index in range(inspection["recorded_steps"])
            for suffix in ("stdout", "stderr")
        ),
    ]
    for name in names:
        raw = (
            first
            if name == "execution.jsonl"
            else read_blob(record.parent / name, min(1024**2, remaining))
        )
        if len(raw) > remaining:
            invalid()
        remaining -= len(raw)
        files.append(
            dict(
                name=name,
                bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                base64=base64.b64encode(raw).decode("ascii"),
            )
        )
    if read_blob(record, 4 * 1024**2) != first:
        invalid()
    payload = dict(schema_version=1, kind="worldctl-run-archive", files=files)
    archive = dict(**payload, sha256=digest(payload))
    verified = inspect_payload(archive)
    verified["archive_created"] = not getattr(options, "dry_run", False)
    if verified["archive_created"]:
        verified["archive"] = archive
    return verified


def inspect_archive(
    path: Path | None,
    *,
    expected_head: str | None = None,
    expected_suite: str | None = None,
) -> dict[str, Any]:
    validate_expectations(expected_head, expected_suite)
    if path is None:
        raise CLIError(
            2,
            "run_archive_selection_required",
            "Provide --archive with a saved archive-run report.",
        )
    try:
        envelope = json.loads(read_blob(path, MAX_REPORT), object_pairs_hook=pairs)
        if (
            not isinstance(envelope, dict)
            or envelope.get("schema_version") != "agent-cli-result-v1"
            or envelope.get("command") != "test archive-run"
            or not isinstance(envelope.get("data"), dict)
            or envelope["data"].get("archive_created") is not True
        ):
            invalid()
        result = inspect_payload(envelope["data"]["archive"])
        # The embedded journal is authoritative for saved outcomes, not a header.
        if (
            envelope.get("status") != result["status"]
            or type(envelope.get("exit_code")) is not int
            or envelope["exit_code"] != result["exit_code"]
        ):
            invalid()
        return match_identity(
            result, expected_head=expected_head, expected_suite=expected_suite
        )
    except (KeyError, ValueError, TypeError, RecursionError):
        invalid()
