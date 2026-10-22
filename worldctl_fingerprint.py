"""Bounded content snapshots, never tensor deserialization or execution authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import time
from typing import Any

from worldctl import CLIError, git
from worldctl_preflight import preflight


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def input_roots(options: argparse.Namespace) -> list[tuple[str, Path]]:
    roots = [("tae", options.tae)]
    roots.extend(
        ("holdout:" + x.partition("=")[0], Path(x.partition("=")[2]))
        for x in options.holdout
    )
    roots.extend(
        (role, path)
        for role, path in [
            ("weights", options.weights),
            ("assets", options.assets),
            ("base", options.base),
            ("y-cache", options.y_cache),
        ]
        if path is not None
    )
    return sorted(roots)


def fingerprint(root: Path, options: argparse.Namespace) -> dict[str, Any]:
    if (
        options.max_bytes is None
        or not 1 <= options.max_bytes <= 64 * 1024**3
        or options.max_files is None
        or not 1 <= options.max_files <= 4096
        or options.hash_timeout is None
        or not 1 <= options.hash_timeout <= 60
    ):
        raise CLIError(
            2,
            "fingerprint_budget_required",
            "Provide max-bytes (1-64 GiB), max-files (1-4096), hash-timeout (1-60 seconds).",
        )
    result = preflight(root, options)
    if result["exit_code"] != 0:
        return result
    if getattr(options, "dry_run", False):
        result.update(fingerprint_created=False, input_contents_hashed=False)
        return result
    deadline = time.monotonic() + options.hash_timeout
    head = git(root, "rev-parse", "HEAD")
    tree = git(root, "rev-parse", "HEAD^{tree}")
    total = 0
    entries = 0
    snapshots: list[tuple[Path, tuple[int, ...]]] = []

    def tick() -> None:
        if time.monotonic() >= deadline:
            raise CLIError(
                5,
                "fingerprint_timeout",
                "Hash budget expired; no complete fingerprint was produced.",
            )

    def collect(
        path: Path,
        relative: str,
        files: list[tuple[Path, str]],
        directories: list[str],
        depth: int = 0,
    ) -> None:
        nonlocal entries
        tick()
        entries += 1
        if entries > options.max_files or depth > 32:
            raise CLIError(
                3,
                "fingerprint_entry_budget",
                "Input entries/depth exceed the declared bounds (directories count too).",
            )
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise CLIError(
                3, "fingerprint_symlink", "Input snapshots cannot contain symlinks."
            )
        snapshots.append((path, identity(before)))
        if stat.S_ISDIR(before.st_mode):
            directories.append(relative)
            with os.scandir(path) as children:
                for child in children:
                    collect(
                        Path(child.path),
                        child.name if relative == "." else relative + "/" + child.name,
                        files,
                        directories,
                        depth + 1,
                    )
        elif stat.S_ISREG(before.st_mode):
            files.append((path, relative))
        else:
            raise CLIError(
                3,
                "fingerprint_special_file",
                "Only regular files and directories are supported.",
            )

    inputs = []
    for role, path in input_roots(options):
        files: list[tuple[Path, str]] = []
        directories: list[str] = []
        collect(path, ".", files, directories)
        hashed = []
        for source, relative in sorted(files, key=lambda item: item[1]):
            tick()
            before = source.lstat()
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_size > options.max_bytes - total
            ):
                raise CLIError(
                    3,
                    "fingerprint_byte_budget",
                    "Inputs changed or exceed the total byte budget.",
                )
            fd = os.open(
                source,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            hasher = hashlib.sha256()
            count = 0
            with os.fdopen(fd, "rb") as handle:
                opened = os.fstat(handle.fileno())
                if not stat.S_ISREG(opened.st_mode) or identity(opened) != identity(
                    before
                ):
                    raise CLIError(
                        3,
                        "fingerprint_input_changed",
                        "Input changed while opening it.",
                    )
                while True:
                    tick()
                    block = handle.read(min(1024 * 1024, options.max_bytes - total + 1))
                    if not block:
                        break
                    total += len(block)
                    count += len(block)
                    if total > options.max_bytes:
                        raise CLIError(
                            3,
                            "fingerprint_byte_budget",
                            "Inputs exceed the total byte budget.",
                        )
                    hasher.update(block)
                if identity(os.fstat(handle.fileno())) != identity(before):
                    raise CLIError(
                        3,
                        "fingerprint_input_changed",
                        "Input changed while hashing it.",
                    )
            hashed.append(dict(path=relative, bytes=count, sha256=hasher.hexdigest()))
        inputs.append(dict(role=role, files=hashed, directories=sorted(directories)))
    tick()
    if any(identity(path.lstat()) != observed for path, observed in snapshots):
        raise CLIError(
            3, "fingerprint_input_changed", "Input snapshot changed before completion."
        )
    if git(root, "rev-parse", "HEAD") != head or git(
        root, "status", "--porcelain=v1", "--untracked-files=normal"
    ):
        raise CLIError(
            3, "fingerprint_checkout_changed", "Checkout changed during the snapshot."
        )
    configuration = dict(result["configuration"])
    if "holdouts" in configuration:
        configuration["holdouts"] = sorted(x.partition("=")[0] for x in options.holdout)
    payload = dict(
        experiment=options.experiment,
        script_sha256=result["experiment"]["sha256"],
        source_tree=tree,
        configuration=configuration,
        packages=result["package_versions"],
        budget_seconds=options.timeout,
        inputs=inputs,
    )
    result.update(
        fingerprint=dict(schema_version=1, payload=payload, sha256=digest(payload)),
        fingerprint_created=True,
        input_contents_hashed=True,
        input_contents_loaded=True,
        tensors_deserialized=False,
        hashed_bytes=total,
        snapshot_only=True,
        valid_for_future_execution=False,
        provenance_verified=False,
        scope="content snapshot only; hashes do not prove origin or authorize execution",
    )
    return result
