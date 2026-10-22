"""Read-only, bounded device snapshot using the frozen cleanliness verdict."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import io
import os
from pathlib import Path
import subprocess
import time
from typing import Any

from worldctl import CLIError, ROOT

GATE_SHA256 = "cae58ecb8c06cc132bbc1a51c0b81489dc3cb6f15b41204d2335ae09e1ece7bb"


def query(fields: str, deadline: float) -> list[list[str]]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired("nvidia-smi", 0)
    result = subprocess.run(
        ["nvidia-smi", fields, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
        timeout=remaining,
        env={
            key: value
            for key, value in os.environ.items()
            if key in {"PATH", "SystemRoot", "SYSTEMROOT", "WINDIR"}
        },
    )
    if time.monotonic() > deadline:
        raise subprocess.TimeoutExpired("nvidia-smi", 0)
    if len(result.stdout) > 65536:
        raise ValueError("oversize telemetry")
    return [
        [field.strip() for field in row]
        for row in csv.reader(io.StringIO(result.stdout))
        if row
    ]


def check(root: Path, options: argparse.Namespace) -> dict[str, Any]:
    if (
        options.gpu_command != "check"
        or options.timeout is None
        or not 1 <= options.timeout <= 10
    ):
        raise CLIError(
            2,
            "invalid_gpu_check",
            "Select gpu check with an explicit 1-10 second total budget.",
        )
    if root != ROOT:
        raise CLIError(
            3,
            "gpu_check_root_mismatch",
            "Use the actual CLI checkout for the frozen gate.",
        )
    source = root / "gpu_cleanliness.py"
    if (
        source.is_symlink()
        or source.stat().st_size > 65536
        or hashlib.sha256(source.read_bytes()).hexdigest() != GATE_SHA256
    ):
        raise CLIError(
            3,
            "gpu_gate_source_changed",
            "The frozen GPU gate source must match its reviewed hash.",
        )
    gate = importlib.import_module("gpu_cleanliness")
    if gate.__file__ is None or Path(gate.__file__).resolve() != source.resolve():
        raise CLIError(
            3,
            "gpu_gate_module_mismatch",
            "The imported GPU gate must be the reviewed local source.",
        )
    thresholds = dict(
        max_unattributed_mib=gate.MAX_UNATTRIBUTED_MIB, min_free_mib=gate.MIN_FREE_MIB
    )
    data: dict[str, Any] = dict(
        status="PASS",
        exit_code=0,
        errors=[],
        thresholds=thresholds,
        snapshot_only=True,
        valid_for_future_execution=False,
        runtime_started=False,
        execution_authorized=False,
        release_qualified=False,
        gpu_queried=False,
        total_budget_seconds=options.timeout,
        clean=None,
    )
    if getattr(options, "dry_run", False):
        return data
    deadline = time.monotonic() + options.timeout
    try:
        rows = query(
            "--query-gpu=uuid,name,memory.total,memory.used,memory.free", deadline
        )
        if len(rows) != 1 or len(rows[0]) != 5:
            raise ValueError("exactly one GPU required")
        uuid, name, total, used, free = rows[0]
        if not uuid.startswith("GPU-") or not name:
            raise ValueError("invalid device identity")
        total_i, used_i, free_i = int(total), int(used), int(free)
        if total_i <= 0 or min(used_i, free_i) < 0 or used_i + free_i > total_i + 1:
            raise ValueError("invalid device memory")
        apps = query("--query-compute-apps=gpu_uuid,used_memory", deadline)
        attributed = 0
        for row in apps:
            if len(row) != 2 or row[0] != uuid:
                raise ValueError("ambiguous process attribution")
            value = int(row[1])
            if value < 0:
                raise ValueError("invalid process memory")
            attributed += value
        if attributed > used_i:
            raise ValueError("inconsistent snapshots")
        record = dict(
            ok=True,
            sampled_at=time.time(),
            uuid=uuid,
            name=name,
            total_mib=total_i,
            used_mib=used_i,
            free_mib=free_i,
            wsl_attributed_mib=attributed,
            unattributed_mib=used_i - attributed,
            thresholds=thresholds,
        )
        clean, reasons, _ = gate.verdict(record)
        data.update(
            gpu_queried=True,
            clean=clean,
            sample=record,
            reasons=reasons,
            status="PASS" if clean else "BLOCKED",
            exit_code=0 if clean else 3,
            errors=[]
            if clean
            else [
                dict(
                    code="gpu_not_clean",
                    message="The fixed GPU cleanliness gate refused this snapshot.",
                )
            ],
        )
    except subprocess.TimeoutExpired:
        data.update(
            status="BLOCKED",
            exit_code=5,
            gpu_queried=True,
            errors=[
                dict(
                    code="gpu_query_timeout",
                    message="The shared device-query budget expired.",
                )
            ],
        )
    except (OSError, ValueError, subprocess.CalledProcessError, csv.Error):
        data.update(
            status="BLOCKED",
            exit_code=3,
            gpu_queried=True,
            errors=[
                dict(
                    code="gpu_telemetry_unavailable",
                    message="Device or process attribution could not be verified; raw tool errors are not echoed.",
                )
            ],
        )
    return data
