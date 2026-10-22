"""Read historical raw results without importing or executing experiment scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
from typing import Any, NoReturn

from worldctl import CLIError
from worldctl_experiments import entry
from worldctl_fingerprint import identity


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def legacy_constant(token: str) -> None:
    if token != "NaN":
        raise ValueError("infinite metric")
    return None


def read_result(path: Path) -> tuple[dict[str, Any], str]:
    try:
        if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
            raise ValueError("symlink")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 4 * 1024**2:
            raise ValueError("unbounded or nonregular result")
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        with os.fdopen(fd, "rb") as handle:
            if identity(os.fstat(handle.fileno())) != identity(before):
                raise ValueError("changed result")
            raw = handle.read(4 * 1024**2 + 1)
            if len(raw) > 4 * 1024**2 or identity(
                os.fstat(handle.fileno())
            ) != identity(before):
                raise ValueError("changed result")
        if identity(path.lstat()) != identity(before):
            raise ValueError("changed result")
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=legacy_constant)
        if not isinstance(value, dict):
            raise ValueError("result must be an object")
        return value, hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, RecursionError):
        raise CLIError(
            3,
            "raw_result_unavailable",
            "Result must be a stable, bounded regular JSON file; raw content is not echoed.",
        ) from None


def number(value: Any, *, low: float = 0, high: float = 1e12) -> bool:
    return type(value) in (float, int) and low <= value <= high and math.isfinite(value)


def same(actual: Any, expected: float) -> bool:
    return number(actual, low=-1e12) and math.isclose(
        actual, expected, rel_tol=1e-6, abs_tol=1e-6
    )


def invalid() -> NoReturn:
    raise CLIError(
        3,
        "raw_result_schema_invalid",
        "Required raw rows/configuration are missing, unsupported or invalid; no historical PASS can be established.",
    )


def memory_result(raw: dict[str, Any]) -> dict[str, Any]:
    expected = dict(
        weight="bf16", preset="temporal_stability_8_2", chunks=65, chunk_size=3
    )
    if any(
        type(raw.get(k)) is not type(v) or raw.get(k) != v for k, v in expected.items()
    ):
        invalid()
    rows, points = raw.get("rows"), raw.get("points")
    tags = [
        "1_after_model_load",
        "2_after_tae_load",
        "3_after_condition_ready",
        "4_kv_window_full",
        "5_after_long_loop",
    ]
    if (
        not isinstance(rows, list)
        or len(rows) != 65
        or not isinstance(points, list)
        or len(points) != len(tags)
    ):
        invalid()
    for point, tag in zip(points, tags):
        if (
            not isinstance(point, dict)
            or point.get("tag") != tag
            or not number(point.get("driver_total"), low=1, high=1e7)
        ):
            invalid()
    total = points[0]["driver_total"]
    if any(p["driver_total"] != total for p in points):
        invalid()
    for record in [*points, *rows]:
        if not isinstance(record, dict) or not all(
            number(record.get(k), high=total)
            for k in ("alloc", "reserved", "max_alloc", "max_reserved", "driver_free")
        ):
            invalid()
        if not (
            record["alloc"] <= record["reserved"] <= record["max_reserved"]
            and record["alloc"] <= record["max_alloc"] <= record["max_reserved"]
        ):
            invalid()
    for index, row in enumerate(rows):
        if (
            type(row.get("chunk")) is not int
            or row["chunk"] != index
            or not all(number(row.get(k)) for k in ("dit", "kv", "tae", "ttfnf"))
        ):
            invalid()
    q = len(rows) // 4

    def growth(key: str) -> float:
        first = statistics.mean(row[key] for row in rows[:q])
        last = statistics.mean(row[key] for row in rows[-q:])
        return float((last - first) / max(first, 1) * 100)

    peak = max(row["max_reserved"] for row in rows)
    metrics = dict(
        peak_max_reserved_mib=peak,
        margin_mib=total - peak,
        min_driver_free_mib=min(row["driver_free"] for row in rows),
        growth_alloc_pct=growth("alloc"),
        growth_reserved_pct=growth("reserved"),
        ttfnf_p50_ms=statistics.mean(row["ttfnf"] for row in rows[1:]) * 1000,
        ttfnf_worst_ms=max(row["ttfnf"] for row in rows) * 1000,
        dit_ms=statistics.mean(row["dit"] for row in rows[1:]) * 1000,
        kv_ms=statistics.mean(row["kv"] for row in rows[1:]) * 1000,
        tae_ms=statistics.mean(row["tae"] for row in rows[1:]) * 1000,
    )
    if raw.get("overall") not in ("PASS", "FAIL") or any(
        not number(raw.get(key), low=-1e12) for key in metrics
    ):
        invalid()
    mismatches = [
        key for key, value in metrics.items() if not same(raw.get(key), value)
    ]
    passed = (
        metrics["growth_alloc_pct"] < 2
        and metrics["growth_reserved_pct"] < 2
        and metrics["margin_mib"] >= 500
    )
    computed = "PASS" if passed else "FAIL"
    if raw.get("overall") != computed:
        mismatches.append("overall")
    okay = not mismatches and passed
    return dict(
        status="PASS" if okay else "FAIL",
        exit_code=0 if okay else 4,
        errors=[]
        if okay
        else [
            dict(
                code="memory_summary_inconsistent"
                if mismatches
                else "memory_numeric_gate_failed",
                message="Raw summary disagrees with its rows or the historical numeric criteria failed.",
            )
        ],
        historical_numeric_gate=computed,
        summary_consistent=not mismatches,
        inconsistent_fields=sorted(mismatches),
        recorded_chunks=len(rows),
        recomputed_metrics=metrics,
        ttfnf_label_semantics="legacy p50 field is arithmetic mean excluding warmup, not a percentile",
        ttfnf_median_ms=statistics.median(row["ttfnf"] for row in rows[1:]) * 1000,
        no_oom_independently_verified=False,
        missing_evidence=[
            "process exit and no-OOM observation",
            "original GPU cleanliness",
            "input/source/run identity binding",
            "geometry/task/seed not recorded in raw result",
            "latency acceptance not defined by this numeric gate",
        ],
    )


def inspect_result(root: Path, options: argparse.Namespace) -> dict[str, Any]:
    if options.experiment != "memory-gate" or options.result is None:
        raise CLIError(
            2,
            "invalid_result_selection",
            "Select experiment inspect --experiment memory-gate --result FILE.",
        )
    source = entry(root, options.experiment)
    if not source["source_matches"]:
        raise CLIError(
            3,
            "experiment_source_changed",
            "The registered historical writer/reference must match its reviewed source.",
        )
    raw, sha = read_result(options.result)
    result = memory_result(raw)
    result.update(
        experiment=options.experiment,
        writer_sha256=source["sha256"],
        raw_result_sha256=sha,
        scope="historical raw numeric evidence only; no deployment or process authority",
        process_completion_verified=False,
        provenance_verified=False,
        current_inputs_rechecked=False,
        runtime_started=False,
        execution_authorized=False,
        release_qualified=False,
        historical_reference_only=True,
        original_result_modified=False,
    )
    return result
