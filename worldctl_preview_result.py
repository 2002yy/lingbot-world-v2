"""Inspect legacy preview summaries; unavailable direction is never zero errors."""

from __future__ import annotations

import statistics
from typing import Any

from worldctl_result import invalid, number, same


def preview_result(raw: dict[str, Any]) -> dict[str, Any]:
    scenes = raw.get("per_scene")
    if not isinstance(scenes, dict) or not 1 <= len(scenes) <= 8:
        invalid()
    all_rows: list[dict[str, Any]] = []
    mismatches = []
    unknown = dict(progress_ratio=0, correction_ratio=0, cos_AP_AF=0)
    for tag, rows in scenes.items():
        if (
            not isinstance(tag, str)
            or not 1 <= len(tag) <= 128
            or not isinstance(rows, list)
            or not rows
            or len(all_rows) + len(rows) > 4096
        ):
            invalid()
        previous = None
        for index, row in enumerate(rows):
            if (
                not isinstance(row, dict)
                or type(row.get("chunk")) is not int
                or type(row.get("prev")) is not int
                or not 0 <= row["prev"] < row["chunk"]
                or (previous is not None and row["prev"] != previous)
            ):
                invalid()
            previous = row["chunk"]
            if not all(
                number(row.get(k), high=1) for k in ("d_AF", "d_AP", "d_PF")
            ) or not all(
                number(row.get(k), low=-1, high=1)
                for k in ("ssim_AF", "ssim_AP", "ssim_PF", "e_AF", "e_AP", "e_PF")
            ):
                invalid()
            for key in unknown:
                if key not in row or (
                    row[key] is not None
                    and not number(
                        row[key],
                        low=-1.000001 if key == "cos_AP_AF" else 0,
                        high=1.000001 if key == "cos_AP_AF" else 1e12,
                    )
                ):
                    invalid()
                unknown[key] += int(row[key] is None)
            for key, distance in (
                ("progress_ratio", "d_AP"),
                ("correction_ratio", "d_PF"),
            ):
                expected = row[distance] / row["d_AF"] if row["d_AF"] > 1e-9 else None
                if not (
                    row[key] is None if expected is None else same(row[key], expected)
                ):
                    mismatches.append(f"{tag}[{index}].{key}")
            all_rows.append(row)
    n = len(all_rows)
    counts = dict(
        n=n,
        wrong_direction=sum(
            row["cos_AP_AF"] is not None and row["cos_AP_AF"] < 0 for row in all_rows
        ),
        overshoot=sum(
            row["correction_ratio"] is not None and row["correction_ratio"] > 1
            for row in all_rows
        ),
        gross_jumps=sum(row["ssim_PF"] < 0.30 for row in all_rows),
    )
    for key, value in counts.items():
        if type(raw.get(key)) is not int or raw[key] < 0:
            invalid()
        if raw[key] != value:
            mismatches.append(key)

    def aggregate(key: str, rows: list[dict[str, Any]] | None = None) -> float | None:
        values = [row[key] for row in (rows or all_rows) if row[key] is not None]
        return float(statistics.median(values)) if values else None

    def matches(actual: Any, expected: float | None) -> bool:
        return actual is None if expected is None else same(actual, expected)

    aggregate_fields = dict(
        progress_ratio=aggregate("progress_ratio"),
        correction_ratio=aggregate("correction_ratio"),
        cos=aggregate("cos_AP_AF"),
    )
    for key, value in aggregate_fields.items():
        if key not in raw:
            invalid()
        if not matches(raw[key], value):
            mismatches.append(key)
    groups = dict(
        start=[row for row in all_rows if row["chunk"] <= 2],
        steady=[row for row in all_rows if row["chunk"] > 2],
    )
    for group, rows in groups.items():
        observed = raw.get(group)
        if not isinstance(observed, dict):
            invalid()
        for key, source in (
            ("progress", "progress_ratio"),
            ("correction", "correction_ratio"),
            ("cos", "cos_AP_AF"),
        ):
            if key not in observed:
                invalid()
            if not matches(observed[key], aggregate(source, rows)):
                mismatches.append(group + "." + key)
    complete = not any(unknown.values())
    pr, cr = aggregate_fields["progress_ratio"], aggregate_fields["correction_ratio"]
    if not complete or pr is None or cr is None:
        recommendation = "insufficient_metrics"
    elif counts["wrong_direction"] == 0 and cr < 0.8 and pr < 1:
        recommendation = "legacy_direct_replacement_likely_sufficient"
    elif counts["wrong_direction"] == 0 and cr >= 0.8:
        recommendation = "legacy_blend_worth_testing"
    else:
        recommendation = "legacy_else_branch"
    okay = not mismatches and complete
    return dict(
        status="FAIL" if mismatches else ("PASS" if complete else "BLOCKED"),
        exit_code=4 if mismatches else (0 if complete else 3),
        errors=[]
        if okay
        else [
            dict(
                code="preview_summary_inconsistent"
                if mismatches
                else "preview_metrics_incomplete",
                message="Preview summaries disagree with rows or required metrics are unavailable.",
            )
        ],
        inspection_pass_semantics="raw row formulas and summaries agree; not a binary preview quality acceptance",
        summary_consistent=not mismatches,
        inconsistent_fields=sorted(mismatches),
        metrics_complete=complete,
        unknown_metric_counts=unknown,
        recomputed_counts=counts,
        recomputed_medians=aggregate_fields,
        per_scene_transitions={tag: len(rows) for tag, rows in scenes.items()},
        actual_group_counts={key: len(rows) for key, rows in groups.items()},
        legacy_empty_group_fallback=[key for key, rows in groups.items() if not rows],
        direction_assessment="incomplete"
        if unknown["cos_AP_AF"]
        else (
            "wrong_direction_recorded"
            if counts["wrong_direction"]
            else "no_wrong_direction_recorded"
        ),
        legacy_recommendation=recommendation,
        legacy_else_text_overstates_wrong_direction=recommendation
        == "legacy_else_branch"
        and counts["wrong_direction"] == 0,
        image_and_latent_metrics_independently_verified=False,
        missing_evidence=[
            "process exit",
            "input/source/run identity binding",
            "original images and latents not independently remeasured",
            "current deployment and human visual acceptance",
        ],
    )
