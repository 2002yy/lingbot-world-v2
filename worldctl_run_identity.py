"""Match caller-selected run context without authenticating its saved origin."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from worldctl import CLIError
from worldctl_run_inspect import inspect_run


def validate_expectations(
    expected_head: str | None, expected_suite: str | None
) -> None:
    if (
        expected_head is not None
        and re.fullmatch(r"[0-9a-f]{40}", expected_head) is None
    ) or (
        expected_suite is not None
        and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", expected_suite) is None
    ):
        raise CLIError(
            2,
            "invalid_run_identity",
            "Use a full lowercase 40-character commit and a 1-64 character suite name.",
        )


def match_identity(
    result: dict[str, Any],
    *,
    expected_head: str | None = None,
    expected_suite: str | None = None,
) -> dict[str, Any]:
    validate_expectations(expected_head, expected_suite)
    requested = {
        key: value
        for key, value in {
            "execution_head_sha": expected_head,
            "suite": expected_suite,
        }.items()
        if value is not None
    }
    context = result["recorded_context"]
    missing = sorted(key for key in requested if not context.get(key))
    mismatched = sorted(
        key
        for key, value in requested.items()
        if key not in missing and context[key] != value
    )
    result.update(
        identity_expectations=requested,
        identity_missing_fields=missing,
        identity_mismatched_fields=mismatched,
        identity_check_requested=bool(requested),
        recorded_identity_matches=(not missing and not mismatched)
        if requested
        else None,
        identity_origin_authenticated=False,
    )
    if missing or mismatched:
        result.update(
            status="BLOCKED" if missing else "FAIL",
            exit_code=3 if missing else 4,
            errors=[
                dict(
                    code="run_identity_unavailable"
                    if missing
                    else "run_identity_mismatch",
                    message="Saved run context is missing or differs from the caller's expected identity; the recorded child outcome is retained separately.",
                )
            ],
        )
    return result


def inspect_identity(
    record: Path | None,
    *,
    expected_head: str | None = None,
    expected_suite: str | None = None,
) -> dict[str, Any]:
    validate_expectations(expected_head, expected_suite)
    return match_identity(
        inspect_run(record), expected_head=expected_head, expected_suite=expected_suite
    )
