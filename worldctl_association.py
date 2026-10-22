"""Explicit manual association, never retroactive execution provenance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from worldctl import CLIError
from worldctl_compare import load as load_fingerprint
from worldctl_fingerprint import digest
from worldctl_result import inspect_result, read_result


def build(
    root: Path, options: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        options.experiment not in {"memory-gate", "preview-handoff"}
        or options.result is None
        or options.fingerprint is None
        or not 1 <= len(options.purpose.strip()) <= 512
    ):
        raise CLIError(
            2,
            "association_selection_required",
            "Provide a registered experiment, result, fingerprint receipt and 1-512 character purpose.",
        )
    fingerprint = load_fingerprint(options.fingerprint)
    planned = fingerprint["payload"]
    if planned["experiment"] != options.experiment:
        raise CLIError(
            4,
            "association_experiment_mismatch",
            "Result selection and fingerprint experiment differ.",
        )
    inspection = inspect_result(root, options)
    raw, observed_sha = read_result(options.result)
    if observed_sha != inspection["raw_result_sha256"]:
        raise CLIError(
            3,
            "association_result_changed",
            "Result changed between inspection and association.",
        )
    if options.experiment == "memory-gate":
        observed = {
            key: raw[key] for key in ("weight", "preset", "chunks", "chunk_size")
        }
        declared = planned["configuration"]
        if any(
            type(declared.get(key)) is not type(value) or declared[key] != value
            for key, value in observed.items()
        ):
            raise CLIError(
                4,
                "association_configuration_mismatch",
                "Recorded result configuration differs from the fingerprint.",
            )
        missing = sorted(set(declared) - set(observed))
    else:
        observed = dict(holdout_tags=sorted(raw["per_scene"]))
        if observed["holdout_tags"] != planned["configuration"]["holdouts"]:
            raise CLIError(
                4,
                "association_scene_mismatch",
                "Result scenes differ from the fingerprint holdout tags.",
            )
        missing = ["input_tensor_contents", "seed"]
    payload = dict(
        kind="manual-association",
        experiment=options.experiment,
        purpose=options.purpose.strip(),
        raw_result_sha256=observed_sha,
        fingerprint_sha256=fingerprint["sha256"],
        registered_writer_sha256=inspection["writer_sha256"],
        observed_configuration=observed,
        unobserved_configuration=missing,
        inspection_status=inspection["status"],
        inspection_exit_code=inspection["exit_code"],
        run_binding_verified=False,
        missing_run_evidence=[
            "source tree at execution",
            "input hashes consumed by the process",
            "process exit and completion",
            "execution budget enforcement",
            "GPU cleanliness at execution",
        ],
    )
    return dict(schema_version=1, payload=payload, sha256=digest(payload)), inspection


def associate(root: Path, options: argparse.Namespace) -> dict[str, Any]:
    record, inspection = build(root, options)
    result = dict(
        status=inspection["status"],
        exit_code=inspection["exit_code"],
        errors=inspection["errors"],
        association_created=not getattr(options, "dry_run", False),
        association_kind="manual-association",
        run_binding_verified=False,
        provenance_verified=False,
        process_completion_verified=False,
        current_inputs_rechecked=False,
        results_comparable=False,
        runtime_started=False,
        execution_authorized=False,
        release_qualified=False,
        inspection=inspection,
        scope="manual file association only; no execution provenance is inferred",
    )
    if result["association_created"]:
        result["association"] = record
    return result
