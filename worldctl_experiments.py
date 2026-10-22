"""Static plans for named historical scripts; no deserialization or execution."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys
from typing import Any

from worldctl import CLIError, check_output

REGISTRY = {
    "preview-handoff": dict(
        script="preview_2a_handoff.py",
        sha256="20b58e05d6ce6ae681f592196a56895fc2cf1f3c99980cb9d92fcf0d790c6156",
        reference="docs/PREVIEW_2A_HANDOFF.md",
        purpose="Audit transitions from previous authoritative frame to preview and next authoritative frame",
        side_effects=[
            "loads trusted holdout tensors and TAE weights",
            "uses CUDA",
            "writes preview_2a.json in the selected output directory",
        ],
    ),
    "memory-gate": dict(
        script="m0_memgate.py",
        sha256="0e1de074311eecce29158f2180c1e22ea8930217092f50092b25003a03c5865e",
        reference="docs/M0_MEMORY_GATE.md",
        purpose="Historical BF16 worst-case memory experiment, not the current deployment preset",
        side_effects=[
            "loads model, shared assets, TAE and trusted cached tensors",
            "uses CUDA for 65 chunks",
            "writes m0.json in the selected output directory",
        ],
    ),
}


def entry(root: Path, name: str) -> dict[str, Any]:
    spec = REGISTRY[name]
    path = root / str(spec["script"])
    available = False
    digest = None
    if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root):
        with path.open("rb") as handle:
            raw = handle.read(1024 * 1024 + 1)
        if len(raw) <= 1024 * 1024:
            digest = hashlib.sha256(raw).hexdigest()
            available = (
                digest == spec["sha256"] and (root / str(spec["reference"])).is_file()
            )
    return dict(
        id=name,
        **spec,
        observed_sha256=digest,
        source_matches=available,
        historical_reference_only=True,
        run_available=False,
    )


def catalogue(root: Path) -> dict[str, Any]:
    entries = [entry(root, name) for name in REGISTRY]
    complete = all(x["source_matches"] for x in entries)
    return dict(
        status="PASS" if complete else "PARTIAL",
        exit_code=0 if complete else 3,
        errors=[]
        if complete
        else [
            dict(
                code="registered_source_unavailable",
                message="Some registered sources or references are unavailable or changed.",
            )
        ],
        experiments=entries,
        runtime_started=False,
        network="not_requested",
        discovery_scope="explicit tracked-source allowlist only; untracked experiments excluded",
    )


def validate_input(path: Path | None, *, directory: bool) -> Path:
    if (
        path is None
        or not (path.is_dir() if directory else path.is_file())
        or path.is_symlink()
    ):
        raise CLIError(
            3,
            "experiment_input_unavailable",
            "Explicit existing experiment inputs are required; no data is loaded.",
        )
    return path.resolve()


def plan(root: Path, options: argparse.Namespace) -> dict[str, Any]:
    if (
        options.experiment not in REGISTRY
        or options.timeout is None
        or not 1 <= options.timeout <= 3600
        or options.out_dir is None
    ):
        raise CLIError(
            2,
            "invalid_experiment_plan",
            "Select a registered experiment, new output directory and explicit 1-3600 second budget.",
        )
    source = entry(root, options.experiment)
    if not source["source_matches"]:
        raise CLIError(
            3,
            "experiment_source_changed",
            "Registered script/reference must match its reviewed source.",
        )
    check_output(options.out_dir)
    tae = validate_input(options.tae, directory=False)
    argv = [
        sys.executable,
        str(root / str(source["script"])),
        "--tae_pth",
        str(tae),
        "--out_dir",
        str(options.out_dir.absolute()),
    ]
    configuration: dict[str, Any]
    if options.experiment == "preview-handoff":
        if (
            any(
                x is not None
                for x in (
                    options.weights,
                    options.assets,
                    options.base,
                    options.y_cache,
                )
            )
            or not 1 <= len(options.holdout) <= 8
        ):
            raise CLIError(
                2,
                "invalid_preview_inputs",
                "Preview requires 1-8 tag=holdout paths and no memory-experiment arguments.",
            )
        holdouts = []
        for value in options.holdout:
            tag, separator, name = value.partition("=")
            if not separator or re.fullmatch(r"[A-Za-z0-9_-]{1,32}", tag) is None:
                raise CLIError(
                    2, "invalid_holdout", "Holdout syntax is a bounded tag=path."
                )
            path = validate_input(Path(name), directory=False)
            holdouts.append(f"{tag}={path}")
        if len({x.split("=", 1)[0] for x in holdouts}) != len(holdouts):
            raise CLIError(2, "duplicate_holdout_tag", "Holdout tags must be unique.")
        argv.extend(["--holdouts", *holdouts])
        configuration = dict(
            holdouts=holdouts,
            model_weights="TAE only",
            seed="recorded input provenance not inspected",
        )
    else:
        if options.holdout:
            raise CLIError(
                2,
                "invalid_memory_inputs",
                "Memory experiment does not accept preview holdouts.",
            )
        for flag, path, directory in [
            ("--ckpt_dir", options.weights, True),
            ("--assets_dir", options.assets, True),
            ("--base", options.base, True),
            ("--y_cache", options.y_cache, False),
        ]:
            argv.extend([flag, str(validate_input(path, directory=directory))])
        argv.extend(
            [
                "--weight",
                "bf16",
                "--task",
                "i2v-1.3B",
                "--area",
                "512x320",
                "--seed",
                "42",
                "--chunks",
                "65",
                "--chunk_size",
                "3",
                "--preset",
                "temporal_stability_8_2",
            ]
        )
        configuration = dict(
            weight="bf16",
            task="i2v-1.3B",
            original_area_argument="512x320",
            seed=42,
            chunks=65,
            chunk_size=3,
            preset="temporal_stability_8_2",
        )
    return dict(
        status="PASS",
        exit_code=0,
        errors=[],
        experiment=source,
        argv=argv,
        cwd=str(root),
        configuration=configuration,
        execution_budget_seconds=options.timeout,
        budget_enforced=False,
        plan_only=True,
        run_available=False,
        execution_authorized=False,
        runtime_started=False,
        runtime_ready=None,
        network="not_requested",
        release_qualified=False,
        requires_trusted_inputs=True,
        unresolved_prerequisites=[
            "original GPU cleanliness gate",
            "model/dependency compatibility",
            "explicit execution authorization",
            "GPU-enabled executor integration not implemented",
        ],
        output_directory_created=False,
    )
