"""Static experiment prerequisites; no tensor reads, model imports or GPU calls."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
from typing import Any

from worldctl import CLIError, ROOT, git
from worldctl_experiments import plan


def preflight(root: Path, options: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= len(options.purpose.strip()) <= 512:
        raise CLIError(
            2,
            "experiment_purpose_required",
            "Provide a 1-512 character experiment purpose.",
        )
    if root != ROOT or git(
        root, "status", "--porcelain=v1", "--untracked-files=normal"
    ):
        raise CLIError(
            3, "experiment_checkout_not_clean", "Use the actual clean CLI checkout."
        )
    if any(
        (p / ".env").exists() or (p / ".env").is_symlink()
        for p in (root, *root.parents)
    ):
        raise CLIError(
            3,
            "experiment_environment_unreviewed",
            "Ancestor .env files require separate review; no credentials are read.",
        )
    result = plan(root, options)
    inputs = [options.tae, *[Path(x.partition("=")[2]) for x in options.holdout]]
    inputs.extend(
        p
        for p in (options.weights, options.assets, options.base, options.y_cache)
        if p is not None
    )
    observed = []
    for path in inputs:
        assert path is not None
        resolved = path.resolve()
        if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
            raise CLIError(
                3, "experiment_input_symlink", "Input paths must not traverse symlinks."
            )
        stat = resolved.stat()
        if resolved.is_file() and stat.st_size == 0:
            raise CLIError(
                3, "experiment_input_empty", "Empty input files cannot pass preflight."
            )
        if resolved.is_dir() and next(resolved.iterdir(), None) is None:
            raise CLIError(
                3,
                "experiment_directory_empty",
                "Input directories must contain existing assets.",
            )
        observed.append(
            dict(
                path=str(resolved),
                kind="directory" if resolved.is_dir() else "file",
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    packages = (
        ("torch",)
        if options.experiment == "preview-handoff"
        else (
            "torch",
            "torchvision",
            "numpy",
            "transformers",
            "diffusers",
            "flash-attn",
            "Pillow",
            "einops",
        )
    )
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except (importlib.metadata.PackageNotFoundError, OSError, ValueError):
            versions[name] = None
    tae_module = Path.home() / "ai/taehv/taehv.py"
    checks = dict(
        trusted_inputs_acknowledged=bool(options.trusted_inputs),
        package_metadata_present=all(versions.values()),
        tae_source_present=tae_module.is_file() and not tae_module.is_symlink(),
    )
    passed = all(checks.values())
    result.update(
        status="PASS" if passed else "BLOCKED",
        exit_code=0 if passed else 3,
        errors=[]
        if passed
        else [
            dict(
                code="experiment_preflight_incomplete",
                message="Static prerequisites are incomplete; inspect checks.",
            )
        ],
        scope="static prerequisites only; PASS does not authorize execution",
        purpose=options.purpose.strip(),
        checks=checks,
        input_metadata=observed,
        package_versions=versions,
        provenance_verified=False,
        input_contents_loaded=False,
        gpu="not_queried",
        dependency_compatibility_verified=False,
        unresolved_prerequisites=[
            "live GPU cleanliness at execution time",
            "model/dependency compatibility",
            "trusted input provenance verification",
            "explicit execution authorization and GPU-enabled executor integration",
        ],
    )
    return result
