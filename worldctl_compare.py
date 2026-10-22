"""Compare bounded self-reported snapshots, without granting research authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from worldctl import CLIError
from worldctl_experiments import REGISTRY
from worldctl_fingerprint import digest, identity


def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def hex_string(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None
    )


def relative(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 4096
        and "\\" not in value
        and "\0" not in value
        and not PurePosixPath(value).is_absolute()
        and str(PurePosixPath(value)) == value
        and ".." not in PurePosixPath(value).parts
    )


def validate(payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != {
        "experiment",
        "script_sha256",
        "source_tree",
        "configuration",
        "packages",
        "budget_seconds",
        "inputs",
    }:
        return False
    name = payload["experiment"]
    if (
        not isinstance(name, str)
        or name not in REGISTRY
        or payload["script_sha256"] != REGISTRY[name]["sha256"]
        or not hex_string(payload["source_tree"], 40)
    ):
        return False
    budget = payload["budget_seconds"]
    if type(budget) is not int or not 1 <= budget <= 3600:
        return False
    packages = payload["packages"]
    expected_packages = (
        {"torch"}
        if name == "preview-handoff"
        else {
            "torch",
            "torchvision",
            "numpy",
            "transformers",
            "diffusers",
            "flash-attn",
            "Pillow",
            "einops",
        }
    )
    if (
        not isinstance(packages, dict)
        or set(packages) != expected_packages
        or not all(isinstance(v, str) and 1 <= len(v) <= 256 for v in packages.values())
    ):
        return False
    configuration = payload["configuration"]
    if name == "preview-handoff":
        if not isinstance(configuration, dict) or set(configuration) != {
            "holdouts",
            "model_weights",
            "seed",
        }:
            return False
        tags = configuration["holdouts"]
        if (
            not isinstance(tags, list)
            or not 1 <= len(tags) <= 8
            or not all(
                isinstance(t, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,32}", t)
                for t in tags
            )
            or tags != sorted(set(tags))
        ):
            return False
        if (
            configuration["model_weights"] != "TAE only"
            or configuration["seed"] != "recorded input provenance not inspected"
        ):
            return False
        roles = {"tae", *("holdout:" + tag for tag in tags)}
    else:
        if configuration != dict(
            weight="bf16",
            task="i2v-1.3B",
            original_area_argument="512x320",
            seed=42,
            chunks=65,
            chunk_size=3,
            preset="temporal_stability_8_2",
        ):
            return False
        roles = {"tae", "weights", "assets", "base", "y-cache"}
    inputs = payload["inputs"]
    if not isinstance(inputs, list) or len(inputs) != len(roles):
        return False
    observed = []
    entries = 0
    for item in inputs:
        if (
            not isinstance(item, dict)
            or set(item) != {"role", "files", "directories"}
            or not isinstance(item["role"], str)
        ):
            return False
        observed.append(item["role"])
        files, directories = item["files"], item["directories"]
        if (
            not isinstance(files, list)
            or not isinstance(directories, list)
            or not all(relative(d) for d in directories)
            or directories != sorted(set(directories))
        ):
            return False
        paths = []
        for record in files:
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "bytes", "sha256"}
                or not relative(record["path"])
                or type(record["bytes"]) is not int
                or not 0 <= record["bytes"] <= 64 * 1024**3
                or not hex_string(record["sha256"], 64)
            ):
                return False
            paths.append(record["path"])
        if paths != sorted(set(paths)) or set(paths) & set(directories):
            return False
        if item["role"] in {"weights", "assets", "base"}:
            if (
                "." not in directories
                or (not files and len(directories) < 2)
                or any(
                    str(PurePosixPath(p).parent) not in directories
                    for p in [*paths, *directories]
                    if p != "."
                )
            ):
                return False
        elif directories or paths != ["."] or files[0]["bytes"] == 0:
            return False
        entries += len(files) + len(directories)
    return (
        observed == sorted(roles)
        and entries <= 4096
        and sum(record["bytes"] for item in inputs for record in item["files"])
        <= 64 * 1024**3
    )


def load(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise CLIError(
            2,
            "comparison_inputs_required",
            "Provide --left and --right fingerprint receipt files.",
        )
    try:
        if any(p.is_symlink() for p in (path.absolute(), *path.absolute().parents)):
            raise ValueError("symlink")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 64 * 1024**2:
            raise ValueError("not a bounded regular file")
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
        )
        with os.fdopen(fd, "rb") as handle:
            if identity(os.fstat(handle.fileno())) != identity(before):
                raise ValueError("file changed")
            raw = handle.read(64 * 1024**2 + 1)
            if len(raw) > 64 * 1024**2 or identity(
                os.fstat(handle.fileno())
            ) != identity(before):
                raise ValueError("file changed")
        if identity(path.lstat()) != identity(before):
            raise ValueError("file changed")
        envelope = json.loads(raw, object_pairs_hook=pairs)
        if (
            not isinstance(envelope, dict)
            or envelope.get("status") != "PASS"
            or type(envelope.get("exit_code")) is not int
            or envelope["exit_code"] != 0
            or envelope.get("command") != "experiment fingerprint"
        ):
            raise ValueError("not a successful fingerprint receipt")
        data = envelope["data"]
        if (
            not isinstance(data, dict)
            or data.get("fingerprint_created") is not True
            or data.get("input_contents_hashed") is not True
            or data.get("runtime_started") is not False
            or data.get("execution_authorized") is not False
            or data.get("provenance_verified") is not False
        ):
            raise ValueError("incomplete or contradictory receipt")
        record = data["fingerprint"]
        if (
            not isinstance(record, dict)
            or set(record) != {"schema_version", "sha256", "payload"}
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or not validate(record["payload"])
            or record["sha256"] != digest(record["payload"])
        ):
            raise ValueError("invalid schema or hash")
        return dict(record)
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        raise CLIError(
            3,
            "fingerprint_receipt_invalid",
            "Fingerprint receipt is missing, malformed, incomplete, changed or inconsistent; input content is not echoed.",
        ) from None


def compare(options: argparse.Namespace) -> dict[str, Any]:
    left, right = load(options.left), load(options.right)
    differing = [
        key for key in left["payload"] if left["payload"][key] != right["payload"][key]
    ]
    match = not differing
    return dict(
        status="PASS" if match else "FAIL",
        exit_code=0 if match else 4,
        errors=[]
        if match
        else [
            dict(
                code="fingerprint_conditions_differ",
                message="Recorded static experiment conditions differ.",
            )
        ],
        recorded_conditions_match=match,
        differing_fields=sorted(differing),
        left_sha256=left["sha256"],
        right_sha256=right["sha256"],
        receipt_hashes_validated=True,
        current_inputs_rechecked=False,
        provenance_verified=False,
        observed_outcomes_compared=False,
        runtime_started=False,
        execution_authorized=False,
        release_qualified=False,
        scope="recorded static conditions only; self-reported hashes do not authenticate origin or establish result quality",
    )
