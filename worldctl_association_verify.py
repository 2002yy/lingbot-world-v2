"""Recheck manual association bytes without authenticating their claimed origin."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from worldctl import CLIError
from worldctl_association import build
from worldctl_compare import hex_string
from worldctl_experiments import REGISTRY
from worldctl_fingerprint import digest
from worldctl_result import read_result

PAYLOAD_FIELDS = {
    "kind",
    "experiment",
    "purpose",
    "raw_result_sha256",
    "fingerprint_sha256",
    "registered_writer_sha256",
    "observed_configuration",
    "unobserved_configuration",
    "inspection_status",
    "inspection_exit_code",
    "run_binding_verified",
    "missing_run_evidence",
}
AUTHORITY_FLAGS = (
    "run_binding_verified",
    "provenance_verified",
    "process_completion_verified",
    "current_inputs_rechecked",
    "results_comparable",
    "runtime_started",
    "execution_authorized",
    "release_qualified",
)
STATUSES = {"PASS": 0, "FAIL": 4, "BLOCKED": 3}


def load_association(path: Path) -> dict[str, Any]:
    envelope, _ = read_result(path)
    try:
        status = envelope.get("status")
        if (
            envelope.get("schema_version") != "agent-cli-result-v1"
            or envelope.get("command") != "experiment associate"
            or not isinstance(status, str)
            or status not in STATUSES
            or type(envelope.get("exit_code")) is not int
            or envelope["exit_code"] != STATUSES[status]
        ):
            raise ValueError("invalid envelope")
        data = envelope["data"]
        if (
            not isinstance(data, dict)
            or data.get("association_created") is not True
            or data.get("association_kind") != "manual-association"
            or any(data.get(key) is not False for key in AUTHORITY_FLAGS)
        ):
            raise ValueError("incomplete association or authority claim")
        record = data["association"]
        if (
            not isinstance(record, dict)
            or set(record) != {"schema_version", "payload", "sha256"}
            or type(record["schema_version"]) is not int
            or record["schema_version"] != 1
            or not hex_string(record["sha256"], 64)
        ):
            raise ValueError("unsupported record")
        payload = record["payload"]
        if (
            not isinstance(payload, dict)
            or set(payload) != PAYLOAD_FIELDS
            or payload["kind"] != "manual-association"
            or not isinstance(payload["experiment"], str)
            or payload["experiment"] not in REGISTRY
            or not isinstance(payload["purpose"], str)
            or not 1 <= len(payload["purpose"].strip()) <= 512
        ):
            raise ValueError("invalid payload")
        if (
            any(
                not hex_string(payload[key], 64)
                for key in (
                    "raw_result_sha256",
                    "fingerprint_sha256",
                    "registered_writer_sha256",
                )
            )
            or payload["registered_writer_sha256"]
            != REGISTRY[payload["experiment"]]["sha256"]
            or payload["run_binding_verified"] is not False
            or payload["inspection_status"] != status
            or type(payload["inspection_exit_code"]) is not int
            or payload["inspection_exit_code"] != STATUSES[status]
        ):
            raise ValueError("inconsistent payload")
        if record["sha256"] != digest(payload):
            raise ValueError("changed payload")
        return dict(record)
    except (KeyError, ValueError, TypeError, RecursionError):
        raise CLIError(
            3,
            "association_receipt_invalid",
            "Association receipt is unsupported, incomplete, changed or claims authority; raw content is not echoed.",
        ) from None


def verify_association(root: Path, options: argparse.Namespace) -> dict[str, Any]:
    if (
        options.association is None
        or options.result is None
        or options.fingerprint is None
    ):
        raise CLIError(
            2,
            "association_verification_inputs_required",
            "Provide --association, --result and --fingerprint receipt files.",
        )
    stored = load_association(options.association)
    payload = stored["payload"]
    current, inspection = build(
        root,
        argparse.Namespace(
            experiment=payload["experiment"],
            purpose=payload["purpose"],
            result=options.result,
            fingerprint=options.fingerprint,
        ),
    )
    match = stored["sha256"] == current["sha256"]
    differing = sorted(
        key
        for key in PAYLOAD_FIELDS
        if digest(payload[key]) != digest(current["payload"][key])
    )
    return dict(
        status=inspection["status"] if match else "FAIL",
        exit_code=inspection["exit_code"] if match else 4,
        errors=inspection["errors"]
        if match
        else [
            dict(
                code="association_files_changed",
                message="Supplied files or derived inspection differ from the stored manual association.",
            )
        ],
        association_files_match=match,
        differing_fields=differing,
        stored_association_sha256=stored["sha256"],
        current_association_sha256=current["sha256"],
        supplied_files_rechecked=True,
        inspection=inspection,
        **{key: False for key in AUTHORITY_FLAGS},
        scope="manual association/current-file consistency only; execution and input-consumption provenance remain unverified",
    )
