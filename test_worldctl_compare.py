"""Receipt comparison verifies static identity, never observed outcomes."""

import copy
import json
import unittest

import test_worldctl_fingerprint as snapshots
from worldctl_fingerprint import digest


class CompareTests(unittest.TestCase):
    invoke = snapshots.FingerprintTests.invoke

    def setUp(self):
        snapshots.FingerprintTests.setUp(self)
        self.receipt = snapshots.FingerprintTests.call(self)
        self.left = self.root.parent / "left.json"
        self.right = self.root.parent / "right.json"
        self.write(self.left, self.receipt)
        self.write(self.right, self.receipt)

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def call(self, *extra):
        return self.invoke(
            "experiment",
            "compare",
            "--left",
            str(self.left),
            "--right",
            str(self.right),
            *extra,
        )

    def test_identical_static_conditions_do_not_grant_result_authority(self):
        result = self.call()
        self.assertEqual(result["exit_code"], 0)
        data = result["data"]
        self.assertTrue(data["recorded_conditions_match"])
        for key in (
            "current_inputs_rechecked",
            "provenance_verified",
            "observed_outcomes_compared",
            "execution_authorized",
            "runtime_started",
            "release_qualified",
        ):
            self.assertFalse(data[key])

    def test_different_budget_is_reported(self):
        value = copy.deepcopy(self.receipt)
        record = value["data"]["fingerprint"]
        record["payload"]["budget_seconds"] = 300
        record["sha256"] = digest(record["payload"])
        self.write(self.right, value)
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["data"]["differing_fields"], ["budget_seconds"])

    def test_tampered_payload_without_new_hash_is_rejected(self):
        self.receipt["data"]["fingerprint"]["payload"]["budget_seconds"] = 300
        self.write(self.right, self.receipt)
        self.assertEqual(self.call()["exit_code"], 3)

    def test_duplicate_json_keys_and_bad_json_are_rejected_without_echo(self):
        for raw in ('{"private":"secret","private":2}', "secret-not-json"):
            self.right.write_text(raw)
            result = self.call()
            self.assertEqual(result["exit_code"], 3)
            self.assertNotIn("secret", str(result))

    def test_partial_or_execution_claiming_receipt_is_rejected(self):
        for key, value in (
            ("fingerprint_created", False),
            ("runtime_started", True),
            ("execution_authorized", True),
            ("provenance_verified", True),
        ):
            receipt = copy.deepcopy(self.receipt)
            receipt["data"][key] = value
            self.write(self.right, receipt)
            self.assertEqual(self.call()["exit_code"], 3)

    def test_unknown_schema_and_duplicate_roles_are_rejected(self):
        for mode in ("schema", "roles", "traversal", "bool-budget"):
            value = copy.deepcopy(self.receipt)
            record = value["data"]["fingerprint"]
            if mode == "schema":
                record["schema_version"] = 2
            elif mode == "roles":
                record["payload"]["inputs"][1]["role"] = record["payload"]["inputs"][0][
                    "role"
                ]
            elif mode == "traversal":
                record["payload"]["inputs"][0]["files"][0]["path"] = "../secret"
            else:
                record["payload"]["budget_seconds"] = True
            record["sha256"] = digest(record["payload"])
            self.write(self.right, value)
            self.assertEqual(self.call()["exit_code"], 3)

    def test_symlink_and_oversize_receipts_are_rejected(self):
        self.right.unlink()
        self.right.symlink_to(self.left)
        self.assertEqual(self.call()["exit_code"], 3)
        self.right.unlink()
        with self.right.open("wb") as handle:
            handle.truncate(64 * 1024**2 + 1)
        self.assertEqual(self.call()["exit_code"], 3)

    def test_missing_arguments_are_parameter_errors(self):
        self.assertEqual(self.invoke("experiment", "compare")["exit_code"], 2)

    def test_report_is_non_overwriting_and_dry_run_does_not_write(self):
        output = self.root.parent / "comparison.json"
        result = self.call("--output", str(output), "--dry-run")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(output.exists())
        self.assertEqual(self.call("--output", str(output))["exit_code"], 0)
        self.assertEqual(self.call("--output", str(output))["exit_code"], 3)

    def test_generated_memory_directory_receipt_is_supported(self):
        receipt = snapshots.FingerprintTests.memory(self)
        self.write(self.left, receipt)
        self.write(self.right, receipt)
        self.assertEqual(self.call()["exit_code"], 0)

    def test_malformed_nested_types_do_not_escape_as_tracebacks(self):
        for key, value in (("configuration", []), ("packages", []), ("inputs", [{}])):
            receipt = copy.deepcopy(self.receipt)
            receipt["data"]["fingerprint"]["payload"][key] = value
            self.write(self.right, receipt)
            self.assertEqual(self.call()["exit_code"], 3)


if __name__ == "__main__":
    unittest.main()
