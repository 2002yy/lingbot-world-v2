"""Current-file verification never upgrades manual claims to execution evidence."""

import json
import unittest

import test_worldctl_association as manual
import test_worldctl_memory_result as memory
import test_worldctl_preview_result as preview
from worldctl_fingerprint import digest


class VerifyAssociationTests(unittest.TestCase):
    invoke = manual.AssociationTests.invoke

    def setUp(self):
        manual.AssociationTests.setUp(self)
        self.association = self.root.parent / "association.json"
        self.save(manual.AssociationTests.call(self))

    def save(self, result):
        self.association.write_text(json.dumps(result))

    def call(self, *extra):
        return self.invoke(
            "experiment",
            "verify-association",
            "--association",
            str(self.association),
            "--result",
            str(self.raw),
            "--fingerprint",
            str(self.fingerprint),
            *extra,
        )

    def test_matching_files_are_verified_without_run_authority(self):
        result = self.call()
        self.assertEqual(result["exit_code"], 0)
        data = result["data"]
        self.assertTrue(data["association_files_match"])
        self.assertTrue(data["supplied_files_rechecked"])
        for key in (
            "run_binding_verified",
            "provenance_verified",
            "results_comparable",
            "current_inputs_rechecked",
            "process_completion_verified",
            "release_qualified",
        ):
            self.assertFalse(data[key])

    def test_changed_but_valid_raw_bytes_are_detected(self):
        self.raw.write_text(json.dumps(preview.sample(), indent=2))
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertFalse(result["data"]["association_files_match"])
        self.assertIn("raw_result_sha256", result["data"]["differing_fields"])

    def test_replaced_valid_fingerprint_is_detected(self):
        receipt = json.loads(self.fingerprint.read_text())
        record = receipt["data"]["fingerprint"]
        record["payload"]["budget_seconds"] += 1
        record["sha256"] = digest(record["payload"])
        self.fingerprint.write_text(json.dumps(receipt))
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertIn("fingerprint_sha256", result["data"]["differing_fields"])

    def test_stale_hash_and_authority_claims_are_rejected(self):
        original = self.association.read_text()
        for field in ("payload", "authority"):
            receipt = json.loads(original)
            if field == "payload":
                receipt["data"]["association"]["payload"]["purpose"] = "changed"
            else:
                receipt["data"]["run_binding_verified"] = True
            self.save(receipt)
            self.assertEqual(self.call()["exit_code"], 3)

    def test_dry_run_association_is_not_a_complete_receipt(self):
        self.save(manual.AssociationTests.call(self, "--dry-run"))
        self.assertEqual(self.call()["exit_code"], 3)

    def test_saved_failure_remains_failure_even_when_files_match(self):
        manual.AssociationTests.use_memory(self)
        raw = memory.sample()
        for row in raw["rows"][-16:]:
            row["alloc"] = 102
        raw.update(growth_alloc_pct=2.0, overall="FAIL")
        self.raw.write_text(json.dumps(raw))
        self.save(manual.AssociationTests.call(self, "--experiment", "memory-gate"))
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertTrue(result["data"]["association_files_match"])

    def test_saved_unknown_metrics_remain_blocked(self):
        raw = preview.sample()
        raw["per_scene"]["scene"][0]["cos_AP_AF"] = None
        raw["cos"] = 0.9
        raw["start"]["cos"] = None
        self.raw.write_text(json.dumps(raw))
        self.save(manual.AssociationTests.call(self))
        result = self.call()
        self.assertEqual(result["exit_code"], 3)
        self.assertTrue(result["data"]["association_files_match"])

    def test_unknown_schema_and_bad_nested_types_block(self):
        original = self.association.read_text()
        for field in ("schema", "payload"):
            value = json.loads(original)
            if field == "schema":
                value["data"]["association"]["schema_version"] = 2
            else:
                value["data"]["association"]["payload"] = []
            self.save(value)
            self.assertEqual(self.call()["exit_code"], 3)

    def test_missing_inputs_are_parameter_errors(self):
        self.assertEqual(
            self.invoke("experiment", "verify-association")["exit_code"], 2
        )

    def test_configuration_numbers_are_not_coerced_during_verification(self):
        manual.AssociationTests.use_memory(self)
        result = manual.AssociationTests.call(self, "--experiment", "memory-gate")
        record = result["data"]["association"]
        record["payload"]["observed_configuration"]["chunk_size"] = 3.0
        record["sha256"] = digest(record["payload"])
        self.save(result)
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertIn("observed_configuration", result["data"]["differing_fields"])

    def test_symlink_association_is_rejected(self):
        linked = self.root.parent / "linked.json"
        linked.symlink_to(self.association)
        self.assertEqual(self.call("--association", str(linked))["exit_code"], 3)

    def test_receipt_contents_are_not_echoed_on_parse_failure(self):
        self.association.write_text('{"private":"secret","private":1}')
        result = self.call()
        self.assertEqual(result["exit_code"], 3)
        self.assertNotIn("secret", str(result))

    def test_verification_output_is_non_overwriting(self):
        output = self.root.parent / "verified.json"
        self.assertEqual(
            self.call("--output", str(output), "--dry-run")["exit_code"], 0
        )
        self.assertFalse(output.exists())
        self.assertEqual(self.call("--output", str(output))["exit_code"], 0)
        saved = output.read_bytes()
        self.assertEqual(self.call("--output", str(output))["exit_code"], 3)
        self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
