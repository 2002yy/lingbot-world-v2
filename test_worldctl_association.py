"""Manual associations retain missing run evidence and original failures."""

import hashlib
import json
import unittest
from unittest.mock import patch

import test_worldctl_fingerprint as snapshots
import test_worldctl_memory_result as memory
import test_worldctl_preview_result as preview
import worldctl_association as associations


class AssociationTests(unittest.TestCase):
    invoke = snapshots.FingerprintTests.invoke

    def setUp(self):
        snapshots.FingerprintTests.setUp(self)
        self.fingerprint = self.root.parent / "fingerprint.json"
        self.fingerprint.write_text(json.dumps(snapshots.FingerprintTests.call(self)))
        self.raw = self.root.parent / "raw.json"
        self.raw.write_text(json.dumps(preview.sample()))

    def call(self, *extra):
        return self.invoke(
            "experiment",
            "associate",
            "--experiment",
            "preview-handoff",
            "--result",
            str(self.raw),
            "--fingerprint",
            str(self.fingerprint),
            "--purpose",
            "Manual historic result association",
            *extra,
        )

    def use_memory(self):
        self.fingerprint.write_text(json.dumps(snapshots.FingerprintTests.memory(self)))
        self.raw.write_text(json.dumps(memory.sample()))

    def test_association_links_exact_bytes_without_claiming_execution(self):
        result = self.call()
        self.assertEqual(result["exit_code"], 0)
        data = result["data"]
        record = data["association"]
        self.assertEqual(
            record["payload"]["raw_result_sha256"],
            hashlib.sha256(self.raw.read_bytes()).hexdigest(),
        )
        self.assertEqual(record["sha256"], associations.digest(record["payload"]))
        for key in (
            "run_binding_verified",
            "provenance_verified",
            "process_completion_verified",
            "current_inputs_rechecked",
            "runtime_started",
            "release_qualified",
            "execution_authorized",
        ):
            self.assertFalse(data[key])
        self.assertTrue(record["payload"]["missing_run_evidence"])

    def test_memory_unrecorded_seed_and_geometry_are_not_invented(self):
        self.use_memory()
        result = self.call("--experiment", "memory-gate")
        self.assertEqual(result["exit_code"], 0)
        missing = result["data"]["association"]["payload"]["unobserved_configuration"]
        self.assertIn("seed", missing)
        self.assertIn("original_area_argument", missing)

    def test_different_experiments_cannot_be_associated(self):
        self.assertEqual(self.call("--experiment", "memory-gate")["exit_code"], 4)

    def test_different_scene_tags_are_rejected(self):
        raw = preview.sample()
        raw["per_scene"]["different"] = raw["per_scene"].pop("scene")
        self.raw.write_text(json.dumps(raw))
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertNotIn("association", result["data"])

    def test_failed_memory_criteria_remain_failure(self):
        self.use_memory()
        raw = memory.sample()
        for row in raw["rows"][-16:]:
            row["alloc"] = 102.0
        raw.update(growth_alloc_pct=2.0, overall="FAIL")
        self.raw.write_text(json.dumps(raw))
        result = self.call("--experiment", "memory-gate")
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(
            result["data"]["association"]["payload"]["inspection_exit_code"], 4
        )

    def test_partial_raw_record_cannot_create_association(self):
        self.raw.write_text('{"n":21}')
        self.assertEqual(self.call()["exit_code"], 3)

    def test_unknown_preview_metrics_remain_blocked_in_association(self):
        raw = preview.sample()
        raw["per_scene"]["scene"][0]["cos_AP_AF"] = None
        raw["cos"] = 0.9
        raw["start"]["cos"] = None
        self.raw.write_text(json.dumps(raw))
        result = self.call()
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(
            result["data"]["association"]["payload"]["inspection_exit_code"], 3
        )
        self.assertFalse(result["data"]["results_comparable"])

    def test_tampered_fingerprint_is_rejected(self):
        value = json.loads(self.fingerprint.read_text())
        value["data"]["fingerprint"]["payload"]["budget_seconds"] += 1
        self.fingerprint.write_text(json.dumps(value))
        self.assertEqual(self.call()["exit_code"], 3)

    def test_changed_result_during_association_blocks(self):
        with patch.object(
            associations, "read_result", return_value=(preview.sample(), "0" * 64)
        ):
            self.assertEqual(self.call()["exit_code"], 3)

    def test_purpose_and_input_selection_are_required(self):
        self.assertEqual(self.call("--purpose", " ")["exit_code"], 2)
        self.assertEqual(self.invoke("experiment", "associate")["exit_code"], 2)

    def test_dry_run_does_not_create_complete_record_or_file(self):
        output = self.root.parent / "report.json"
        result = self.call("--dry-run", "--output", str(output))
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["data"]["association_created"])
        self.assertNotIn("association", result["data"])
        self.assertFalse(output.exists())

    def test_report_preserves_inputs_and_never_overwrites(self):
        before = [p.read_bytes() for p in (self.raw, self.fingerprint)]
        output = self.root.parent / "report.json"
        self.assertEqual(self.call("--output", str(output))["exit_code"], 0)
        saved = output.read_bytes()
        self.assertEqual(self.call("--output", str(output))["exit_code"], 3)
        self.assertEqual(output.read_bytes(), saved)
        self.assertEqual(before, [p.read_bytes() for p in (self.raw, self.fingerprint)])


if __name__ == "__main__":
    unittest.main()
