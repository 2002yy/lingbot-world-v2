"""Preview inspection preserves unknown metrics and negative findings."""

import json
import unittest

import test_worldctl_experiments as catalogue


def sample():
    base = dict(
        ssim_AF=0.7,
        ssim_AP=0.8,
        ssim_PF=0.9,
        e_AF=0.7,
        e_AP=0.8,
        e_PF=0.9,
        d_AF=0.2,
        d_AP=0.16,
        d_PF=0.1,
        progress_ratio=0.8,
        correction_ratio=0.5,
    )
    rows = [
        dict(chunk=1, prev=0, cos_AP_AF=0.8, **base),
        dict(chunk=3, prev=1, cos_AP_AF=0.9, **base),
    ]
    return dict(
        per_scene=dict(scene=rows),
        n=2,
        wrong_direction=0,
        overshoot=0,
        gross_jumps=0,
        progress_ratio=0.8,
        correction_ratio=0.5,
        cos=0.85,
        start=dict(progress=0.8, correction=0.5, cos=0.8),
        steady=dict(progress=0.8, correction=0.5, cos=0.9),
    )


class PreviewResultTests(unittest.TestCase):
    invoke = catalogue.ExperimentTests.invoke

    def setUp(self):
        catalogue.ExperimentTests.setUp(self)
        self.result = self.root.parent / "preview_2a.json"
        self.raw = sample()

    def call(self, *extra):
        self.result.write_text(json.dumps(self.raw))
        return self.invoke(
            "experiment",
            "inspect",
            "--experiment",
            "preview-handoff",
            "--result",
            str(self.result),
            *extra,
        )

    def test_finite_consistent_evidence_is_not_quality_authority(self):
        result = self.call()
        self.assertEqual(result["exit_code"], 0)
        data = result["data"]
        self.assertTrue(data["summary_consistent"])
        self.assertEqual(data["recomputed_counts"]["n"], 2)
        self.assertEqual(
            data["legacy_recommendation"], "legacy_direct_replacement_likely_sufficient"
        )
        self.assertFalse(data["image_and_latent_metrics_independently_verified"])
        self.assertFalse(data["release_qualified"])
        self.assertFalse(data["process_completion_verified"])

    def test_unknown_direction_is_not_reported_as_correct(self):
        self.raw["per_scene"]["scene"][0]["cos_AP_AF"] = float("nan")
        self.raw["cos"] = 0.9
        self.raw["start"]["cos"] = float("nan")
        result = self.call()
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["data"]["direction_assessment"], "incomplete")
        self.assertEqual(result["data"]["unknown_metric_counts"]["cos_AP_AF"], 1)
        self.assertNotIn("NaN", json.dumps(result))

    def test_zero_baseline_ratios_remain_unavailable(self):
        row = self.raw["per_scene"]["scene"][0]
        row.update(
            d_AF=0,
            d_AP=0,
            d_PF=0,
            progress_ratio=None,
            correction_ratio=None,
            cos_AP_AF=None,
        )
        self.raw.update(cos=0.9, start=dict(progress=None, correction=None, cos=None))
        result = self.call()
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(
            result["data"]["unknown_metric_counts"],
            dict(progress_ratio=1, correction_ratio=1, cos_AP_AF=1),
        )

    def test_wrong_direction_is_preserved_even_when_summary_is_consistent(self):
        self.raw["per_scene"]["scene"][0]["cos_AP_AF"] = -0.8
        self.raw.update(wrong_direction=1, cos=0.05)
        self.raw["start"]["cos"] = -0.8
        result = self.call()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(
            result["data"]["direction_assessment"], "wrong_direction_recorded"
        )
        self.assertIn("not a binary", result["data"]["inspection_pass_semantics"])

    def test_legacy_else_branch_does_not_invent_wrong_direction(self):
        for row in self.raw["per_scene"]["scene"]:
            row.update(d_AP=0.22, progress_ratio=1.1)
        self.raw["progress_ratio"] = 1.1
        self.raw["start"]["progress"] = self.raw["steady"]["progress"] = 1.1
        data = self.call()["data"]
        self.assertTrue(data["legacy_else_text_overstates_wrong_direction"])
        self.assertEqual(data["recomputed_counts"]["wrong_direction"], 0)

    def test_fabricated_counter_is_rejected(self):
        self.raw["wrong_direction"] = 1
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertIn("wrong_direction", result["data"]["inconsistent_fields"])

    def test_row_ratio_is_recomputed_from_distances(self):
        self.raw["per_scene"]["scene"][0]["progress_ratio"] = 0.9
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertIn("scene[0].progress_ratio", result["data"]["inconsistent_fields"])

    def test_empty_group_fallback_is_disclosed(self):
        self.raw["per_scene"]["scene"].pop(0)
        self.raw.update(n=1, cos=0.9, start=dict(progress=0.8, correction=0.5, cos=0.9))
        data = self.call()["data"]
        self.assertEqual(data["actual_group_counts"]["start"], 0)
        self.assertEqual(data["legacy_empty_group_fallback"], ["start"])

    def test_missing_raw_scene_or_field_blocks(self):
        self.raw = {"n": 21, "wrong_direction": 0}
        self.assertEqual(self.call()["exit_code"], 3)
        self.raw = sample()
        del self.raw["per_scene"]["scene"][0]["cos_AP_AF"]
        self.assertEqual(self.call()["exit_code"], 3)

    def test_broken_chunk_chain_blocks(self):
        self.raw["per_scene"]["scene"][1]["prev"] = 0
        self.assertEqual(self.call()["exit_code"], 3)

    def test_invalid_metric_or_boolean_counter_blocks(self):
        self.raw["n"] = True
        self.assertEqual(self.call()["exit_code"], 3)
        self.raw = sample()
        self.raw["per_scene"]["scene"][0]["cos_AP_AF"] = 2
        self.assertEqual(self.call()["exit_code"], 3)

    def test_changed_registered_source_blocks(self):
        (self.root / "preview_2a_handoff.py").write_text("changed")
        self.assertEqual(self.call()["exit_code"], 3)

    def test_symlink_raw_result_blocks(self):
        self.call()
        link = self.root.parent / "linked.json"
        link.symlink_to(self.result)
        self.assertEqual(
            self.invoke(
                "experiment",
                "inspect",
                "--experiment",
                "preview-handoff",
                "--result",
                str(link),
            )["exit_code"],
            3,
        )

    def test_dry_run_does_not_create_report_or_modify_raw(self):
        report = self.root.parent / "report.json"
        self.assertEqual(
            self.call("--dry-run", "--output", str(report))["exit_code"], 0
        )
        self.assertFalse(report.exists())
        self.assertEqual(json.loads(self.result.read_text()), self.raw)


if __name__ == "__main__":
    unittest.main()
