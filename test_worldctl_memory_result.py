"""Historical memory rows, completion and numerical criteria stay distinct."""

import hashlib
import json
import unittest

import test_worldctl_experiments as catalogue


def sample():
    memory = dict(
        alloc=100.0,
        reserved=200.0,
        max_alloc=150.0,
        max_reserved=300.0,
        driver_free=7900.0,
    )
    points = [
        dict(tag=tag, driver_total=8192.0, **memory)
        for tag in (
            "1_after_model_load",
            "2_after_tae_load",
            "3_after_condition_ready",
            "4_kv_window_full",
            "5_after_long_loop",
        )
    ]
    rows = [
        dict(chunk=i, dit=0.1, kv=0.2, tae=0.3, ttfnf=2.0 if i < 2 else 1.0, **memory)
        for i in range(65)
    ]
    return dict(
        weight="bf16",
        preset="temporal_stability_8_2",
        chunks=65,
        chunk_size=3,
        points=points,
        rows=rows,
        peak_max_reserved_mib=300.0,
        margin_mib=7892.0,
        min_driver_free_mib=7900.0,
        growth_alloc_pct=0.0,
        growth_reserved_pct=0.0,
        ttfnf_p50_ms=1015.625,
        ttfnf_worst_ms=2000.0,
        dit_ms=100.0,
        kv_ms=200.0,
        tae_ms=300.0,
        overall="PASS",
    )


class MemoryResultTests(unittest.TestCase):
    invoke = catalogue.ExperimentTests.invoke

    def setUp(self):
        catalogue.ExperimentTests.setUp(self)
        self.result = self.root.parent / "raw-memory.json"
        self.raw = sample()

    def call(self, raw=None, *extra):
        self.result.write_text(json.dumps(self.raw if raw is None else raw))
        return self.invoke(
            "experiment",
            "inspect",
            "--experiment",
            "memory-gate",
            "--result",
            str(self.result),
            *extra,
        )

    def test_recomputes_numeric_evidence_and_explains_mislabeled_p50(self):
        result = self.call()
        self.assertEqual(result["exit_code"], 0)
        data = result["data"]
        self.assertEqual(data["historical_numeric_gate"], "PASS")
        self.assertEqual(data["ttfnf_median_ms"], 1000.0)
        self.assertEqual(data["recomputed_metrics"]["ttfnf_p50_ms"], 1015.625)
        self.assertIn("arithmetic mean", data["ttfnf_label_semantics"])
        self.assertEqual(
            data["raw_result_sha256"],
            hashlib.sha256(self.result.read_bytes()).hexdigest(),
        )
        for key in (
            "process_completion_verified",
            "provenance_verified",
            "release_qualified",
            "runtime_started",
            "execution_authorized",
            "no_oom_independently_verified",
        ):
            self.assertFalse(data[key])

    def test_margin_boundary_is_the_frozen_500_mib(self):
        for margin, expected in ((500, 0), (499, 4)):
            raw = sample()
            for row in raw["rows"]:
                row["max_reserved"] = 8192 - margin
            raw.update(
                peak_max_reserved_mib=8192 - margin,
                margin_mib=margin,
                overall="PASS" if margin == 500 else "FAIL",
            )
            self.assertEqual(self.call(raw)["exit_code"], expected)

    def test_exact_two_percent_growth_is_failure(self):
        for row in self.raw["rows"][-16:]:
            row["alloc"] = 102.0
        self.raw.update(growth_alloc_pct=2.0, overall="FAIL")
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertTrue(result["data"]["summary_consistent"])

    def test_success_header_cannot_mask_failed_rows(self):
        for row in self.raw["rows"][-16:]:
            row["alloc"] = 102.0
        self.raw["growth_alloc_pct"] = 2.0
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["data"]["historical_numeric_gate"], "FAIL")
        self.assertIn("overall", result["data"]["inconsistent_fields"])

    def test_incorrect_summary_is_reported(self):
        self.raw["margin_mib"] += 1
        result = self.call()
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["data"]["inconsistent_fields"], ["margin_mib"])

    def test_missing_duplicate_or_out_of_order_chunks_block(self):
        for mode in ("missing", "duplicate", "reordered"):
            raw = sample()
            if mode == "missing":
                raw["rows"].pop()
            elif mode == "duplicate":
                raw["rows"][2]["chunk"] = 1
            else:
                raw["rows"][0], raw["rows"][1] = raw["rows"][1], raw["rows"][0]
            self.assertEqual(self.call(raw)["exit_code"], 3)

    def test_invalid_numeric_types_and_nonfinite_metrics_block(self):
        for value in (True, None, -1, 10**400, float("nan")):
            raw = sample()
            raw["rows"][0]["alloc"] = value
            self.assertEqual(self.call(raw)["exit_code"], 3)

    def test_summary_without_raw_rows_cannot_pass(self):
        self.assertEqual(self.call({"overall": "PASS"})["exit_code"], 3)

    def test_other_preset_and_float_configuration_are_unsupported(self):
        for key, value in (("preset", "different"), ("chunks", 65.0)):
            raw = sample()
            raw[key] = value
            self.assertEqual(self.call(raw)["exit_code"], 3)

    def test_missing_summary_field_blocks(self):
        del self.raw["margin_mib"]
        self.assertEqual(self.call()["exit_code"], 3)

    def test_bad_json_and_duplicate_keys_do_not_echo_content(self):
        for raw in ('{"private":"secret","private":1}', "secret-invalid-json"):
            self.result.write_text(raw)
            result = self.invoke(
                "experiment",
                "inspect",
                "--experiment",
                "memory-gate",
                "--result",
                str(self.result),
            )
            self.assertEqual(result["exit_code"], 3)
            self.assertNotIn("secret", str(result))

    def test_changed_registered_writer_blocks_inspection(self):
        (self.root / "m0_memgate.py").write_text("changed")
        self.assertEqual(self.call()["exit_code"], 3)

    def test_output_is_non_overwriting_and_dry_run_preserves_raw(self):
        report = self.root.parent / "report.json"
        self.assertEqual(
            self.call(None, "--output", str(report), "--dry-run")["exit_code"], 0
        )
        self.assertFalse(report.exists())
        self.assertEqual(self.call(None, "--output", str(report))["exit_code"], 0)
        before = report.read_bytes()
        self.assertEqual(self.call(None, "--output", str(report))["exit_code"], 3)
        self.assertEqual(report.read_bytes(), before)

    def test_missing_selection_is_a_parameter_error(self):
        self.assertEqual(self.invoke("experiment", "inspect")["exit_code"], 2)


if __name__ == "__main__":
    unittest.main()
