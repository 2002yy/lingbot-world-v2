"""Mock device telemetry; tests never query the real GPU or import torch."""

import argparse
import contextlib
import io
import json
import subprocess
import unittest
from unittest.mock import patch

import worldctl
import worldctl_gpu as gpu


DEVICE = [["GPU-fixture", "RTX fixture", "8192", "400", "7792"]]


class GpuCheckTests(unittest.TestCase):
    def call(self, rows=None, apps=None, **kwargs):
        options = argparse.Namespace(
            gpu_command="check", timeout=5, dry_run=False, **kwargs
        )
        with patch.object(
            gpu,
            "query",
            side_effect=[
                DEVICE if rows is None else rows,
                [] if apps is None else apps,
            ],
        ) as query:
            result = gpu.check(worldctl.ROOT, options)
            if query.call_count == 2:
                self.assertEqual(
                    query.call_args_list[0].args[1], query.call_args_list[1].args[1]
                )
        return result

    def test_clean_snapshot_has_no_execution_authority(self):
        data = self.call()
        self.assertEqual(data["exit_code"], 0)
        self.assertTrue(data["clean"])
        self.assertFalse(data["valid_for_future_execution"])
        self.assertFalse(data["runtime_started"])
        self.assertFalse(data["release_qualified"])
        self.assertEqual(
            data["thresholds"], dict(max_unattributed_mib=700, min_free_mib=6000)
        )

    def test_frozen_threshold_edges(self):
        for used, free, expected in [(700, 6000, 0), (701, 6000, 3), (0, 5999, 3)]:
            data = self.call(
                [["GPU-fixture", "RTX fixture", "8192", str(used), str(free)]]
            )
            self.assertEqual(data["exit_code"], expected)

    def test_attributed_usage_is_separate_from_ghost_occupancy(self):
        data = self.call(
            [["GPU-fixture", "RTX fixture", "8192", "1400", "6792"]],
            [["GPU-fixture", "800"]],
        )
        self.assertEqual(data["exit_code"], 0)
        self.assertEqual(data["sample"]["unattributed_mib"], 600)

    def test_unknown_app_memory_is_not_treated_as_zero(self):
        self.assertEqual(self.call(apps=[["GPU-fixture", "N/A"]])["exit_code"], 3)

    def test_wrong_gpu_or_snapshot_race_is_blocked(self):
        for apps in (
            [["GPU-other", "100"]],
            [["GPU-fixture", "401"]],
            [["GPU-fixture", "-1"]],
        ):
            self.assertEqual(self.call(apps=apps)["exit_code"], 3)

    def test_multiple_or_missing_gpu_is_blocked(self):
        for rows in ([], DEVICE * 2):
            self.assertEqual(self.call(rows=rows)["exit_code"], 3)

    def test_invalid_memory_is_blocked(self):
        for values in (
            ["N/A", "1", "10"],
            ["8192", "-1", "8000"],
            ["8192", "8000", "8000"],
        ):
            self.assertEqual(
                self.call(rows=[["GPU-fixture", "RTX fixture", *values]])["exit_code"],
                3,
            )

    def test_query_failure_and_timeout_are_sanitized(self):
        for error, expected in [
            (FileNotFoundError("private"), 3),
            (subprocess.CalledProcessError(1, "private", stderr="secret"), 3),
            (subprocess.TimeoutExpired("private", 5), 5),
        ]:
            with patch.object(gpu, "query", side_effect=error):
                data = gpu.check(
                    worldctl.ROOT, argparse.Namespace(gpu_command="check", timeout=5)
                )
            self.assertEqual(data["exit_code"], expected)
            self.assertNotIn("private", json.dumps(data))
            self.assertNotIn("secret", json.dumps(data))

    def test_dry_run_has_no_device_calls(self):
        with patch.object(gpu, "query") as query:
            data = gpu.check(
                worldctl.ROOT,
                argparse.Namespace(gpu_command="check", timeout=5, dry_run=True),
            )
        query.assert_not_called()
        self.assertFalse(data["gpu_queried"])
        self.assertIsNone(data["clean"])

    def test_failed_process_query_never_yields_clean(self):
        with patch.object(
            gpu,
            "query",
            side_effect=[DEVICE, subprocess.CalledProcessError(1, "nvidia-smi")],
        ):
            data = gpu.check(
                worldctl.ROOT, argparse.Namespace(gpu_command="check", timeout=5)
            )
        self.assertEqual(data["exit_code"], 3)
        self.assertIsNone(data["clean"])

    def test_query_over_budget_is_rejected_after_return(self):
        child = subprocess.CompletedProcess([], 0, stdout="GPU-fixture, 42\n")
        with (
            patch.object(gpu.time, "monotonic", side_effect=[10, 13]),
            patch.object(gpu.subprocess, "run", return_value=child),
        ):
            with self.assertRaises(subprocess.TimeoutExpired):
                gpu.query("--query-compute-apps=gpu_uuid,used_memory", 12)

    def test_changed_gate_is_rejected_before_import_or_query(self):
        with (
            patch.object(gpu, "GATE_SHA256", "wrong"),
            patch.object(gpu, "query") as query,
        ):
            with self.assertRaises(worldctl.CLIError):
                gpu.check(
                    worldctl.ROOT, argparse.Namespace(gpu_command="check", timeout=5)
                )
        query.assert_not_called()

    def test_invalid_budget_has_no_device_calls(self):
        for timeout in (None, 0, 11):
            with patch.object(gpu, "query") as query:
                with self.assertRaises(worldctl.CLIError):
                    gpu.check(
                        worldctl.ROOT,
                        argparse.Namespace(gpu_command="check", timeout=timeout),
                    )
            query.assert_not_called()

    def test_cli_envelope_is_single_json(self):
        output = io.StringIO()
        with (
            patch.object(gpu, "query", side_effect=[DEVICE, []]),
            contextlib.redirect_stdout(output),
        ):
            code = worldctl.main(["gpu", "check", "--timeout", "5", "--json"])
        self.assertEqual(code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["command"], "gpu check")

    def test_query_uses_remaining_budget_and_allowlisted_environment(self):
        child = subprocess.CompletedProcess([], 0, stdout="GPU-fixture, 42\n")
        with (
            patch.object(gpu.time, "monotonic", return_value=10),
            patch.object(gpu.subprocess, "run", return_value=child) as run,
            patch.dict(gpu.os.environ, {"PRIVATE_KEY": "secret", "PATH": "/usr/bin"}),
        ):
            self.assertEqual(
                gpu.query("--query-compute-apps=gpu_uuid,used_memory", 12),
                [["GPU-fixture", "42"]],
            )
        self.assertEqual(run.call_args.kwargs["timeout"], 2)
        self.assertNotIn("PRIVATE_KEY", run.call_args.kwargs["env"])


if __name__ == "__main__":
    unittest.main()
