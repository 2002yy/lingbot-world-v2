"""Identity selection is independent of the CLI checkout and child success."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import worldctl
from worldctl_execution import execute
from worldctl_run_identity import inspect_identity


class RunIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def create(self, source="pass", context=None, timeout=2):
        with contextlib.redirect_stderr(io.StringIO()):
            result = execute(
                self.root,
                [[sys.executable, "-c", source]],
                timeout,
                record=True,
                record_context=context or {},
            )
        return Path(result["execution_record"])

    def test_match_checks_recorded_execution_head_not_checker_checkout(self):
        record = self.create(context=dict(execution_head_sha="a" * 40, suite="offline"))
        result = inspect_identity(
            record, expected_head="a" * 40, expected_suite="offline"
        )
        self.assertTrue(result["recorded_identity_matches"])
        self.assertFalse(result["identity_origin_authenticated"])
        self.assertFalse(result["run_binding_verified"])

    def test_wrong_head_is_failure_and_preserves_actual_recorded_exit(self):
        result = inspect_identity(
            self.create(context=dict(execution_head_sha="b" * 40)),
            expected_head="a" * 40,
        )
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["recorded_exit_code"], 0)
        self.assertEqual(result["identity_mismatched_fields"], ["execution_head_sha"])

    def test_wrong_suite_fails(self):
        result = inspect_identity(
            self.create(context=dict(suite="other")), expected_suite="offline"
        )
        self.assertEqual(result["exit_code"], 4)

    def test_missing_context_is_blocked(self):
        result = inspect_identity(self.create(), expected_head="a" * 40)
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["identity_missing_fields"], ["execution_head_sha"])

    def test_no_expectations_does_not_claim_match(self):
        result = inspect_identity(self.create())
        self.assertIsNone(result["recorded_identity_matches"])
        self.assertFalse(result["identity_check_requested"])

    def test_matching_identity_does_not_upgrade_child_failure(self):
        result = inspect_identity(
            self.create("raise SystemExit(9)", dict(suite="offline")),
            expected_suite="offline",
        )
        self.assertTrue(result["recorded_identity_matches"])
        self.assertEqual(result["exit_code"], 4)

    def test_matching_identity_does_not_upgrade_timeout(self):
        result = inspect_identity(
            self.create("import time; time.sleep(10)", dict(suite="offline"), 0.1),
            expected_suite="offline",
        )
        self.assertEqual(result["exit_code"], 5)

    def test_invalid_selections_reject_before_read(self):
        for options in [
            dict(expected_head="short"),
            dict(expected_head="A" * 40),
            dict(expected_suite=""),
            dict(expected_suite="x" * 65),
            dict(expected_suite="../offline"),
        ]:
            with (
                self.subTest(options=options),
                patch("worldctl_run_identity.inspect_run") as inspect,
            ):
                with self.assertRaises(worldctl.CLIError) as error:
                    inspect_identity(None, **options)
                self.assertEqual(error.exception.code, 2)
                inspect.assert_not_called()

    def test_log_change_is_rejected_before_identity_match(self):
        record = self.create(context=dict(suite="offline"))
        (record.parent / "0.stdout").write_text("changed")
        with self.assertRaises(worldctl.CLIError) as error:
            inspect_identity(record, expected_suite="offline")
        self.assertEqual(error.exception.code, 4)

    def test_real_cli_returns_single_json_and_dry_run_creates_no_report(self):
        record = self.create(context=dict(suite="offline", execution_head_sha="a" * 40))
        output = self.root / "absent.json"
        child = subprocess.run(
            [
                sys.executable,
                str(worldctl.ROOT / "worldctl.py"),
                "test",
                "inspect-run",
                "--record",
                str(record),
                "--expected-head",
                "a" * 40,
                "--expected-suite",
                "offline",
                "--dry-run",
                "--output",
                str(output),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stderr, "")
        self.assertTrue(json.loads(child.stdout)["data"]["recorded_identity_matches"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
