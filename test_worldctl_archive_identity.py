"""Archive identity expectations use embedded execution context, never checker HEAD."""

import argparse
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
from worldctl_run_archive import archive_run, inspect_archive


class ArchiveIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def create(self, source="pass", context=None, timeout=2):
        with contextlib.redirect_stderr(io.StringIO()):
            run = execute(
                self.root,
                [[sys.executable, "-c", source]],
                timeout,
                record=True,
                record_root=self.root / "records",
                record_context=context or {},
            )
        output = self.root / "archive.json"
        data = archive_run(
            argparse.Namespace(
                record=Path(run["execution_record"]), output=output, dry_run=False
            )
        )
        output.write_text(
            json.dumps(
                dict(
                    schema_version="agent-cli-result-v1",
                    command="test archive-run",
                    status=data["status"],
                    exit_code=data["exit_code"],
                    data=data,
                )
            )
        )
        return output

    def test_match_uses_saved_execution_head_and_keeps_authority_false(self):
        path = self.create(context=dict(execution_head_sha="a" * 40, suite="offline"))
        result = inspect_archive(path, expected_head="a" * 40, expected_suite="offline")
        self.assertTrue(result["recorded_identity_matches"])
        self.assertTrue(result["archive_verified"])
        for key in (
            "identity_origin_authenticated",
            "execution_origin_authenticated",
            "run_binding_verified",
            "provenance_verified",
            "release_qualified",
        ):
            self.assertFalse(result[key])

    def test_wrong_head_fails_but_retains_recorded_exit(self):
        result = inspect_archive(
            self.create(context=dict(execution_head_sha="a" * 40)),
            expected_head="b" * 40,
        )
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["recorded_exit_code"], 0)
        self.assertFalse(result["recorded_identity_matches"])

    def test_wrong_suite_fails(self):
        result = inspect_archive(
            self.create(context=dict(suite="offline")), expected_suite="other"
        )
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["identity_mismatched_fields"], ["suite"])

    def test_missing_identity_is_blocked(self):
        result = inspect_archive(
            self.create(), expected_head="a" * 40, expected_suite="offline"
        )
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(
            result["identity_missing_fields"], ["execution_head_sha", "suite"]
        )

    def test_no_expectation_is_not_identity_verification(self):
        result = inspect_archive(self.create())
        self.assertIsNone(result["recorded_identity_matches"])
        self.assertFalse(result["identity_check_requested"])

    def test_matching_identity_preserves_failure(self):
        result = inspect_archive(
            self.create("raise SystemExit(7)", dict(suite="offline")),
            expected_suite="offline",
        )
        self.assertTrue(result["recorded_identity_matches"])
        self.assertEqual(result["exit_code"], 4)

    def test_matching_identity_preserves_timeout(self):
        result = inspect_archive(
            self.create("import time; time.sleep(5)", dict(suite="offline"), 0.1),
            expected_suite="offline",
        )
        self.assertTrue(result["recorded_identity_matches"])
        self.assertEqual(result["exit_code"], 5)

    def test_bad_expectations_reject_before_archive_read(self):
        for options in (dict(expected_head="short"), dict(expected_suite="../offline")):
            with (
                self.subTest(options=options),
                patch("worldctl_run_archive.read_blob") as read,
            ):
                with self.assertRaises(worldctl.CLIError) as error:
                    inspect_archive(None, **options)
                self.assertEqual(error.exception.code, 2)
                read.assert_not_called()

    def test_success_header_cannot_mask_embedded_failure_before_identity_match(self):
        path = self.create("raise SystemExit(7)", dict(suite="offline"))
        raw = json.loads(path.read_text())
        raw.update(status="PASS", exit_code=0)
        path.write_text(json.dumps(raw))
        with self.assertRaises(worldctl.CLIError) as error:
            inspect_archive(path, expected_suite="offline")
        self.assertEqual(error.exception.code, 3)

    def test_actual_cli_accepts_expected_identity_and_dry_run_writes_nothing(self):
        path = self.create(context=dict(execution_head_sha="a" * 40, suite="offline"))
        output = self.root / "absent.json"
        command = [
            sys.executable,
            str(worldctl.ROOT / "worldctl.py"),
            "test",
            "inspect-archive",
            "--archive",
            str(path),
            "--expected-head",
            "a" * 40,
            "--expected-suite",
            "offline",
            "--dry-run",
            "--output",
            str(output),
            "--json",
        ]
        child = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr)
        self.assertEqual(child.stderr, "")
        self.assertTrue(json.loads(child.stdout)["data"]["recorded_identity_matches"])
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
