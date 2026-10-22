"""Real CPU children exercise recording, failures, bounds and cleanup."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import worldctl_execution as execution
from worldctl_execution_record import Journal, snapshot
from worldctl_fingerprint import digest


class ExecutionRecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_child(self, source, timeout=2, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return execution.execute(
                self.root,
                [[sys.executable, "-c", source]],
                timeout,
                record=True,
                **kwargs,
            )

    def events(self, result):
        return [
            json.loads(line)
            for line in Path(result["execution_record"]).read_text().splitlines()
        ]

    def test_success_records_order_pid_exit_and_log_hashes(self):
        result = self.run_child("print('recorded')")
        self.assertEqual(result["exit_code"], 0)
        events = self.events(result)
        self.assertEqual(
            [r["event"] for r in events],
            [
                "run_started",
                "step_started",
                "child_started",
                "step_finished",
                "run_finished",
            ],
        )
        previous = None
        for index, event in enumerate(events):
            self.assertEqual(event["seq"], index)
            self.assertEqual(event["previous_sha256"], previous)
            self.assertEqual(
                event["sha256"],
                digest({k: v for k, v in event.items() if k != "sha256"}),
            )
            previous = event["sha256"]
        step = events[3]["data"]
        self.assertEqual(step["record"]["pid"], events[2]["data"]["pid"])
        self.assertEqual(step["record"]["child_exit_code"], 0)
        self.assertEqual(
            step["artifacts"][0]["sha256"], hashlib.sha256(b"recorded\n").hexdigest()
        )
        self.assertTrue(result["execution_record_complete"])
        self.assertFalse(result["recording_origin_authenticated"])

    def test_child_failure_stops_later_launches(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = execution.execute(
                self.root,
                [
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    [sys.executable, "-c", "raise Exception('never')"],
                ],
                2,
                record=True,
            )
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(
            sum(e["event"] == "child_started" for e in self.events(result)), 1
        )
        self.assertEqual(result["steps"][0]["child_exit_code"], 7)

    def test_timeout_records_real_cleanup_and_exit(self):
        result = self.run_child("import time; time.sleep(10)", timeout=0.2)
        self.assertEqual(result["exit_code"], 5)
        self.assertTrue(result["steps"][0]["cleanup_complete"])
        self.assertEqual(self.events(result)[-1]["data"]["exit_code"], 5)

    def test_cancel_is_not_reported_as_success(self):
        cancel = threading.Event()
        timer = threading.Timer(0.15, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        result = self.run_child("import time; time.sleep(10)", cancel=cancel)
        self.assertEqual(result["exit_code"], 5)
        self.assertEqual(result["steps"][0]["stop_reason"], "cancelled")

    def test_launch_failure_is_not_execution_started(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = execution.execute(
                self.root, [[str(self.root / "missing-python")]], 2, record=True
            )
        self.assertEqual(result["exit_code"], 3)
        self.assertFalse(result["execution_started"])
        self.assertFalse(
            any(e["event"] == "child_started" for e in self.events(result))
        )

    def test_dry_run_creates_no_journal(self):
        with patch.object(
            execution.tempfile, "mkdtemp", side_effect=AssertionError("must not create")
        ):
            result = self.run_child("raise Exception('never')", dry_run=True)
        self.assertFalse(result["execution_started"])
        self.assertNotIn("execution_record", result)

    def test_failed_admission_records_no_child(self):
        result = self.run_child(
            "raise Exception('never')", before_step=lambda argv: False
        )
        self.assertEqual(
            [r["event"] for r in self.events(result)], ["run_started", "run_finished"]
        )
        self.assertEqual(result["exit_code"], 3)

    def test_start_record_failure_prevents_execution(self):
        with patch(
            "worldctl_execution_record.os.fsync",
            side_effect=OSError("disk unavailable"),
        ):
            result = self.run_child("open('must-not-exist','w').write('x')")
        self.assertEqual(result["exit_code"], 3)
        self.assertFalse(result["execution_started"])
        self.assertFalse((self.root / "must-not-exist").exists())

    def test_child_record_failure_still_cleans_process(self):
        original = Journal.emit

        def emit(journal, event, **data):
            if event == "child_started":
                journal.healthy = False
                return False
            return original(journal, event, **data)

        with patch.object(Journal, "emit", emit):
            result = self.run_child("import time; time.sleep(10)")
        self.assertEqual(result["exit_code"], 3)
        self.assertTrue(result["steps"][0]["cleanup_complete"])
        self.assertFalse(result["execution_record_complete"])

    def test_terminal_record_failure_does_not_claim_complete(self):
        original = Journal.emit

        def emit(journal, event, **data):
            if event == "run_finished":
                journal.healthy = False
                return False
            return original(journal, event, **data)

        with patch.object(Journal, "emit", emit):
            result = self.run_child("pass")
        self.assertEqual(result["observed_execution_exit_code"], 0)
        self.assertEqual(result["exit_code"], 3)
        self.assertFalse(result["execution_record_complete"])

    def test_log_overflow_is_recorded_without_hashing_unbounded_bytes(self):
        result = self.run_child("print('x'*10000)", log_limit=100)
        self.assertEqual(result["exit_code"], 5)
        artifact = self.events(result)[-2]["data"]["artifacts"][0]
        self.assertEqual(artifact["status"], "oversize")
        self.assertIsNone(artifact["sha256"])

    def test_context_bounds_reject_unknown_fields_before_any_child(self):
        with self.assertRaises(ValueError):
            self.run_child("pass", record_context={"PRIVATE_KEY": "must-not-record"})

    def test_artifact_snapshot_rejects_symlinks(self):
        path = self.root / "out"
        path.write_bytes(b"fixture")
        link = self.root / "link"
        link.symlink_to(path)
        self.assertEqual(snapshot(link, 100)["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
