"""Journals are checked, never replayed or promoted to execution provenance."""

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import worldctl
from worldctl_execution import execute
from worldctl_fingerprint import digest
from worldctl_run_inspect import inspect_run


class RunInspectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def create(self, source="print('fixture')", timeout=2, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            result = execute(
                self.root,
                [[sys.executable, "-c", source]],
                timeout,
                record=True,
                **kwargs,
            )
        return Path(result["execution_record"])

    def change(self, path, mutate, rehash=True):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(events)
        if rehash:
            previous = None
            for index, event in enumerate(events):
                event["seq"] = index
                event["previous_sha256"] = previous
                event["sha256"] = digest(
                    {k: v for k, v in event.items() if k != "sha256"}
                )
                previous = event["sha256"]
        path.write_text("".join(json.dumps(event) + "\n" for event in events))

    def blocked(self, path, code=3):
        with self.assertRaises(worldctl.CLIError) as error:
            inspect_run(path)
        self.assertEqual(error.exception.code, code)

    def test_success_rechecks_logs_without_authenticating_execution(self):
        result = inspect_run(self.create())
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["record_complete"])
        self.assertTrue(result["log_hashes_verified"])
        self.assertFalse(result["execution_origin_authenticated"])
        self.assertFalse(result["run_binding_verified"])

    def test_failed_child_stays_failed_after_consistent_inspection(self):
        result = inspect_run(self.create("raise SystemExit(7)"))
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["recorded_exit_code"], 4)
        self.assertEqual(result["child_outcomes"][0]["child_exit_code"], 7)

    def test_timeout_stays_timeout(self):
        result = inspect_run(self.create("import time; time.sleep(10)", timeout=0.15))
        self.assertEqual(result["exit_code"], 5)

    def test_partial_record_and_reordered_events_block(self):
        path = self.create()
        original = path.read_bytes()
        self.change(path, lambda rows: rows.pop())
        self.blocked(path)
        path.write_bytes(original)

        def reorder(rows):
            rows[1], rows[2] = rows[2], rows[1]

        self.change(path, reorder)
        self.blocked(path)

    def test_stale_chain_and_success_header_cannot_mask_failure(self):
        path = self.create("raise SystemExit(7)")
        original = path.read_bytes()

        def pretend(rows):
            rows[-1]["data"].update(status="PASS", exit_code=0, stop_code="")

        self.change(path, pretend, False)
        self.blocked(path)
        path.write_bytes(original)
        self.change(path, pretend, True)
        self.blocked(path)

    def test_changed_or_missing_logs_do_not_pass(self):
        path = self.create()
        log = path.parent / "0.stdout"
        original = log.read_bytes()
        log.write_bytes(original + b"changed")
        self.blocked(path, 4)
        log.unlink()
        self.blocked(path, 3)

    def test_artifact_path_traversal_is_never_followed(self):
        path = self.create()

        def escape(rows):
            rows[-2]["data"]["artifacts"][0]["name"] = "../../outside"

        self.change(path, escape)
        self.blocked(path)

    def test_record_and_log_symlinks_are_rejected(self):
        path = self.create()
        linked = self.root / "linked.jsonl"
        linked.symlink_to(path)
        self.blocked(linked)
        log = path.parent / "0.stdout"
        log.unlink()
        log.symlink_to(path)
        self.blocked(path)

    def test_log_overflow_preserves_budget_exit_without_claiming_hashes(self):
        result = inspect_run(self.create("print('x'*10000)", log_limit=100))
        self.assertEqual(result["exit_code"], 5)
        self.assertFalse(result["log_hashes_verified"])

    def test_malformed_nested_types_block_without_traceback(self):
        path = self.create()
        original = path.read_bytes()
        for target in ("status", "reason", "artifact"):
            path.write_bytes(original)

            def malformed(rows):
                if target == "status":
                    rows[-1]["data"]["status"] = {}
                elif target == "reason":
                    rows[-2]["data"]["record"]["stop_reason"] = {}
                else:
                    rows[-2]["data"]["artifacts"][0]["status"] = {}

            self.change(path, malformed)
            self.blocked(path)

    def test_combined_log_budget_cannot_be_hidden(self):
        path = self.create(
            "import sys; print('x'*60); print('y'*60,file=sys.stderr)", log_limit=100
        )

        def pretend(rows):
            rows[-2]["data"]["record"].update(stop_reason="", child_exit_code=0)
            rows[-1]["data"].update(status="PASS", exit_code=0, stop_code="")

        self.change(path, pretend)
        self.blocked(path)

    def test_cli_is_single_json_and_dry_run_writes_no_report(self):
        path = self.create()
        report = self.root / "report.json"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = worldctl.main(
                [
                    "test",
                    "inspect-run",
                    "--record",
                    str(path),
                    "--json",
                    "--dry-run",
                    "--output",
                    str(report),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertFalse(report.exists())

    def test_missing_record_selection_is_parameter_error(self):
        self.blocked(None, 2)

    def test_launch_failure_remains_unstarted(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = execute(self.root, [[str(self.root / "missing")]], 2, record=True)
        record = inspect_run(Path(result["execution_record"]))
        self.assertEqual(record["exit_code"], 3)


class RecordedOutcomeTests(unittest.TestCase):
    """Stop code, exit code and status must agree without any live child.

    Reading a record compares fstat against lstat identity, which only agrees on
    POSIX; elsewhere every record is rejected before pairing can be checked.
    """

    def record(self, end):
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        events = [
            (
                "run_started",
                dict(
                    commands=[["step"]],
                    log_limit=1024,
                    context={},
                    budget_seconds=5,
                    origin_authenticated=False,
                    release_qualified=False,
                ),
            ),
            ("run_finished", end),
        ]
        run_id = "a" * 32
        previous = None
        lines = []
        for index, (name, data) in enumerate(events):
            item = dict(
                schema_version=1,
                run_id=run_id,
                seq=index,
                event=name,
                recorded_at=f"2026-10-22T00:00:0{index}+00:00",
                elapsed_seconds=float(index),
                previous_sha256=previous,
                data=data,
            )
            item["sha256"] = digest(item)
            previous = item["sha256"]
            lines.append(json.dumps(item))
        path = folder / "execution.jsonl"
        path.write_text("\n".join(lines) + "\n")
        return path

    def blocked(self, path):
        with self.assertRaises(worldctl.CLIError):
            inspect_run(path)

    @unittest.skipUnless(os.name == "posix", "record identity checks are POSIX-only")
    def test_consistent_terminal_pairing_is_accepted(self):
        for stop_code, exit_code in [
            ("execution_not_started", 5),
            ("execution_precondition_changed", 3),
            ("execution_record_unavailable", 3),
        ]:
            path = self.record(
                dict(
                    status="BLOCKED",
                    exit_code=exit_code,
                    step_count=0,
                    execution_started=False,
                    stop_code=stop_code,
                )
            )
            self.assertEqual(inspect_run(path)["exit_code"], exit_code)

    @unittest.skipUnless(os.name == "posix", "record identity checks are POSIX-only")
    def test_lookalike_terminal_pairing_is_rejected(self):
        for stop_code, exit_code in [
            ("execution_precondition_changed", 5),
            ("execution_not_started", 3),
            ("execution_record_unavailable", 5),
        ]:
            path = self.record(
                dict(
                    status="BLOCKED",
                    exit_code=exit_code,
                    step_count=0,
                    execution_started=False,
                    stop_code=stop_code,
                )
            )
            self.blocked(path)


if __name__ == "__main__":
    unittest.main()
