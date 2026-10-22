"""CLI admission and real small children, using a frozen fake test suite."""

import contextlib
import io
import subprocess
import unittest
from unittest.mock import patch

import test_worldctl as foundation
import worldctl
import worldctl_tests as tests


class TestRunTests(unittest.TestCase):
    invoke = foundation.WorldctlTests.invoke

    def setUp(self):
        foundation.WorldctlTests.setUp(self)
        (self.root / "test_one.py").write_text("print('first')\n")
        (self.root / "test_two.py").write_text("print('second')\n")
        self.commit()
        self.head = worldctl.git(self.root, "rev-parse", "HEAD")
        for module, name, value in [
            (worldctl, "ROOT", self.root),
            (tests, "ROOT", self.root),
            (tests, "BASELINE", self.head),
            (tests, "SCRIPTS", ("test_one.py", "test_two.py")),
        ]:
            item = patch.object(module, name, value)
            item.start()
            self.addCleanup(item.stop)

    def commit(self):
        subprocess.run(
            ["git", "-C", str(self.root), "add", "test_one.py", "test_two.py"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(self.root), "commit", "-qm", "test suite"], check=True
        )

    def run_suite(self, *args):
        with contextlib.redirect_stderr(io.StringIO()):
            return self.invoke(
                "test", "run", "--suite", "offline", "--timeout", "3", *args
            )

    def test_actual_suite_runs_with_raw_results(self):
        result = self.run_suite()
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(
            [x["child_exit_code"] for x in result["data"]["steps"]], [0, 0]
        )
        self.assertFalse(result["data"]["runtime_started"])

    def test_bad_selector_has_no_child(self):
        self.assertEqual(self.invoke("test", "run")["exit_code"], 2)
        self.assertEqual(
            self.invoke("test", "run", "--suite", "gpu", "--timeout", "3")["exit_code"],
            2,
        )

    def test_other_checkout_is_not_executed(self):
        with patch.object(tests, "ROOT", self.root.parent):
            self.assertEqual(self.run_suite()["exit_code"], 3)

    def test_dirty_or_env_file_blocks(self):
        (self.root / "untracked.py").write_text("print('never')")
        self.assertEqual(self.run_suite()["exit_code"], 3)
        (self.root / "untracked.py").unlink()
        (self.root / ".env").write_text("FAKE_PRIVATE_VALUE=do-not-read")
        result = self.run_suite()
        self.assertEqual(result["exit_code"], 3)
        self.assertNotIn("do-not-read", str(result))

    def test_hash_check_catches_assume_unchanged_source(self):
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "update-index",
                "--assume-unchanged",
                "test_one.py",
            ],
            check=True,
        )
        (self.root / "test_one.py").write_text("raise Exception('unreviewed')")
        self.assertEqual(
            self.run_suite()["errors"][0]["code"], "offline_source_changed"
        )

    def test_dry_run_has_no_logs_or_children(self):
        result = self.run_suite("--dry-run")
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["data"]["execution_started"])
        self.assertEqual(result["artifacts"], [])

    def test_existing_report_blocks_before_execution(self):
        output = self.root / "existing.json"
        output.write_text("preserve")
        result = self.run_suite("--output", str(output))
        self.assertEqual(result["errors"][0]["code"], "output_exists")
        self.assertEqual(output.read_text(), "preserve")

    def test_new_report_keeps_child_log_links(self):
        # Output lives outside the execution checkout so admission remains clean.
        output = self.root.parent / "receipt.json"
        result = self.run_suite("--output", str(output))
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(len(result["artifacts"]), 5)
        self.assertEqual(result["artifacts"][-1], str(output))


if __name__ == "__main__":
    unittest.main()
