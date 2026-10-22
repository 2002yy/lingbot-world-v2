"""Real child-process regressions, CPU-only and without model imports."""

import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import worldctl_execution as execution


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_child(self, code, timeout=2, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return execution.execute(
                self.root, [[sys.executable, "-c", code]], timeout, **kwargs
            )

    def test_success_preserves_logs_and_exit(self):
        result = self.run_child(
            "import sys; print('out'); print('err',file=sys.stderr)"
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(Path(result["artifacts"][0]).read_text().strip(), "out")
        self.assertEqual(Path(result["artifacts"][1]).read_text().strip(), "err")
        self.assertTrue(result["steps"][0]["cleanup_complete"])

    def test_failure_does_not_start_next_child(self):
        with contextlib.redirect_stderr(io.StringIO()):
            result = execution.execute(
                self.root,
                [
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    [sys.executable, "-c", "open('wrong','w').write('x')"],
                ],
                2,
            )
        self.assertEqual(result["exit_code"], 4)
        self.assertEqual(result["steps"][0]["child_exit_code"], 7)
        self.assertFalse((self.root / "wrong").exists())

    def test_timeout_stops_descendant(self):
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(p.pid,flush=True); time.sleep(20)"
        result = self.run_child(code, timeout=0.4)
        self.assertEqual(result["exit_code"], 5)
        self.assertEqual(result["steps"][0]["stop_reason"], "timeout")
        pid = int(Path(result["artifacts"][0]).read_text())
        path = Path(f"/proc/{pid}/stat")
        self.assertTrue(
            not path.exists() or path.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        )
        self.assertTrue(result["steps"][0]["cleanup_complete"])

    def test_parent_exit_does_not_leave_descendant(self):
        code = "import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)']); print(p.pid,flush=True)"
        result = self.run_child(code)
        self.assertEqual(result["exit_code"], 0)
        pid = int(Path(result["artifacts"][0]).read_text())
        path = Path(f"/proc/{pid}/stat")
        self.assertTrue(
            not path.exists() or path.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        )

    def test_cancellation_uses_same_cleanup(self):
        cancel = threading.Event()
        timer = threading.Timer(0.2, cancel.set)
        timer.start()
        self.addCleanup(timer.cancel)
        result = self.run_child("import time; time.sleep(20)", cancel=cancel)
        self.assertEqual(result["exit_code"], 5)
        self.assertEqual(result["steps"][0]["stop_reason"], "cancelled")
        self.assertTrue(result["steps"][0]["cleanup_complete"])

    def test_dry_run_does_not_create_logs_or_child(self):
        with patch.object(
            execution.tempfile,
            "mkdtemp",
            side_effect=AssertionError("directory created"),
        ):
            result = self.run_child("raise Exception('never')", dry_run=True)
        self.assertFalse(result["execution_started"])

    def test_changed_precondition_stops_before_any_child(self):
        result = self.run_child(
            "raise Exception('never')", before_step=lambda argv: False
        )
        self.assertEqual(result["exit_code"], 3)
        self.assertFalse(result["execution_started"])

    def test_keyboard_interrupt_cleans_the_owned_child(self):
        original_sleep = time.sleep
        interrupted = []

        def interrupt_once(seconds):
            if not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt
            original_sleep(seconds)

        with patch.object(execution.time, "sleep", interrupt_once):
            result = self.run_child("import time; time.sleep(20)")
        self.assertEqual(result["steps"][0]["stop_reason"], "cancelled")
        self.assertTrue(result["steps"][0]["cleanup_complete"])

    def test_env_does_not_inherit_secrets_or_test_options(self):
        with patch.dict(
            os.environ,
            {"DEEPSEEK_API_KEY": "fake-do-not-inherit", "PYTEST_ADDOPTS": "bad-option"},
        ):
            result = self.run_child(
                "import os; assert 'DEEPSEEK_API_KEY' not in os.environ; assert 'PYTEST_ADDOPTS' not in os.environ; assert os.environ['CUDA_VISIBLE_DEVICES']==''"
            )
        self.assertEqual(result["exit_code"], 0)

    def test_log_overflow_stops_instead_of_false_success(self):
        result = self.run_child("print('x'*10000)", log_limit=100)
        self.assertEqual(result["exit_code"], 5)
        self.assertEqual(result["steps"][0]["stop_reason"], "log_budget")


if __name__ == "__main__":
    unittest.main()
