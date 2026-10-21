"""Standalone, stdlib-only CLI contracts; never import the model runtime."""

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


class WorldctlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "中文 space"
        self.root.mkdir()
        for args in [
            ("init", "-q"),
            ("config", "user.name", "CLI Tests"),
            ("config", "user.email", "cli@example.invalid"),
        ]:
            subprocess.run(
                ["git", "-C", str(self.root), *args], check=True, capture_output=True
            )
        for name in (
            *worldctl.OWNERS,
            "run.sh",
            "run.ps1",
            "gpu_cleanliness.py",
            "pyproject.toml",
        ):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "commit", "-qm", "fixture"], check=True
        )

    def invoke(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = worldctl.main(["--repo", str(self.root), *args])
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        data = json.loads(output.getvalue())
        self.assertEqual(code, data["exit_code"])
        return data

    def test_status_counts_and_source_head(self):
        (self.root / "run.sh").write_text("changed")
        (self.root / "untracked file").write_text("preserve")
        result = self.invoke("status")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["data"]["tracked_changes"], 1)
        self.assertEqual(result["data"]["untracked_entries"], 1)
        self.assertEqual(result["head_sha"], worldctl.repository(worldctl.ROOT)[1])
        self.assertEqual(
            result["data"]["requested_repo_head_sha"], worldctl.repository(self.root)[1]
        )
        self.assertFalse(result["data"]["runtime_started"])

    def test_doctor_missing_heavy_packages_is_not_gpu_or_release_pass(self):
        with patch.object(
            worldctl.importlib.metadata,
            "version",
            side_effect=worldctl.importlib.metadata.PackageNotFoundError,
        ):
            result = self.invoke("doctor")
        self.assertEqual(result["status"], "PASS")
        self.assertIsNone(result["data"]["runtime_ready"])
        self.assertFalse(result["data"]["release_qualified"])
        self.assertEqual(result["data"]["gpu"], "not_queried")
        self.assertTrue(
            all(
                x["status"] == "missing"
                for x in result["data"]["runtime_packages"].values()
            )
        )

    def test_doctor_missing_required_file_blocks(self):
        (self.root / "gpu_cleanliness.py").unlink()
        self.assertEqual(self.invoke("doctor")["exit_code"], 3)

    def test_bad_arguments_are_sanitized(self):
        for args in [(), ("private-command",), ("status", "--private-token=secret")]:
            with self.subTest(args=args):
                data = self.invoke(*args)
                self.assertEqual(data["exit_code"], 2)
                self.assertNotIn("secret", json.dumps(data))

    def test_no_repository_is_a_precondition_failure(self):
        with patch.object(worldctl, "ROOT", self.root.parent):
            self.assertEqual(self.invoke("status")["exit_code"], 3)

    def test_report_round_trip_is_one_json_and_non_overwriting(self):
        output = self.root / "report.json"
        result = self.invoke("doctor", "--output", str(output))
        self.assertEqual(json.loads(output.read_text()), result)
        before = output.read_bytes()
        self.assertEqual(self.invoke("status", "--output", str(output))["exit_code"], 3)
        self.assertEqual(output.read_bytes(), before)

    def test_dry_run_writes_nothing(self):
        output = self.root / "new.json"
        result = self.invoke("doctor", "--dry-run", "--output", str(output))
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(output.exists())

    def test_symlink_output_or_parent_is_rejected(self):
        target = self.root / "target"
        target.mkdir()
        link = self.root / "link"
        link.symlink_to(target, target_is_directory=True)
        self.assertEqual(
            self.invoke("doctor", "--output", str(link / "new.json"))["exit_code"], 3
        )
        self.assertFalse((target / "new.json").exists())

    def test_atomic_publication_does_not_overwrite_racing_writer(self):
        output = self.root / "racing.json"
        original_link = worldctl.os.link

        def racing_link(source, destination):
            output.write_text("another-writer")
            return original_link(source, destination)

        with patch.object(worldctl.os, "link", racing_link):
            self.assertEqual(
                self.invoke("doctor", "--output", str(output))["exit_code"], 3
            )
        self.assertEqual(output.read_text(), "another-writer")

    def test_cancel_is_structured(self):
        with patch.object(worldctl, "repository", side_effect=KeyboardInterrupt):
            self.assertEqual(self.invoke("status")["exit_code"], 5)

    def test_only_git_subprocesses_and_no_model_import(self):
        original_run = worldctl.subprocess.run

        def guarded(argv, **kwargs):
            self.assertEqual(argv[0], "git")
            return original_run(argv, **kwargs)

        with patch.object(worldctl.subprocess, "run", guarded):
            self.assertEqual(self.invoke("doctor")["exit_code"], 0)
            self.assertEqual(self.invoke("status")["exit_code"], 0)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("wan", sys.modules)

    def test_actual_entrypoint_has_single_json_and_empty_stderr(self):
        child = subprocess.run(
            [
                sys.executable,
                str(worldctl.ROOT / "worldctl.py"),
                "doctor",
                "--repo",
                str(self.root),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(child.returncode, 0)
        self.assertEqual(len(child.stdout.splitlines()), 1)
        self.assertFalse(child.stderr)
        self.assertFalse(json.loads(child.stdout)["data"]["runtime_started"])


if __name__ == "__main__":
    unittest.main()
