"""Historical experiment plans never import tensors or launch executors."""

import json
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_worldctl as foundation
import worldctl
import worldctl_experiments as experiments


class ExperimentTests(unittest.TestCase):
    invoke = foundation.WorldctlTests.invoke

    def setUp(self):
        foundation.WorldctlTests.setUp(self)
        for spec in experiments.REGISTRY.values():
            for name in (str(spec["script"]), str(spec["reference"])):
                path = self.root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(worldctl.ROOT / name, path)
        self.tae = self.root / "tae.pth"
        self.holdout = self.root / "holdout.pt"
        self.tae.write_text("not loaded")
        self.holdout.write_text("not loaded")
        self.output = self.root / "new experiment"

    def preview(self, *args):
        return self.invoke(
            "experiment",
            "plan",
            "--experiment",
            "preview-handoff",
            "--timeout",
            "120",
            "--out-dir",
            str(self.output),
            "--tae",
            str(self.tae),
            "--holdout",
            "scene=" + str(self.holdout),
            *args,
        )

    def test_catalog_only_lists_named_tracked_sources(self):
        (self.root / "untracked_experiment.py").write_text(
            "raise Exception('do not import')"
        )
        result = self.invoke("experiment", "list")
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(
            {x["id"] for x in result["data"]["experiments"]},
            {"preview-handoff", "memory-gate"},
        )
        self.assertTrue(
            all(not x["run_available"] for x in result["data"]["experiments"])
        )

    def test_changed_source_is_visible_and_cannot_be_planned(self):
        (self.root / "preview_2a_handoff.py").write_text("changed")
        self.assertEqual(self.invoke("experiment", "list")["status"], "PARTIAL")
        self.assertEqual(self.preview()["exit_code"], 3)

    def test_preview_plan_has_original_argv_and_does_not_create_output(self):
        before = self.holdout.read_bytes()
        data = self.preview()["data"]
        self.assertEqual(data["argv"][1], str(self.root / "preview_2a_handoff.py"))
        self.assertIn("scene=" + str(self.holdout), data["argv"])
        self.assertTrue(data["plan_only"])
        self.assertFalse(data["budget_enforced"])
        self.assertFalse(data["execution_authorized"])
        self.assertFalse(self.output.exists())
        self.assertEqual(self.holdout.read_bytes(), before)

    def test_memory_plan_retains_historical_configuration(self):
        directories = [self.root / x for x in ("weights", "assets", "scene")]
        for path in directories:
            path.mkdir()
        result = self.invoke(
            "experiment",
            "plan",
            "--experiment",
            "memory-gate",
            "--timeout",
            "300",
            "--out-dir",
            str(self.output),
            "--tae",
            str(self.tae),
            "--weights",
            str(directories[0]),
            "--assets",
            str(directories[1]),
            "--base",
            str(directories[2]),
            "--y-cache",
            str(self.holdout),
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(
            result["data"]["configuration"]["original_area_argument"], "512x320"
        )
        self.assertEqual(result["data"]["configuration"]["chunks"], 65)
        self.assertFalse(result["data"]["release_qualified"])

    def test_unknown_experiment_and_missing_budget_are_argument_errors(self):
        for args in [
            [],
            ["--experiment", "private-script"],
            ["--experiment", "preview-handoff", "--timeout", "0"],
        ]:
            self.assertEqual(self.invoke("experiment", "plan", *args)["exit_code"], 2)

    def test_missing_input_or_existing_output_blocks(self):
        self.holdout.unlink()
        self.assertEqual(self.preview()["exit_code"], 3)
        self.holdout.write_text("not loaded")
        self.output.mkdir()
        self.assertEqual(self.preview()["exit_code"], 3)

    def test_invalid_or_duplicate_tags_are_not_silently_accepted(self):
        self.assertEqual(
            self.preview("--holdout", "scene=" + str(self.holdout))["exit_code"], 2
        )
        self.assertEqual(
            self.preview("--holdout", "bad-tag-with-!=" + str(self.holdout))[
                "exit_code"
            ],
            2,
        )

    def test_dry_run_and_plan_never_launch_original_script(self):
        original = worldctl.subprocess.run

        def guarded(argv, **kwargs):
            self.assertEqual(argv[0], "git")
            return original(argv, **kwargs)

        with patch.object(worldctl.subprocess, "run", guarded):
            result = self.preview("--dry-run")
        self.assertEqual(result["exit_code"], 0)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("wan", sys.modules)

    def test_actual_entrypoint_error_remains_single_json(self):
        child = subprocess.run(
            [
                sys.executable,
                str(worldctl.ROOT / "worldctl.py"),
                "experiment",
                "plan",
                "--experiment",
                "preview-handoff",
                "--timeout",
                "120",
                "--out-dir",
                str(self.output),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(child.returncode, 3)
        self.assertEqual(len(child.stdout.splitlines()), 1)
        self.assertFalse(child.stderr)
        self.assertEqual(
            json.loads(child.stdout)["errors"][0]["code"],
            "experiment_input_unavailable",
        )


if __name__ == "__main__":
    unittest.main()
