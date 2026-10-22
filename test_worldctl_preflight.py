"""Preflight keeps metadata checks separate from GPU/model readiness."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import test_worldctl_experiments as catalogue
import worldctl_preflight as preflight


class PreflightTests(unittest.TestCase):
    invoke = catalogue.ExperimentTests.invoke

    def setUp(self):
        catalogue.ExperimentTests.setUp(self)
        tae_source = self.root / "ai/taehv/taehv.py"
        tae_source.parent.mkdir(parents=True)
        tae_source.write_text("raise RuntimeError('never import me')")
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "commit", "-qm", "inputs"], check=True
        )
        for target, name, value in [
            (preflight, "ROOT", self.root),
            (Path, "home", None),
        ]:
            context = (
                patch.object(target, name, return_value=self.root)
                if value is None
                else patch.object(target, name, value)
            )
            context.start()
            self.addCleanup(context.stop)
        context = patch.object(
            preflight.importlib.metadata, "version", return_value="fixture"
        )
        context.start()
        self.addCleanup(context.stop)

    def call(self, *extra):
        return self.invoke(
            "experiment",
            "preflight",
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
            "--purpose",
            "Check preview handoff",
            *extra,
        )

    def test_static_pass_never_claims_runtime_authority(self):
        imported = set(sys.modules)
        before = self.holdout.read_bytes()
        result = self.call("--trusted-inputs")
        self.assertEqual(result["exit_code"], 0)
        data = result["data"]
        self.assertEqual(len(data["input_metadata"]), 2)
        self.assertFalse(data["runtime_started"])
        self.assertFalse(data["execution_authorized"])
        self.assertFalse(data["provenance_verified"])
        self.assertFalse(data["dependency_compatibility_verified"])
        self.assertEqual(data["gpu"], "not_queried")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.holdout.read_bytes(), before)
        self.assertFalse({"torch", "wan", "taehv"} & (set(sys.modules) - imported))

    def test_trust_acknowledgement_required(self):
        self.assertEqual(self.call()["exit_code"], 3)

    def test_blank_or_excessive_purpose_is_rejected(self):
        for purpose in (" ", "x" * 513):
            self.assertEqual(self.call("--purpose", purpose)["exit_code"], 2)

    def test_dirty_or_untracked_checkout_blocks(self):
        (self.root / "unknown.py").write_text("do not execute")
        self.assertEqual(self.call("--trusted-inputs")["exit_code"], 3)

    def test_ancestor_dotenv_is_not_read(self):
        (self.root.parent / ".env").write_text("private-content")
        result = self.call("--trusted-inputs")
        self.assertEqual(result["exit_code"], 3)
        self.assertNotIn("private-content", str(result))

    def test_missing_metadata_blocks_without_importing_packages(self):
        with patch.object(
            preflight.importlib.metadata,
            "version",
            side_effect=preflight.importlib.metadata.PackageNotFoundError,
        ):
            self.assertFalse(
                self.call("--trusted-inputs")["data"]["checks"][
                    "package_metadata_present"
                ]
            )

    def test_missing_tae_source_blocks(self):
        with patch.object(Path, "home", return_value=self.root.parent):
            self.assertEqual(self.call("--trusted-inputs")["exit_code"], 3)

    def test_empty_file_blocks_even_when_git_clean(self):
        self.holdout.write_bytes(b"")
        subprocess.run(["git", "-C", str(self.root), "add", "."], check=True)
        subprocess.run(
            ["git", "-C", str(self.root), "commit", "-qm", "empty"], check=True
        )
        self.assertEqual(
            self.call("--trusted-inputs")["errors"][0]["code"], "experiment_input_empty"
        )

    def test_symlink_parent_is_rejected(self):
        link = self.root.parent / "linked"
        link.symlink_to(self.root, target_is_directory=True)
        result = self.call("--trusted-inputs", "--tae", str(link / self.tae.name))
        self.assertEqual(result["errors"][0]["code"], "experiment_input_symlink")

    def test_other_repository_cannot_be_execution_preflight(self):
        with patch.object(preflight, "ROOT", self.root.parent):
            self.assertEqual(self.call("--trusted-inputs")["exit_code"], 3)


if __name__ == "__main__":
    unittest.main()
