"""Content identity is bounded and never an execution/provenance grant."""

import hashlib
import shutil
import unittest
from unittest.mock import patch

import test_worldctl_preflight as preparation
import worldctl_fingerprint as fingerprints


class FingerprintTests(unittest.TestCase):
    invoke = preparation.PreflightTests.invoke

    def setUp(self):
        preparation.PreflightTests.setUp(self)

    def call(self, *extra):
        return self.invoke(
            "experiment",
            "fingerprint",
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
            "Snapshot preview inputs",
            "--trusted-inputs",
            "--max-bytes",
            "1048576",
            "--max-files",
            "32",
            "--hash-timeout",
            "10",
            *extra,
        )

    def memory(self, *extra):
        dirs = []
        for name in ("weights", "assets", "scene"):
            path = self.root.parent / name
            path.mkdir(exist_ok=True)
            (path / "content.bin").write_bytes(b"fixture bytes")
            dirs.append(path)
        return self.invoke(
            "experiment",
            "fingerprint",
            "--experiment",
            "memory-gate",
            "--timeout",
            "120",
            "--out-dir",
            str(self.output),
            "--tae",
            str(self.tae),
            "--weights",
            str(dirs[0]),
            "--assets",
            str(dirs[1]),
            "--base",
            str(dirs[2]),
            "--y-cache",
            str(self.holdout),
            "--purpose",
            "Snapshot memory inputs",
            "--trusted-inputs",
            "--max-bytes",
            "1048576",
            "--max-files",
            "32",
            "--hash-timeout",
            "10",
            *extra,
        )

    def test_complete_bytes_are_hashed_and_not_deserialized(self):
        data = self.call()["data"]
        self.assertTrue(data["fingerprint_created"])
        payload = data["fingerprint"]["payload"]
        self.assertEqual(data["fingerprint"]["sha256"], fingerprints.digest(payload))
        holdout = next(
            item for item in payload["inputs"] if item["role"] == "holdout:scene"
        )
        self.assertEqual(
            holdout["files"][0]["sha256"],
            hashlib.sha256(self.holdout.read_bytes()).hexdigest(),
        )
        self.assertFalse(data["execution_authorized"])
        self.assertFalse(data["runtime_started"])
        self.assertFalse(data["provenance_verified"])
        self.assertFalse(self.output.exists())

    def test_relocated_identical_inputs_keep_identity(self):
        before = self.call()["data"]["fingerprint"]
        moved = self.root.parent / "relocated.pth"
        shutil.copyfile(self.tae, moved)
        after = self.call("--tae", str(moved))["data"]["fingerprint"]
        self.assertEqual(before, after)

    def test_changed_bytes_change_identity(self):
        moved = self.root.parent / "different.pth"
        moved.write_bytes(b"different bytes")
        before = self.call()["data"]["fingerprint"]["sha256"]
        after = self.call("--tae", str(moved))["data"]["fingerprint"]["sha256"]
        self.assertNotEqual(before, after)

    def test_byte_and_entry_limits_never_emit_partial_fingerprint(self):
        for args in (("--max-bytes", "1"), ("--max-files", "1")):
            data = self.call(*args)
            self.assertEqual(data["exit_code"], 3)
            self.assertNotIn("fingerprint", data["data"])

    def test_explicit_valid_budgets_required(self):
        for name in ("--max-bytes", "--max-files", "--hash-timeout"):
            self.assertEqual(self.call(name, "0")["exit_code"], 2)

    def test_timeout_emits_no_fingerprint(self):
        with patch.object(
            fingerprints.time, "monotonic", side_effect=[10, 10, 11, 11, 11, 11, 11]
        ):
            data = self.call("--hash-timeout", "1")
        self.assertEqual(data["exit_code"], 5)
        self.assertNotIn("fingerprint", data["data"])

    def test_dry_run_does_not_hash(self):
        data = self.call("--dry-run")["data"]
        self.assertFalse(data["fingerprint_created"])
        self.assertFalse(data["input_contents_hashed"])
        self.assertNotIn("fingerprint", data)

    def test_recursive_directory_inventory_includes_empty_dirs(self):
        self.memory()
        (self.root.parent / "weights/empty").mkdir()
        data = self.memory()["data"]
        weights = next(
            item
            for item in data["fingerprint"]["payload"]["inputs"]
            if item["role"] == "weights"
        )
        self.assertEqual(weights["directories"], [".", "empty"])
        self.assertEqual(weights["files"][0]["path"], "content.bin")

    def test_nested_symlink_is_rejected(self):
        self.memory()
        (self.root.parent / "weights/link").symlink_to(self.holdout)
        self.assertEqual(self.memory()["errors"][0]["code"], "fingerprint_symlink")

    def test_mid_read_mutation_is_detected(self):
        original = fingerprints.os.fstat
        calls = 0

        def observe(fd):
            nonlocal calls
            calls += 1
            if calls == 2:
                with self.holdout.open("ab") as handle:
                    handle.write(b"changed")
            return original(fd)

        with patch.object(fingerprints.os, "fstat", side_effect=observe):
            result = self.call()
        self.assertEqual(result["exit_code"], 3)
        self.assertNotIn("fingerprint", result["data"])


if __name__ == "__main__":
    unittest.main()
