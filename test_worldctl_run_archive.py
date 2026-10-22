"""Archives remain inspectable without source files and never upgrade outcomes."""

import argparse
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import worldctl
from worldctl_execution import execute
from worldctl_fingerprint import digest
from worldctl_run_archive import (
    archive_run,
    inspect_archive,
    inspect_payload,
    read_blob,
)


class RunArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def create(self, source="print('fixture')", timeout=2):
        with contextlib.redirect_stderr(io.StringIO()):
            result = execute(
                self.root,
                [[sys.executable, "-c", source]],
                timeout,
                record=True,
                record_context=dict(suite="offline", execution_head_sha="a" * 40),
            )
        record = Path(result["execution_record"])
        self.addCleanup(shutil.rmtree, record.parent, True)
        return record

    def build(self, record, dry_run=False):
        return archive_run(
            argparse.Namespace(
                record=record, output=self.root / "saved.json", dry_run=dry_run
            )
        )

    def save(self, data):
        path = self.root / "saved.json"
        path.write_text(
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
        return path

    def invalid(self, operation, code=3):
        with self.assertRaises(worldctl.CLIError) as error:
            operation()
        self.assertEqual(error.exception.code, code)

    def test_roundtrip_survives_source_removal_without_runtime_authority(self):
        record = self.create()
        saved = self.save(self.build(record))
        shutil.rmtree(record.parent)
        result = inspect_archive(saved)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["archived_files"], 3)
        self.assertEqual(result["recorded_context"]["execution_head_sha"], "a" * 40)
        for key in (
            "execution_origin_authenticated",
            "run_binding_verified",
            "release_qualified",
            "runtime_started",
        ):
            self.assertFalse(result[key])

    def test_failed_execution_remains_failure(self):
        result = inspect_archive(
            self.save(self.build(self.create("raise SystemExit(7)")))
        )
        self.assertEqual(result["exit_code"], 4)
        self.assertTrue(result["archive_verified"])

    def test_timeout_remains_timeout(self):
        result = inspect_archive(
            self.save(self.build(self.create("import time; time.sleep(10)", 0.1)))
        )
        self.assertEqual(result["exit_code"], 5)

    def test_dry_run_cannot_be_used_as_complete_archive(self):
        result = self.build(self.create(), True)
        self.assertFalse(result["archive_created"])
        self.assertNotIn("archive", result)
        self.invalid(lambda: inspect_archive(self.save(result)))

    def test_partial_source_record_is_rejected(self):
        record = self.create()
        record.write_bytes(record.read_bytes().splitlines(True)[0])
        self.invalid(lambda: self.build(record))

    def test_changed_source_log_is_rejected(self):
        record = self.create()
        (record.parent / "0.stdout").write_text("different")
        self.invalid(lambda: self.build(record), 4)

    def test_snapshot_hash_mismatch_is_rejected(self):
        archive = self.build(self.create())["archive"]
        archive["files"][1]["base64"] = "eA=="
        archive["sha256"] = digest({k: v for k, v in archive.items() if k != "sha256"})
        self.invalid(lambda: inspect_payload(archive))

    def test_traversal_and_duplicate_files_are_rejected_after_rehash(self):
        for mode in ("traversal", "duplicate"):
            archive = self.build(self.create())["archive"]
            if mode == "traversal":
                archive["files"][0]["name"] = "../outside"
            else:
                archive["files"].append(dict(archive["files"][0]))
            archive["sha256"] = digest(
                {k: v for k, v in archive.items() if k != "sha256"}
            )
            self.invalid(lambda: inspect_payload(archive))

    def test_extra_file_is_rejected(self):
        archive = self.build(self.create())["archive"]
        archive["files"].append(dict(archive["files"][1], name="1.stdout"))
        archive["sha256"] = digest({k: v for k, v in archive.items() if k != "sha256"})
        self.invalid(lambda: inspect_payload(archive))

    def test_oversize_budget_is_rejected_before_decode(self):
        archive = self.build(self.create())["archive"]
        archive["files"][0]["bytes"] = 6 * 1024**2
        archive["sha256"] = digest({k: v for k, v in archive.items() if k != "sha256"})
        self.invalid(lambda: inspect_payload(archive))

    def test_missing_selections_are_parameter_errors(self):
        self.invalid(
            lambda: archive_run(argparse.Namespace(record=None, output=None)), 2
        )
        self.invalid(
            lambda: archive_run(argparse.Namespace(record=self.create(), output=None)),
            2,
        )
        self.invalid(lambda: inspect_archive(None), 2)

    def test_symlinks_and_source_size_limit_are_rejected(self):
        source = self.root / "small"
        source.write_bytes(b"abc")
        alias = self.root / "alias"
        alias.symlink_to(source)
        self.invalid(lambda: read_blob(alias, 100))
        self.invalid(lambda: read_blob(source, 2))

    def test_success_header_cannot_mask_failed_embedded_execution(self):
        saved = self.save(self.build(self.create("raise SystemExit(7)")))
        data = json.loads(saved.read_text())
        data.update(status="PASS", exit_code=0)
        saved.write_text(json.dumps(data))
        self.invalid(lambda: inspect_archive(saved))

    def test_real_cli_output_roundtrip_and_refusal_to_overwrite(self):
        record = self.create()
        output = self.root / "actual.json"
        command = [
            sys.executable,
            str(worldctl.ROOT / "worldctl.py"),
            "test",
            "archive-run",
            "--record",
            str(record),
            "--output",
            str(output),
            "--json",
        ]
        first = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(first.stderr, "")
        self.assertTrue(json.loads(first.stdout)["data"]["archive_created"])
        original = output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(second.returncode, 3)
        self.assertEqual(output.read_bytes(), original)
        third = subprocess.run(
            [
                sys.executable,
                str(worldctl.ROOT / "worldctl.py"),
                "test",
                "inspect-archive",
                "--archive",
                str(output),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(third.returncode, 0, third.stderr)
        self.assertTrue(json.loads(third.stdout)["data"]["archive_verified"])


if __name__ == "__main__":
    unittest.main()
