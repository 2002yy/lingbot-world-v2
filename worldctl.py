#!/usr/bin/env python3
"""Read-only LingBot operations; no model import or GPU initialization."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parent
OWNERS = (
    "docs/MAIN_INTEGRATION_2026-10-20.md",
    "docs/RC_FROZEN.md",
    "docs/deployment/RTX5060_8GB_CURRENT_AUTHORITY.md",
)
RUNTIME_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "transformers",
    "diffusers",
    "flash-attn",
)


class CLIError(Exception):
    def __init__(self, code: int, reason: str, message: str) -> None:
        self.code, self.reason, self.message = code, reason, message


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CLIError(2, "invalid_arguments", "Invalid arguments; see --help.")


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--help", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=argparse.SUPPRESS)


def parser_for_cli() -> Parser:
    parser = Parser(description=__doc__, add_help=False)
    common(parser)
    commands = parser.add_subparsers(dest="command")
    for name in ("status", "doctor", "help"):
        common(commands.add_parser(name, add_help=False))
    return parser


def git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "-c",
                "core.fsmonitor=false",
                "-C",
                str(root),
                *args,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        raise CLIError(3, "git_unavailable", "Git is unavailable.") from None
    except subprocess.TimeoutExpired:
        raise CLIError(5, "git_timeout", "Git exceeded its read deadline.") from None
    if result.returncode:
        raise CLIError(
            3, "repository_unavailable", "Git repository metadata is unavailable."
        )
    return result.stdout.strip()


def repository(root: Path) -> tuple[Path, str]:
    if not root.is_dir():
        raise CLIError(
            3, "repository_unavailable", "Repository directory is unavailable."
        )
    actual = Path(git(root, "rev-parse", "--show-toplevel")).resolve()
    head = git(actual, "rev-parse", "--verify", "HEAD")
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise CLIError(3, "invalid_head", "An exact committed HEAD is required.")
    return actual, head


def status(root: Path) -> dict[str, Any]:
    rows = git(
        root, "status", "--porcelain=v1", "--untracked-files=normal"
    ).splitlines()
    return dict(
        root=str(root),
        branch=git(root, "branch", "--show-current") or None,
        tracked_changes=sum(not x.startswith("??") for x in rows),
        untracked_entries=sum(x.startswith("??") for x in rows),
        untracked_count_semantics="Git normal-mode entries; directories are not expanded",
        authority_files={name: (root / name).is_file() for name in OWNERS},
        runtime_started=False,
    )


def doctor(root: Path) -> dict[str, Any]:
    checks = {"python_supported": (3, 10) <= sys.version_info[:2] < (4, 0)}
    checks.update(
        {
            name: (root / name).is_file()
            for name in (
                *OWNERS,
                "run.sh",
                "run.ps1",
                "gpu_cleanliness.py",
                "pyproject.toml",
            )
        }
    )
    packages: dict[str, dict[str, Any]] = {}
    for name in RUNTIME_PACKAGES:
        try:
            version = importlib.metadata.version(name)
            packages[name] = dict(status="installed", version=version)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = dict(status="missing", version=None)
        except (OSError, ValueError):
            packages[name] = dict(status="metadata_unavailable", version=None)
    return dict(
        checks=checks,
        scope="offline CLI prerequisites only",
        runtime_packages=packages,
        runtime_ready=None,
        gpu="not_queried",
        weights="not_inspected",
        network="not_requested",
        runtime_started=False,
        release_qualified=False,
    )


def check_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise CLIError(3, "output_exists", "Output already exists; refusing overwrite.")
    if not path.parent.is_dir() or any(
        x.is_symlink() for x in (path.parent, *path.parent.parents)
    ):
        raise CLIError(
            3,
            "output_unavailable",
            "An existing non-symlink output directory is required.",
        )


def save_new(path: Path, value: dict[str, Any]) -> None:
    temporary = None
    try:
        check_output(path)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    except OSError:
        raise CLIError(
            3, "output_unavailable", "Could not create a new report."
        ) from None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    result: dict[str, Any] = dict(
        schema_version="agent-cli-result-v1",
        repo="lingbot-world-v2",
        command="",
        head_sha=None,
        status="PASS",
        exit_code=0,
        duration_ms=0,
        artifacts=[],
        errors=[],
        data={},
    )
    try:
        options = parser_for_cli().parse_args(argv)
        dry_run = getattr(options, "dry_run", False)
        output = getattr(options, "output", None)
        if output is not None and not dry_run:
            check_output(output)
        if getattr(options, "help", False) or options.command == "help":
            result["command"] = "help"
            result["data"] = dict(
                commands=["status", "doctor", "help"],
                side_effects="read-only unless --output is explicit",
            )
        else:
            if options.command is None:
                raise CLIError(2, "missing_command", "Select a command; see --help.")
            root, target_head = repository(getattr(options, "repo", ROOT))
            _, result["head_sha"] = repository(ROOT)
            result["command"] = options.command
            result["data"] = (
                status(root) if options.command == "status" else doctor(root)
            )
            result["data"].update(
                requested_repo_head_sha=target_head, inspection_root=str(ROOT)
            )
            if options.command == "doctor" and not all(
                result["data"]["checks"].values()
            ):
                result.update(status="BLOCKED", exit_code=3)
                result["errors"] = [
                    dict(
                        code="offline_prerequisite_missing",
                        message="Required offline CLI files or Python are unavailable.",
                    )
                ]
        result["data"]["dry_run"] = dry_run
        result["duration_ms"] = round((time.monotonic() - started) * 1000)
        if output is not None and not dry_run:
            result["artifacts"] = [str(output.absolute())]
            save_new(output, result)
    except CLIError as error:
        result.update(
            status="FAIL" if error.code in (2, 4) else "BLOCKED",
            exit_code=error.code,
            artifacts=[],
            errors=[dict(code=error.reason, message=error.message)],
        )
    except KeyboardInterrupt:
        result.update(
            status="BLOCKED",
            exit_code=5,
            artifacts=[],
            errors=[dict(code="cancelled", message="Operation cancelled.")],
        )
    except (OSError, ValueError):
        result.update(
            status="BLOCKED",
            exit_code=3,
            artifacts=[],
            errors=[
                dict(code="local_read_failed", message="Local metadata is unavailable.")
            ],
        )
    if not result["artifacts"]:
        result["duration_ms"] = round((time.monotonic() - started) * 1000)
    print(json.dumps(result, ensure_ascii=True))
    exit_code: int = result["exit_code"]
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
