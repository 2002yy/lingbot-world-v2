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
    gpu = commands.add_parser("gpu", add_help=False)
    common(gpu)
    gpu_commands = gpu.add_subparsers(dest="gpu_command")
    gpu_check = gpu_commands.add_parser("check", add_help=False)
    common(gpu_check)
    gpu_check.add_argument("--timeout", type=int)
    test = commands.add_parser("test", add_help=False)
    common(test)
    tests = test.add_subparsers(dest="test_command")
    run = tests.add_parser("run", add_help=False)
    common(run)
    run.add_argument("--suite", default="")
    run.add_argument("--timeout", type=int)
    inspect_run = tests.add_parser("inspect-run", add_help=False)
    common(inspect_run)
    inspect_run.add_argument("--record", type=Path)
    archive_run = tests.add_parser("archive-run", add_help=False)
    common(archive_run)
    archive_run.add_argument("--record", type=Path)
    inspect_archive = tests.add_parser("inspect-archive", add_help=False)
    common(inspect_archive)
    inspect_archive.add_argument("--archive", type=Path)
    experiment = commands.add_parser("experiment", add_help=False)
    common(experiment)
    experiments = experiment.add_subparsers(dest="experiment_command")
    common(experiments.add_parser("list", add_help=False))
    inspect = experiments.add_parser("inspect", add_help=False)
    common(inspect)
    inspect.add_argument("--experiment", default="")
    inspect.add_argument("--result", type=Path)
    associate = experiments.add_parser("associate", add_help=False)
    common(associate)
    associate.add_argument("--experiment", default="")
    associate.add_argument("--result", type=Path)
    associate.add_argument("--fingerprint", type=Path)
    associate.add_argument("--purpose", default="")
    verify_association = experiments.add_parser("verify-association", add_help=False)
    common(verify_association)
    verify_association.add_argument("--association", type=Path)
    verify_association.add_argument("--result", type=Path)
    verify_association.add_argument("--fingerprint", type=Path)
    compare = experiments.add_parser("compare", add_help=False)
    common(compare)
    compare.add_argument("--left", type=Path)
    compare.add_argument("--right", type=Path)
    for name in ("plan", "preflight", "fingerprint"):
        plan = experiments.add_parser(name, add_help=False)
        experiment_options(plan)
        if name in {"preflight", "fingerprint"}:
            plan.add_argument("--purpose", default="")
            plan.add_argument("--trusted-inputs", action="store_true")
        if name == "fingerprint":
            plan.add_argument("--max-bytes", type=int)
            plan.add_argument("--max-files", type=int)
            plan.add_argument("--hash-timeout", type=int)
    return parser


def experiment_options(plan: Parser) -> None:
    common(plan)
    plan.add_argument("--experiment", default="")
    plan.add_argument("--timeout", type=int)
    plan.add_argument("--out-dir", type=Path)
    plan.add_argument("--tae", type=Path)
    plan.add_argument("--holdout", action="append", default=[])
    plan.add_argument("--weights", type=Path)
    plan.add_argument("--assets", type=Path)
    plan.add_argument("--base", type=Path)
    plan.add_argument("--y-cache", type=Path)


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
                commands=[
                    "status",
                    "doctor",
                    "gpu check",
                    "test run",
                    "test inspect-run",
                    "test archive-run",
                    "test inspect-archive",
                    "experiment list",
                    "experiment plan",
                    "experiment preflight",
                    "experiment fingerprint",
                    "experiment compare",
                    "experiment inspect",
                    "experiment associate",
                    "experiment verify-association",
                    "help",
                ],
                side_effects="test run executes offline checks; archive-run packages selected journal/log bytes; --output writes a new report",
            )
        else:
            if options.command is None:
                raise CLIError(2, "missing_command", "Select a command; see --help.")
            root, target_head = repository(getattr(options, "repo", ROOT))
            _, result["head_sha"] = repository(ROOT)
            result["command"] = options.command
            if options.command == "gpu":
                from worldctl_gpu import check as check_gpu

                result["command"] = "gpu " + (options.gpu_command or "")
                result["data"] = check_gpu(root, options)
                for key in ("status", "exit_code", "errors"):
                    result[key] = result["data"][key]
            elif options.command == "test":
                result["command"] = "test " + (options.test_command or "")
                if options.test_command == "archive-run":
                    from worldctl_run_archive import archive_run

                    result["data"] = archive_run(options)
                elif options.test_command == "inspect-archive":
                    from worldctl_run_archive import inspect_archive

                    result["data"] = inspect_archive(options.archive)
                elif options.test_command == "inspect-run":
                    from worldctl_run_inspect import inspect_run

                    result["data"] = inspect_run(options.record)
                else:
                    from worldctl_tests import run as run_tests

                    result["data"] = run_tests(root, options)
                for key in ("status", "exit_code", "errors", "artifacts"):
                    result[key] = result["data"][key]
            elif options.command == "experiment":
                from worldctl_experiments import catalogue, plan as experiment_plan

                if options.experiment_command not in {
                    "list",
                    "plan",
                    "preflight",
                    "fingerprint",
                    "compare",
                    "inspect",
                    "associate",
                    "verify-association",
                }:
                    raise CLIError(
                        2,
                        "missing_experiment_command",
                        "Select experiment list, plan or preflight.",
                    )
                result["command"] = "experiment " + options.experiment_command
                if options.experiment_command == "verify-association":
                    from worldctl_association_verify import verify_association

                    result["data"] = verify_association(root, options)
                elif options.experiment_command == "associate":
                    from worldctl_association import associate

                    result["data"] = associate(root, options)
                elif options.experiment_command == "inspect":
                    from worldctl_result import inspect_result

                    result["data"] = inspect_result(root, options)
                elif options.experiment_command == "compare":
                    from worldctl_compare import compare

                    result["data"] = compare(options)
                elif options.experiment_command == "fingerprint":
                    from worldctl_fingerprint import fingerprint

                    result["data"] = fingerprint(root, options)
                elif options.experiment_command == "preflight":
                    from worldctl_preflight import preflight

                    result["data"] = preflight(root, options)
                else:
                    result["data"] = (
                        catalogue(root)
                        if options.experiment_command == "list"
                        else experiment_plan(root, options)
                    )
                for key in ("status", "exit_code", "errors"):
                    result[key] = result["data"][key]
            else:
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
            result["artifacts"] = [*result["artifacts"], str(output.absolute())]
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
    # Direct-script dispatch and imported adapters must share one error class.
    sys.modules.setdefault("worldctl", sys.modules[__name__])
    raise SystemExit(main())
