"""Admission for the existing CPU-only regression scripts, never GPU experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

from worldctl import CLIError, ROOT, git
from worldctl_execution import execute

BASELINE = "7eff1b3bb5f4c8b3bd498f51e15043e921761d26"
SCRIPTS = (
    "test_camera_import.py",
    "test_commit_before_mark.py",
    "test_control_exactly_once.py",
    "test_display_isolation.py",
    "test_input_acknowledgement.py",
    "test_interactive_runtime.py",
    "test_preemption_prerequisites.py",
    "test_prewarm_and_reference.py",
    "test_renderer_submit.py",
    "test_single_rebase_preemption.py",
    "test_stale_frontier.py",
)


def stable(root: Path, head: str) -> bool:
    return (
        git(root, "rev-parse", "HEAD") == head
        and not git(root, "status", "--porcelain=v1", "--untracked-files=normal")
        and not any(
            (folder / ".env").exists() or (folder / ".env").is_symlink()
            for folder in (root, *root.parents)
        )
    )


def source_matches(root: Path, script: str) -> bool:
    path = root / script
    if not path.is_file() or path.is_symlink() or path.resolve().parent != root:
        return False
    expected = git(root, "rev-parse", BASELINE + ":" + script)
    return git(root, "hash-object", "--no-filters", str(path)) == expected


def run(root: Path, options: argparse.Namespace) -> dict[str, Any]:
    if (
        options.test_command != "run"
        or options.suite != "offline"
        or options.timeout is None
        or not 1 <= options.timeout <= 3600
    ):
        raise CLIError(
            2,
            "invalid_test_selection",
            "Select test run --suite offline with an explicit 1-3600 second budget.",
        )
    if root != ROOT:
        raise CLIError(
            3,
            "execution_root_mismatch",
            "Test execution is limited to the actual CLI checkout.",
        )
    head = git(root, "rev-parse", "HEAD")
    if not stable(root, head):
        raise CLIError(
            3,
            "execution_checkout_not_clean",
            "A clean checkout without ancestor .env files is required; no credentials are read.",
        )
    if any(not source_matches(root, name) for name in SCRIPTS):
        raise CLIError(
            3,
            "offline_source_changed",
            "The offline allowlist must match its frozen reviewed source.",
        )
    commands = [[sys.executable, str(root / name)] for name in SCRIPTS]

    def guard(argv: list[str]) -> bool:
        try:
            return stable(root, head) and source_matches(root, Path(argv[1]).name)
        except CLIError:
            return False

    result = execute(
        root,
        commands,
        options.timeout,
        dry_run=getattr(options, "dry_run", False),
        before_step=guard,
        record=True,
        record_context=dict(
            suite="offline", execution_head_sha=head, source_baseline=BASELINE
        ),
    )
    result.update(
        suite="offline",
        source_baseline=BASELINE,
        execution_head_sha=head,
        runtime_started=False,
        model_qualified=False,
        artifacts_scope="isolated child stdout/stderr files; not research/model evidence",
    )
    return result
