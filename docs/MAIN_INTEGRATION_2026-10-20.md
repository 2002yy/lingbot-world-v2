# Main integration and local worktree inventory — 2026-10-20

The user authorized organizing the active LingBot checkout and integrating its
mature committed work into main. This is repository integration, not a new GPU,
world-quality, display-latency or release qualification.

## Frozen inputs

- Remote main: `1895d300d8ac936401689b26389f51cbd36530eb`.
- Active committed source: `3ab1efd0fbcc070a122c207cf61bc8ffea018620` on
  `community/rtx5060-interactive-rc`, also the current GitHub default branch.
- Main is an ancestor of the source: 112 commits and 221 changed files.
- Local main was `1c3053d6fb66cf8326ea38cc324694d722e89b8d`, an earlier RC,
  so it must not be mistaken for the remote main baseline.
- Candidate branch: `codex/main-integration-20261020`, created in a separate
  clean worktree from the exact source commit. No untracked source was imported.

## Untracked inventory: preserve first

The original checkout had no tracked modifications, but 1,259 untracked files:

| Group | Files | Bytes |
| --- | ---: | ---: |
| examples | 743 | 550,762,612 |
| repository root | 516 | 2,133,132 |

Extensions include 492 NPY, 251 JPG, 305 shell scripts, 143 Python files,
47 OUT, 16 TXT, 4 LOG and one CUDA source file. Some Python files are experimental
code rather than disposable cache. The complete exact-path/size inventory is
retained locally as `lingbot-untracked-inventory-20261020.json` outside Git.
Already-ignored artifacts are outside this count and remain untouched as well.

No untracked file was deleted, moved, committed or hidden by a blanket ignore
rule. Models, private examples and experiment evidence remain in the original
checkout. Integration does not declare these experiments finished or qualified.
Further promotion of an experiment requires a named, reviewed slice.

## Necessary integration repair

With CUDA explicitly disabled, nine of the ten existing offline test scripts
failed. Control reduction imported `cam_controller`, which imported
`wan.utils.cam_utils`; the package initializer imported T5, whose default device
expression initialized CUDA. CPU-only camera steps never need interpolation.

Move that existing import into `CameraController.to_frames`, where interpolation
is actually used. The function, call arguments, control integration, production
model implementation, GPU cleanliness thresholds and inference settings are
unchanged. Two subprocess regressions first failed on the old implementation and
now prove camera import/control reduction does not load `wan` or `torch`.

Three historical extra blank EOF lines are removed solely to satisfy diff-check
against remote main: `cadence_recheck.py`, `push_angular.py`, `push_physics.py`.
No broad formatting or inference refactor was performed.

## Verification

- All 11 tracked offline test scripts pass, 129 checks total, with
  `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- All 175 tracked Python files parse without syntax errors.
- Ruff passes on the changed behavior module and new regression; the new test
  passes format check. Historical source formatting is not rewritten.
- Focused strict mypy comparison with imports skipped: baseline 23 errors,
  candidate 23, zero newly introduced messages. This is not repository-wide
  type cleanliness, nor a new frozen baseline.
- Diff-check against remote main passes after the three EOF fixes.
- Original checkout tracked state and its untracked inventory are preserved.
- No live inference, weights download, heavy GPU experiment, paid model call,
  display qualification or release upload was run.

There are no GitHub Actions workflows in the candidate's tracked tree and no
existing runs were returned for this repository. Local evidence is therefore
reported separately; it is not described as passing remote CI. Use an expected
candidate head when merging the integration PR. This report is recoverable from
its commit; the PR supplies the final integration/merge identities.

## Next

After main integration, base CLI work on that main commit. Implement worldctl
doctor and experiment list/plan first, reusing run.sh/run.ps1 and existing gates.
No new GPU experiment is implied by this integration. Keep the frozen RC tags
and the existing default-branch setting unchanged unless separately instructed.
