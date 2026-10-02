# RC frozen at ec838b4

## Status

    RC baseline        ec838b4   (tracked tree clean, 93 commits)
    active work        STOPPED
    purpose            immutable comparison baseline for any future work

This is a release candidate, not "the best experimental code so far". The distinction
matters because everything below is a decision with a traceable evidence source, and a
future reader should be able to check a claim rather than relitigate a preference.

Note on the two commit ids: this document is a snapshot written at `ec838b4`, which is
where the frozen behaviour was established. The tagged release is `1c3053d`, one commit
later, containing only the pre-publish audit that untracked three scratch files carrying
absolute local paths. No code changed between them. The tag `rtx5060-interactive-rc1`
points at `1c3053d`.

## Erratum: the handoff image-difference metric is single-channel

    play.py's handoff evaluation accidentally reduced the [B, T, C, H, W] decoder
    output along the CHANNEL axis rather than the temporal one. After dropping B,
    the dim()==4 branch tests dim 1, which on that layout is C, not T.

    This does NOT affect model generation, committed state, or the t0 -> t3 latency
    measurements. Those are unaffected and remain as published.

    It DOES affect the absolute handoff image-difference metric. The reported ~0.25x
    peak-step ratio is primarily a property of the 50 ms / 60 Hz linear blend
    construction -- a linear blend in pixel space is collinear, so the total path
    length is additive and the peak step divides by (n+1) by construction -- not a
    colour-complete RGB quality measurement. The ratio is a real property of the
    construction; only the absolute per-pixel magnitude is single-channel.

    Found while building §Demo-1, which extracts RGB correctly and does not reuse
    play.py's reduction.

The RC is not modified by this erratum. The tooling is frozen deliberately, the tagged
commit is public, and silently editing a published measurement would destroy the
comparison baseline the whole document exists to provide. The historical evidence stands;
the interpretation is corrected here, in one place, with the code path named.

## Why each boundary exists, and where the evidence is

    no fixed 2-step / 1-step     KV divergence is monotonic and unplateaued over 10
                                 chunks: relL2 0.065 -> 0.274 (2-step), 0.095 -> 0.438
                                 (1-step), with per-chunk increments that do not
                                 shrink.   docs/QUANTUM_1A_STEP_ABLATION.md
    no further KV work           the KV update is 0.945x a denoise step, i.e. one
                                 more full forward; the rolling memmove is 1.8%
                                 (~3 ms).   docs/KV_1A_DECOMPOSITION.md
    no learned preview head      Arm T fails the capacity gate at 0.8963 against a
                                 required 0.95 even as a nine-sample overfit, and the
                                 head does not generalise: H1 fails on edgeSSIM and
                                 H2 tracks scene difference at -0.049 / -0.062 /
                                 -0.218.   docs/PREVIEW_HEAD_1B_GENERALIZATION.md
    no async preview stream      the side-stream decode does not overlap at all: the
                                 preview completes at 766 ms, when the chunk does, and
                                 the DiT slows by 28.5 ms.   docs/PREVIEW_1A_*.md
    variant D as the preview     cross-scene PASS with zero gross failures and a
                                 degradation band of 0.015-0.079 that is NOT monotone
                                 in scene difference, i.e. a constant training-free
                                 loss rather than domain sensitivity.
                                 docs/PREVIEW_1C_VARIANT_D_RELEASE.md
    50 ms handoff blend          peak jump 0.0475 -> 0.0119 (0.25x) with the total
                                 path length unchanged at 1.00x and a negligible edge
                                 dip of 0.002.   docs/PREVIEW_2B_SMOOTHING.md
    input-to-display unmeasured  no renderer and no present signal exist anywhere in
                                 this tree.   docs/LATENCY_1A_CONTRACT_AND_AUDIT.md

## What the product semantics are

    ~231 ms   causally-related preview
    ~790 ms   authoritative frame ready
    +50 ms    handoff smoothing
    ~840 ms   100% authoritative display semantics

    renderer submit   UNKNOWN
    physical present  UNKNOWN
    input -> display  UNKNOWN

Those three are unknown on purpose. Reporting a proxy as a measurement is what earlier
stages spent effort undoing, and the discipline of knowing what has not been measured is
worth more here than another benchmark table.

## Reopening conditions

Nothing below should start merely because time has passed. Each needs its specific
condition.

    EarlyExit   only with a reliable per-chunk safety predictor. Note the unfavourable
                correlation: the chunks that most need low latency are the ones where a
                control just arrived and the world is changing most, which are
                plausibly the least safe to exit early on. It would likely save time on
                quiet chunks and help interaction latency less than it appears.

    Pipeline    only with real concurrency to overlap -- a second device, or stages that
                genuinely can run concurrently. The same-card separate-stream experiment
                already showed there is no overlap to be had on this hardware.

    Renderer    only if input-to-display is actually wanted. It requires introducing a
                real renderer AND establishing whether that renderer exposes a
                present-completion signal. Without the second half, t5 stays
                unmeasurable and the work buys nothing.

    anything else   driven by a real problem found in use, not by the profiler.

## Where the next genuine motivation should come from

Not from finding the next optimisation in a profile. From what actual use exposes:

    preview still feels slow at 231 ms
    high-speed turns make the preview visibly unstable
    memory problems over long sessions
    install failures on another 5060-class machine
    visible ghosting in the blend on some scenes

Any of those is a stronger research motivator than a theoretical remaining lever,
because it comes with a reproduction rather than an estimate.

## Handoff state

    repo            ~/ai/lingbot-world-v2, HEAD ec838b4, tracked tree clean
    entrypoints     ./setup.sh and ./run.sh play, defaulting to the frozen stack at the
                    source level rather than approximately
    smoke           ./run.sh smoke -> 11/11 including the GPU path
    unit suites     54/54 across four files plus 4 commit-ordering checks
    findings        the full record, including every negative result, is in the
                    external findings document
    known gaps      no t4/t5 instrumentation; nine chunks per scene over three scenes,
                    one seed per scene, for everything in the preview line
