# LingBot-World 2.0 / 1.3B causal-fast on an RTX 5060 Laptop 8GB

A reproducible deployment of a **single-session interactive world model** on consumer
8 GB hardware, with the interactive controls, the preview path and the latency
semantics all measured rather than asserted.

## Quick start

    git clone <repo> && cd lingbot-world-v2
    ./setup.sh                              # dependency and asset check
    ./run.sh play --preset performance      # the default stack

Windows via WSL2:

    .\setup.ps1
    .\run.ps1 play --preset performance

Presets:

    performance   bf16, faster, ~6.9 GiB peak, ~821 MiB minimum headroom
    lowmem        weight-only FP8, slower, ~5.7 GiB peak, ~1161 MiB headroom,
                  and it completes longer requests that bf16 cannot

Low memory is not a speed mode. On this card the FP8 weight-only path is **slower**,
not faster: it trades capacity for speed, and the preset name says so.

## What you get

    interactive runtime   event identity, application claims, fail-closed commit,
                          committed/in-flight state separation, prewarm isolation
    preview               ~231 ms non-authoritative visual feedback, zero training
    authoritative         ~790 ms, with a 50 ms blend to ~840 ms full authority
    traces                LatencyTrace (authoritative) and PreviewTrace (speculative)
                          as separate records

## Latency, and what the numbers mean

    input -> assigned          ~1 ms class
    preview visible            ~231 ms
    authoritative ready        ~790 ms
    full authoritative display ~840 ms    DISPLAY SEMANTICS, not compute latency

The preview appears about 231 ms after an input, so the interaction does not sit
silent through a full model chunk. It is deliberately non-authoritative: it can never
commit state or be recorded as the first real frame.

## Known limits

    preview ~231 ms       model-side decoded timing, NOT a physical present
    authoritative ready   ~790 ms
    full authority UI     ~840 ms as display semantics, from the blend
    t4 renderer submit    UNAVAILABLE: no renderer exists in this tree
    t5 presented          UNAVAILABLE: no present signal exists
    input-to-display      therefore NOT MEASURABLE here

There is no t4/t5 instrumentation in this project. Where a number is a proxy it is
labelled a proxy; where a measurement does not exist it is reported as unavailable
rather than filled in with the nearest convenient timestamp.

## Reproducing

    ./run.sh smoke          release smoke test, GPU path included
    python test_interactive_runtime.py       runtime contract
    python test_control_exactly_once.py      exactly-once camera
    python test_stale_frontier.py            expired application claims
    python test_prewarm_and_reference.py     prewarm isolation, reproducibility

## Documentation

    docs/BENCHMARK_CARD.md              final effective numbers, one page
    docs/RELEASE_1_AUTHORITY.md         production vs research-only, and the limits
    docs/DEPLOYMENT_PRESETS_FROZEN.md   the two presets and their measurements
    docs/LATENCY_1A_CONTRACT_AND_AUDIT.md   t0..t5 contract and the reachability audit
    docs/PREVIEW_1C_VARIANT_D_RELEASE.md    why the preview is variant D
    docs/QUANTUM_1A_STEP_ABLATION.md        why fewer DiT steps is rejected
    docs/KV_1A_DECOMPOSITION.md             why the KV update has no easy win

## Scope

Target hardware is one RTX 5060 Laptop with 8 GiB. The model's nominal geometry is
512x768; the deployment geometry here is 304x528, which is what fits and what every
measurement in this repository refers to. Model nominal geometry is not deployment
geometry authority.
