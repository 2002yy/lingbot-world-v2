# Release-1: what is production and what is not

## Production

These are the only supported paths. `run.sh` and `play.py` default to them.

    preset performance
      LINGBOT_WEIGHT_MODE=bf16
      geometry 304x528
      FA2 repro attention path
      streamed condition encode (LINGBOT_STREAM_ENCODE=1)
      authoritative runtime: event identity, application claims, fail-closed commit
      variant D preview (no training, no new weights)
      50 ms handoff blend

    preset lowmem
      LINGBOT_WEIGHT_MODE=fp8_lowmem
      everything else identical

    entrypoints
      ./run.sh play --preset performance
      ./run.sh play --preset lowmem
      ./run.sh check          dependency and asset check
      ./run.sh smoke          release smoke test, including the GPU path

## Research-only, and deliberately NOT reachable from the presets

    fixed 2-step / 1-step DiT          Quantum-1A: FAIL, the KV accumulates divergence
    learned preview head               Preview-Head-1B: does not generalise
    async / separate-stream preview    Preview-1A: does not overlap, and slows the DiT
    latent-downsample preview          Preview-1B: quality collapses
    Hybrid / Sage long-horizon mode    S3-C: diverges, short-horizon similarity is not
                                       long-horizon safety
    torch.compile                      S3: not long-horizon trajectory preserving
    paged KV, session runtime          applicable to multi-session serving, not to a
                                       single preallocated rolling window

None of these are exposed as a flag that a user could mistake for a recommended mode.
They live in the research scripts and their conclusions are recorded in the findings
document.

## Why the boundaries are what they are

Every one of those closures came from a measurement rather than a preference, and the
negative ones were the expensive part of the work:

- dropping DiT steps saves 175 ms and poisons the recurrent state, monotonically and
  without plateauing over 10 chunks;
- the KV/state update is 0.945x a denoise step, i.e. one more full forward, with the
  rolling memmove at 1.8% of it -- so there is no cache overhead to engineer away;
- the learned preview head fails both the capacity gate and cross-scene generalisation;
- the training-free variant D degrades gracefully across scenes with no gross failures.

The result is that the shipped preview path costs no training and no new weights, and
the shipped authoritative path is the one whose numerical behaviour has been verified.

## Hardware expectations

    RTX 5060 Laptop, 8 GiB, sm_120, driver 591.86, WSL2 Ubuntu

    preset performance (bf16)
      peak reserved   ~7104 MiB measured on the Deploy-1 freeze
      min free        ~821 MiB over 25 consecutive requests
      warm request    ~29.3 s p50 for an 81-frame request

    preset lowmem (fp8_lowmem)
      peak reserved   ~5846 MiB
      min free        ~1161 MiB
      warm request    ~30.8 s p50
      and it completes repeated requests at 501 and 777 frames where bf16 does not

    the interactive loop, 304x528, bf16, preview on
      chunk generation p50 ~822 ms
      preview decode p50  ~27 ms

Below roughly 7 GiB the lowmem preset is the safer choice; `setup.sh` warns.

## Known limits, stated plainly

    preview ~231 ms         model-side decoded timing, NOT a physical present
    authoritative ready     ~790 ms
    100% authoritative UI   ~840 ms as DISPLAY SEMANTICS, from the 50 ms blend
    t4 renderer submit      UNAVAILABLE: there is no renderer in this tree
    t5 presented            UNAVAILABLE: there is no present signal
    input-to-display        therefore NOT MEASURABLE here

There is no t4/t5 instrumentation anywhere in this project. The blend is executed
against a frame buffer and its semantics are measured, not its present time. This is
stated rather than papered over because the alternative -- quoting a proxy as a
measurement -- is exactly what earlier stages spent effort undoing.
