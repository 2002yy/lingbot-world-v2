# G4-hotpath: PASS -- the streamed prepare does not contaminate the hot path

## Method: hotswap_loop's logic, minimally adapted

The state and event logic of `hotswap_loop.py` is reused verbatim: the same 60 Hz
authoritative CameraController, the same compensator, the same scripted control
timeline, the same step-boundary hot-swap, the same KV and cam bookkeeping.

Four things change, each for a stated reason:

1. GEOMETRY 304x528, the deployment geometry from M0-prod-A, not 320x480.
   Selected by pre-resizing the image, since generate-style code derives the
   geometry from the native aspect.
2. DECODER TAE-HV, matching the real-time display path, not the full streaming VAE.
3. CONDITION y goes through `_condition_latent`, so `LINGBOT_STREAM_ENCODE`
   selects the encoder. That variable is read per call, so both arms run in ONE
   process and can be interleaved.
4. ABBA INTERLEAVING. A laptop GPU is subject to thermal drift, power limits,
   boost clocks and allocator warm-up, so AAAA-BBBB would read drift as an effect.

## Result: 8 chunks x 2 ABBA rounds = 8 arms

    arm order: [0,1,1,0,1,0,0,1]

    condition y hash:  whole=a2233b4e4d91c4aa  streamed=a2233b4e4d91c4aa  identical

    metric                whole   streamed      delta      pct
    DiT p50               434.8      434.3       -0.5    -0.1%
    DiT p95               450.5      451.1       +0.6    +0.1%
    decode p50             25.3       25.2       -0.1    -0.3%
    decode p95             25.5       25.6       +0.1    +0.3%
    hotpath p50           460.1      459.6       -0.5    -0.1%
    hotpath p95           475.8      476.6       +0.9    +0.2%

    reserved after prepare   whole 7187   streamed 7272 MiB
    allocated creep          whole +0.0   streamed +0.0 MiB
    event/state              applied_evt present on every chunk

    1  hot-path <= +3%               PASS  (everything within +-0.3%)
    2  no allocator creep            PASS
    3  y identical + events intact   PASS
    OVERALL: PASS

## Default flipped

`wan/image2video.py::_condition_latent` now defaults `LINGBOT_STREAM_ENCODE` to
"1". The case is four independent properties, not the 118 MiB:

- bit-exact: condition y hash and final output hash match exactly, at 81 and 249
  frames;
- no change to model semantics;
- removes a length-dependent OOM cliff: at 501 and 777 frames the whole-clip path
  cannot complete even the FIRST request while the streamed path can;
- no measurable cost on short requests, and G4-hotpath shows no hot-path
  contamination, with prepare marginally faster.

`LINGBOT_STREAM_ENCODE=0` is kept for debugging and for reproducing pre-M1-2
numbers.

## Two problems hit and fixed this round

1. **My own offload patch had a regression.** It wired
   `_restore_device(offload_model)` into `prewarm()` as well, but prewarm has no
   `offload_model` argument, so every prewarm raised NameError. G4's first test
   caught it within 30 seconds -- which is the argument for running one test after
   every change.
2. **The 8/2 worst-case KV preset OOMs in this harness.** With local_window 8 and
   sink 2 the prepare phase OOMs (4.4 GiB resident plus a 2.7 GiB encode), so the
   development default 6/1 is used. Separately, `reserved` creeps about 40 MiB per
   arm: 8 chunks completes, 10 chunks OOMs on the fourth arm. Recorded as a known
   constraint on arm count for this harness at 304x528/bf16.

## Terminology discipline, enforced in the output

This gate reports **measured hot-path latency**, not control-to-real. The G4-0
audit established that no existing script has a qualifying control-to-real. The
`hotpath_ms` figure here may later serve as one component of a control-to-real
lower bound, but it must never be renamed control-to-real.

## Historical figures demoted

The following must now be labelled **estimated / decode-complete proxy, NOT
measured input-to-present latency**:

- "control-to-real p50 ~950 ms / worst ~1.6 s" (a formula in bench_loop_320x480)
- "control-to-real-frame=..." (a configuration value in realtime_display)
- hotswap_loop's `ctl_to_real` (a decode-return measurement)
- "control-to-real p50 ~1.38 s" (derived from the above)

## Next

1. Re-freeze the 8 GB deployment presets, for the first time on a complete
   configuration: 304x528 + real generate + repeated requests + offload fix +
   streamed encode. Re-measure BF16 and FP8_lowmem latency, peak and minimum
   headroom.
2. §Latency-1 Input-to-Present Contract, as its own item: t0 event enqueued, t1
   assigned to a tick, t2 state committed, t3 first affected real frame decoded,
   t4 submitted to the renderer, t5 actually presented, with
   `control-to-real-display = t_present - t_input`.
