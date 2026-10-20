# G4-0 measurement-contract audit: no existing control-to-real is a real measurement

## The contract the gate requires

    control-to-real = input event formally accepted
                      -> the first real frame that actually carries that event's
                         effect and is committed to the display chain

Explicitly excluded: chunk start time, decode completion time, the CPU receiving a
tensor, a preview frame, or some harness-internal callback.

## Audit result: three scripts, zero qualifying measurements

### bench_loop_320x480.py (formerly production_loop.py)

    :250  control-to-real  p50 ~{tt} ms  worst ~{tt + CADENCE*1000/2} ms

`tt` is the mean TTFNF and CADENCE is 1.25. This is a **formula**, not a
measurement, and the "worst" figure is arithmetic that adds half a cadence to the
median.

### realtime_display.py

    :127  control-to-real-frame={args.real_cadence*1000:.0f}ms

This prints a **command-line configuration value**. Nothing is measured at all.

### hotswap_loop.py -- the only one with real timestamps, still not qualifying

    :252  t_vis = time.perf_counter() - t_start     # t_start = chunk start,
                                                    # taken after
                                                    # vae.stream_decode_step(x0)
    :292  ctl_to_real = vis - t_in

Three problems:

1. `t_vis` is the moment the **decode step returns** -- precisely the
   "decode completion time" the contract excludes -- not a frame committed to a
   display chain.
2. `ctl_to_real = (elapsed time vis) - (script timestamp t_in)` mixes two
   different time bases. It only approximates the intended quantity if the loop's
   t=0 happens to align with the script's t=0. That is a fragile convention, not a
   contract.
3. It decodes with `pipe.vae.stream_decode_step` (the full streaming VAE), not
   TAE-HV.

## Geometry: no interactive script runs at the deployment geometry

    hotswap_loop.py        --area 512x320, aspect-based  -> 320x480
    interactive_loop.py    max_area = 512*320, aspect-based -> 320x480
    bench_loop_320x480.py  --area 512x320, aspect-based  -> 320x480
    realtime_display.py    --area 528x288                (a fourth value)

So **no interactive script runs at 304x528**, which M0-prod-A just established as
the 8 GB Deployment Geometry Authority candidate.

## Ruling

- No existing control-to-real number can underwrite GATE 4.
- Nor can we claim an existing interactive runtime *is* the deployment path: they
  are neither at the deployment geometry nor using a qualifying timing contract.
- This is exactly the mislabelling risk that was flagged: a harness named
  "interactive" or "production" standing in for a contract it does not satisfy.

## A tractable restatement of GATE 4

The streamed encode is by construction outside the hot path, so the real question
is narrower:

> does the streamed prepare leave allocator, cache or state in a condition that
> indirectly slows the interactive hot path that follows?

That can be answered directly, without first settling the display-chain contract:

    same interactive runtime, A LINGBOT_STREAM_ENCODE=0 vs B =1
      -> per-chunk DiT and decode latency distributions
      -> reserved / allocated after prepare
      -> any allocator creep or new stale events

To avoid a second mislabelling, this run must:

1. state plainly that it measures **per-chunk hot-path latency**, not
   control-to-real, which needs a display-chain contract this round does not have;
2. run at 304x528, the deployment geometry, not 320x480;
3. interleave ABBA so thermal drift and allocator warm-up cannot masquerade as an
   effect;
4. if per-chunk latency is converted into a control-to-real figure, label it a
   lower bound or approximation with its assumptions stated, never plain
   control-to-real.

Proposal: rename this stage **G4-hotpath** (hot-path contamination check), and
track the control-to-real contract as its own item.
