# 8 GB deployment presets, FROZEN

The first complete baseline: geometry authority settled, offload restore fixed,
streamed condition encode on by default, and repeated real requests stable.

## Configuration

    geometry     304x528 (8 GB Deployment Geometry Authority, verified in M0-prod-A)
    chain        real wan.WanI2VCausal.generate(), 81 frames, chunk_size 3,
                 4 timesteps, preset local_window 6 / sink 1
    prepare      streamed condition encode (M1-2, default now 1)
    offload      offload_model=True + the in-tree device restore fix
    requests     25 consecutive

## The table

    metric                                   BF16 performance       FP8 lowmem
    ---------------------------------------------------------------------------
    cold first request (s)                             119.86           131.10
    warm request p50 (s)                                29.25            30.84
    warm request p95 (s)                                31.49            31.29
    warm min / max (s)                          27.60 / 46.29    29.56 / 33.32
    output FPS                                          2.769            2.626
    peak max_reserved (MiB)                              7104             5846
    min free VRAM (MiB)                                   821             1161
    steady reserved plateau (MiB)                        5999             5720
    alloc leak over 25 req (MiB)                           +0               +0
    reserved last-vs-prev window (MiB)                     +0              +63
    requests ok                                         25/25            25/25

    condition y hash   7741be09201d5d6d  (identical across both -- the VAE is not
                        quantised, so this is expected)
    stream_encode      1 for both

## Differential

    warm p50     BF16 is 1.59 s faster   = -5.4%
    peak VRAM    BF16 uses 1258 MiB more
    min free     BF16 has 340 MiB less (821 vs 1161)

## Long-horizon viability -- the concrete resilience difference

                   F=249        F=501        F=777
    bf16           2/2 ok       1/2 ok       1/2 ok
    fp8_lowmem     2/2 ok       2/2 ok       2/2 ok

At 501 and 777 frames fp8_lowmem completes two consecutive requests while bf16
completes only one, the second failing because the generation itself exceeds
8 GB. This is measured evidence for the "resilience" role, not an assumption.

## Freeze decision

    all 25 requests ok in both     PASS
    no alloc leak in both          PASS  (allocated flat across all 25)
    reserved plateaued in both     PASS  (last-vs-prev window +0 / +63 MiB)

    presets FROZEN:
      performance : bf16       + streamed condition encode
      lowmem      : fp8_lowmem + streamed condition encode
    geometry      : 304x528
    offload       : offload_model=True + in-tree device restore

## Final positioning, for the README and preset docs

    preset=performance   bf16
      original weights, 5.4% faster request latency, higher memory (peak 7104 MiB,
      821 MiB minimum headroom)
      for a free GPU and the lowest possible request latency
      caveat: at 501 frames and beyond, the second consecutive request OOMs

    preset=lowmem        fp8_lowmem
      weight-only FP8, slower, about 1.26 GiB less memory (peak 5846 MiB, 1161 MiB
      minimum headroom)
      for tight memory, preview, other GPU workloads, WSL variability
      advantage: still completes repeated requests at 501 and 777 frames

## Two of my own bugs, fixed

1. The runner recorded `os.environ.get("LINGBOT_STREAM_ENCODE", "0")`, but the code
   default is now "1", so an unset variable was recorded as "0" while the run
   actually took the streamed path. Now records the effective mode. (The y hash
   7741be09201d5d6d matches the M1-2 streamed run exactly, which is the proof.)
2. The freeze predicate used `abs(x or 999) < 150`, and a creep of exactly 0 is
   falsy, so `0 or 999` became 999 and the BEST possible result was marked FAIL.
   Now an explicit None check.

## Three denominators, still frozen

    micro / kernel benchmark
            |
    DiT + TAE benchmark profile
            |
    full deployment request        <- this table

## Next: Latency-1, Input-to-Present Contract

    t0 event enqueued -> t1 assigned to a tick -> t2 state committed
    -> t3 first affected real frame decoded -> t4 submitted to the renderer
    -> t5 actually presented
    control-to-real-display = t_present - t_input

This only becomes meaningful now: it should be measured against a FROZEN
deployment preset, rather than changing memory or weight mode while trying to
establish a latency authority.
