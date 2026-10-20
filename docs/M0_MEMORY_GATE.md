# M0: bf16 interactive memory gate -- PASS, with margin to spare

## Why this gate, and not the encoder rework first

Already known: bf16 + 65 chunks + a whole-clip condition encode OOMs at
`pipe.vae.encode()`, which encodes 777 frames in one shot. What was NOT known is
whether the bf16 *runtime* stack fits 8 GB with real margin. If it did not --
say 7.7-7.9 GiB and thrashing -- then making the encoder incremental would not
save bf16 as a default, and that rework would be wasted. So this gate isolates
the two peaks:

    encode transient peak   vs   steady-state resident + transient peak

The condition latent `y` is produced by the VAE and is independent of the DiT
weight mode, so it is precomputed and loaded. That is not a cheat; it is the
separation under test.

## Configuration, kept faithful to production

    LINGBOT_WEIGHT_MODE = bf16  (original weights)
    LINGBOT_MODE        = repro (FA2 path)
    preset              = temporal_stability_8_2  (local_window 8, sink 2)
                          i.e. the WORST-case KV window
    chunk_size          = 3  -> M = 1881
    control stack       = 60 Hz CameraController + compensator + presets
    TAE decode          = every chunk, on the interactive path
    condition encode    = precomputed y (39 MB, the 65-chunk build)
    length              = 65 chunks

## Four memory points, four metrics each

    [1_after_model_load     ] alloc 3503  reserved 3516  max_alloc 3503  max_reserved 3516  free 3499 MiB
    [2_after_tae_load       ] alloc 3546  reserved 3548  max_alloc 3546  max_reserved 3548  free 3467 MiB
    [3_after_condition_ready] alloc 3895  reserved 4000  max_alloc 5323  max_reserved 5352  free 3003 MiB
    [4_kv_window_full       ] alloc 4940  reserved 5406  max_alloc 5251  max_reserved 5406  free 1597 MiB
    [5_after_long_loop      ] alloc 4940  reserved 5426  max_alloc 5251  max_reserved 5426  free 1577 MiB

    per-chunk trend   first quarter  alloc 4940  reserved 5410  free 1593 MiB
                      last  quarter  alloc 4940  reserved 5426  free 1577 MiB

## Per-chunk stability

    chunk  0: dit=1787 kv=282 tae=782 ttfnf=2851ms alloc=4940 res=5326 free=1677
    chunk 10: dit=1019 kv=310 tae= 80 ttfnf=1409ms alloc=4940 res=5426 free=1577
    chunk 30: dit= 985 kv=308 tae= 81 ttfnf=1373ms alloc=4940 res=5426 free=1577
    chunk 50: dit= 994 kv=309 tae= 81 ttfnf=1383ms alloc=4940 res=5426 free=1577
    chunk 64: dit= 985 kv=309 tae= 81 ttfnf=1375ms alloc=4940 res=5426 free=1577

chunk 0 is first-chunk warmup (higher dit and tae, peak not yet settled).

## Criteria

    1  no OOM                     PASS  (ran all 65 chunks)
    2  no monotone growth (<2%)   PASS  (alloc +0.0%, reserved +0.3%)
    3  >= 500 MiB margin          PASS  (2725 MiB)
    OVERALL: PASS

    peak max_reserved   5426 / 8151 MiB = 66.6%
    margin              2725 MiB
    min driver free     1577 MiB
    control-to-real     TTFNF p50 1383 ms, stable at 1373-1409 after chunk 0
    DiT 993 | KV 310 | TAE 81 ms

## Conclusion: the condition encode is the ONLY remaining blocker

The bf16 runtime -- worst-case KV window 8/2, real TAE decode, production control
stack -- settles at a 5426 MiB peak with 2725 MiB of margin and zero growth over
65 chunks.

For contrast, the pb3 bf16 21-chunk peak was 6611 MiB, but that figure INCLUDED
the whole-clip encode. Bypassing the encode drops steady state to 5426 MiB, so
the difference of well over 1 GiB is the encode transient.

Therefore **bf16 defaulting is one peak away: the condition encode.** The
incremental-encoder work now has a clear target and a clear acceptance test.

## Two of my own bugs, caught

1. This harness omitted `mf.bump_cam_epoch()` after `pipe.prewarm()`, so
   prewarm's dummy forward at `current_start=0` poisoned the camera cache for
   chunk 0, appearing as a 1800-vs-1881 token mismatch inside the block. This is
   the exact cam-cache poisoning already documented in model_fast.py -- the
   second time it has been hit. Gate, now mandatory for any new harness: call
   `bump_cam_epoch()` immediately after `prewarm()`.

2. Throughput was printed as `1000/period` while `period` is already in seconds,
   reporting 723 chunk/s and 2892 fps-equiv. Both are impossible numbers that
   should have been stopped on sight. The real value is 1/1.383 = 0.72 chunk/s.
   Same family as the 795% figure in P2b-0c: check units and bounds.
