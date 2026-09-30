# P0 fix + P2b-prod: Amdahl's law at the production scale (22.8% -> 6.2%)

## P0: offload_model=True never restored the DiT (correctness bug, fixed)

### The defect

    wan/image2video.py:948 and :1212   if offload_model: self.model.cpu()

The next `generate()` goes straight into a forward without moving the model back,
so the second request necessarily fails:

    RuntimeError: Input type (CUDABFloat16Type) and weight type
                  (CPUBFloat16Type) should be the same

That makes repeated real requests -- exactly what P2b-prod must measure -- broken
by construction.

### The fix

`_restore_device(offload_model)` added to `wan/image2video.py`, called at the
**entry** of the generation function next to the existing `bump_cam_epoch()`, so
it runs once per request and never in the per-chunk hot path. It is a no-op when
the model is already on the target device, so the non-offload configuration pays
nothing, and its cost is logged above 1 ms so it cannot silently become a tax.
Wired into both entry points (`_generate_causal_fast`, `_generate_causal_pretrain`).

### Verification

    A) in-tree fix only, no manual ensure_device, 5 requests:
         all ok, out=4a718b0e1f0a3e5f on every request
    B) manual ensure_device (control), 5 requests:
         all ok, out=4a718b0e1f0a3e5f on every request

Identical output hashes, so the fix changes nothing about the result and only
repairs the repeated-request path. The second request no longer mismatches.

## P2b-prod: 304x528, real production chain, bf16 vs fp8_lowmem

Real `generate()` chain, 81 frames, chunk_size 3, 4 timesteps, worst-case KV
preset (local 8, sink 2), 304x528 (lat 38x66, fsl 627), 5 requests per arm with
phase timing. Phase timing synchronises at every boundary so it inflates the
total; the un-instrumented control is given below.

### Phase attribution (warm, instrumented)

                   bf16       fp8        delta
    DiT forward   10.79 s    14.01 s    +3.22 s  (+29.8%)
    VAE encode     4.71 s     4.71 s      0.00 s
    VAE decode     8.13 s     8.08 s     -0.05 s
    unaccounted    6.57 s     3.36 s     (synchronise tax, differs by arm)
    ---------------------------------------------
    full request  27.54 s    29.26 s    +1.72 s  ( +6.2%)
    global peak   7142 MiB   5804 MiB   +1338 MiB
    min free      1521 MiB   1301 MiB

### Un-instrumented control (more trustworthy totals)

    bf16  warm p50 26.54 s   (from the 25-request plateau run)
    fp8   warm p50 28.03 s   (5-request run)
    -> +1.49 s = +5.6%, consistent with the instrumented +6.2%

## The core result: Amdahl's law, at last on the right denominator

    DiT portion:    BF16 is 29.8% faster     <- the real compute gain
    full request:   BF16 is 5.6-6.2% faster  <- what a user actually sees

Because 42% of the request is entirely immune to the weight mode:

    condition encode  4.71 s (15.6%)  independent of the weight mode
    VAE decode        8.08 s (26.8%)  independent of the weight mode

So the earlier harness figure of -22.8% was a DiT-plus-TAE number. The production
figure is -5.6% to -6.2%. This is not the optimisation failing; it is the
optimisation being placed back over the correct denominator.

## The trade, now precisely quantified

    bf16:  +1338 MiB VRAM  <->  -1.5 to -1.7 s per request (5.6-6.2% faster)
    fp8:   -1338 MiB VRAM  <->  +1.5 to -1.7 s per request

i.e. **1.3 GiB of VRAM buys about 5.6-6.2% of request latency.**

## Headroom, at the real scale

    after plateau (25-request run), bf16 min free:   721 MiB
    5-request run (not yet plateaued):              1521 MiB
    the harness figure had been:                    ~2.7 GiB

So the real deployment headroom is in the ~700 MiB class, not the 2.7 GiB class.
721 MiB passes the gate but is not generous.

## Product positioning

    bf16        performance deployment mode
                faster (5.6-6.2%), but only ~700 MiB of headroom, so the memory
                lifecycle must be tightly controlled

    fp8_lowmem  resilience / compatibility mode
                +1.3 GiB of headroom, for preview, other GPU consumers, and
                Windows/WSL variability

So this is not "bf16 default / fp8 fallback" so much as two modes for two
purposes, and the documentation should say so.

## Next: M1-2 moves up

The condition encode is 4.71 s (15.6%) of a real request, and M1 already showed
the streamed block=4 encode is bit-identical, lower-peak and slightly faster. So
wiring it into the real chain is the highest-value next step -- correct in
denominator and already proven safe -- ahead of further GEMM work.

    offload restore (done) -> P2b-prod (done) -> real Amdahl breakdown (done)
      -> M1-2 streamed encode into the real chain
      -> re-measure warm latency / peak VRAM / input-to-display
      -> freeze the 8 GB deployment preset
