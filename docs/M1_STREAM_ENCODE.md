# M1-0 / M1-1: chunked condition encode -- equivalence and memory

## The key discovery: the encoder is already chunked internally

    vae2_1.py:515  def encode(self, x, scale):
                       self.clear_cache()
                       t = x.shape[2]
                       iter_ = 1 + (t - 1) // 4              # 4-frame blocks
                       for i in range(iter_):
                           self._enc_conv_idx = [0]
                           if i == 0:
                               out  = self.encoder(x[:,:,:1], feat_cache=..., ...)
                           else:
                               out_ = self.encoder(x[:,:,1+4*(i-1):1+4*i], ...)
                               out  = torch.cat([out, out_], 2)

So the 777-frame OOM is NOT the encoder's internals. It is the caller
materialising the whole zero-padded input at once:

    torch.concat([img, torch.zeros(3, F-1, h, w)], dim=1)   ~1.4 GiB
    plus the zeros tensor itself                             ~0.7 GiB

The fix is to feed blocks, building each one on the fly: block 0 is the real
image, every later block is freshly allocated zeros, so the full sequence never
exists.

## M1-0 semantic equivalence: EXACT, bit-identical

    leg A: bf16, 249 frames (21 chunks)
      [whole]     15.26 s  peak_alloc 6244  peak_reserved 6880 MiB
      [streamed]  14.26 s  peak_alloc 5804  peak_reserved 6388 MiB  block=4
      shape  [1,16,63,38,66] == [1,16,63,38,66]
      dtype  float32 == float32
      hash   064bd208b2ccfb91 == 064bd208b2ccfb91   bit-identical=True
      max_abs_diff 0.000000e+00   mean_abs_diff 0.000000e+00
      timesteps with any diff: 0/63
      saving +492 MiB, time -1.37 s (-8.8%)
      VERDICT: EXACT

    leg B: fp8_lowmem, 777 frames (65 chunks) -- bf16's whole reference OOMs here
      [whole]     45.75 s  peak_alloc 5909  peak_reserved 6902 MiB
      [streamed]  44.78 s  peak_alloc 4519  peak_reserved 5510 MiB  block=4
      hash   d1722851e7abd2d0 == d1722851e7abd2d0   bit-identical=True
      max_abs_diff 0.000000e+00
      timesteps with any diff: 0/195
      saving +1392 MiB, time -0.97 s (-2.1%)
      VERDICT: EXACT

Because the block boundaries coincide exactly with the library's internal ones,
the result is bit-identical -- and it is also FASTER, since the large
allocation and zeroing churn is gone, and the peak is lower.

## M1-1 memory Pareto: degenerate -- block=4 wins on every axis

    A) bf16, 249 frames, whole vs streamed, block in {4,8,16,32,64}
       block=4            streamed peak_reserved 6388 MiB   VERDICT EXACT
       block=8/16/32/64   CUDA out of memory

    B) bf16, 777 frames, streamed only, block in {4,8,16,32,64}
       block=4            streamed peak_reserved 6508 MiB   45.85 s
       block=8/16/32/64   CUDA out of memory

The encoder's per-call activation scales with the temporal block length. At
block=4 with bf16 resident (~3.5 GiB) the peak is already 6.5 GiB, leaving about
1.6 GiB, and doubling the block exceeds it.

So block=4 is simultaneously the most accurate, the lowest-memory and the
fastest. There is no "bigger block trades memory" axis here; it is degenerate.
That is also expected on first principles: CausalConv3d's padding position moves
with the block boundary, so block != 4 is not merely a different batching, it is
a different computation.

## Against the stated gate

    gate: BF16 + chunked condition encode max_reserved comfortably below ~7.2 GiB

    measured (bf16, 777 frames, streamed, block=4):
      peak_reserved 6508 MiB = 6.36 GiB  <  7.2 GiB   PASS
      margin against the card's 8151 MiB: 1643 MiB

And the encode does not overlap the interactive loop:

    M0 runtime peak        5426 MiB
    M1 encode peak         6508 MiB
    combined peak = max() = 6508 MiB    PASS

## Two older mysteries resolved

1. The "1800 tokens" mystery (the cam injection mismatch of 1800 against 1881 in
   m0). Root cause: the geometry was derived from the NATIVE image aspect instead
   of the 480/832 training aspect.

       native aspect -> lat 40x60 -> 3 * (40/2) * (60/2) = 1800   exact match
       training aspect -> 304x528 / lat 38x66 -> 3 * 19 * 33 = 1881

   production_loop.py's aspect-based lat_h/lat_w formula yields 320x480, whereas
   pb3_run and the y cache use 304x528. These are DIFFERENT resolutions. M1
   standardises on 304x528 to match every existing measurement and the cache.
   **Open item: the production configuration's resolution must be settled -- this
   is a real divergence in the harnesses.**

2. The first version of m1 fed a 4D tensor to `encode`, which wants 5D
   [B,C,T,H,W], giving an IndexError.

## State discipline

`encode_streamed()` calls `vae_model.clear_cache()` on entry and owns state
independent of the caller, so it can be contaminated neither by a previous
whole-clip encode nor by prewarm. Correctness no longer rests on remembering to
bump an epoch, which is the lesson from the _CAM_EPOCH incident.

## Next: M1-2, wire it into the production prepare phase

    session prepare
      -> incremental condition encode (block=4, lazy zero blocks)
      -> release intermediate activations
      -> enter the interactive loop

The encode must not be spread across input ticks: the goal is the startup/prepare
peak, not trading hot-path latency for memory, given control-to-real p50 is
already ~1.38 s.
