# P2b-3 quality gate: PASS, with one memory constraint

## Gate definition

The definition was deliberately changed from the P2a/S3 shape, because this is
not a normal optimisation: it is the *removal of an expensive lossy VRAM
optimisation*.

- 21ch paired: judged as "does BF16 show degradation", NOT "how close is BF16 to
  FP8". BF16 is the original weights and FP8 is the lossy approximation, so
  closeness would be the wrong target.
- 65ch: judged as BF16's own intrinsic health, NOT "is chunk 64 the same tree as
  FP8's chunk 64". S3-C already established that a causal world model diverges
  into two different worlds from any numerical difference, and that is not
  quality loss.
- Plus a BF16-vs-BF16 determinism control.

## 1. Determinism: PASS, and clean

    DETERMINISM: b_21 vs b2_21  (same seed, same inputs)
      latent hashes identical : True  (0/21 differ)
      decoded frames          : max|diff| = 0.000e+00 over 21 chunks

This establishes a new reference mode: **original bf16 weights + FA2 repro path**
is the true reference. Every future FP8 / Sage / compile change should be judged
against this, rather than continuing to treat weight-only FP8 as ground truth.
That reorganises the whole optimisation tree.

## 2. 21ch paired: diverges as expected, and that is not a FAIL

    segment        LPIPS     SSIM     PSNR  edgeSSIM
    0-3           0.0552   0.8336    23.91    0.7981
    7-12          0.2756   0.4858    13.41    0.4221
    13-20         0.3822   0.3978    11.58    0.3412
    20-32         0.3786   0.3878    11.82    0.3365
    ALL           0.2551   0.5394    15.02    0.4861

Exactly the predicted S3-C shape: near-identical at the start, then two worlds.
Under the corrected gate definition this does not count against BF16.

## 3. 65ch intrinsic health: PASS, and comparable to or better than FP8

    metric                  BF16 65ch    FP8 65ch (comparison)
    latency median           1209.0 ms    1565.7 ms
    latency first10          1261.7       1603.3
    latency last10           1213.3       1565.6
    latency drift            -3.8%        -2.4%     no growth with length
    VRAM drift               +0.5%        +0.7%     stable
    finite/health            OK           OK        no NaN or explosion
    brightness drift         -14.9%       -13.4%
    contrast drift           -11.3%       -6.7%
    edge energy drift        +1.5%        +3.0%     texture retained
    self-SSIM floor          0.1313       0.1256    not collapsing
    consecutive SSIM         0.27-0.47    0.26-0.47 temporal stability comparable

BF16 is healthy on every axis and equal or better on most. The 65ch paired
numbers (LPIPS 0.3117, SSIM 0.4606) are likewise just two different worlds, not
degeneration.

## 4. A real memory constraint (this is the user's point 4, confirmed)

    BF16 + 65 chunks + full-horizon condition encode -> CUDA out of memory
      at pipe.vae.encode()
      frames_n = (65*3-1)*4+1 = 777 frames encoded in one shot
      FP8 fits (1.3 GiB less), BF16 does not

This is not a question of whether the arithmetic says it fits; it was measured not
to fit. It is precisely what the vLLM-Omni design addresses with its
session-owned causal Wan VAE encoder and per-block condition encoding.

Clean workaround used here: `y` (the condition latent) is produced by the VAE and
is INDEPENDENT of the DiT weight mode, so it is computed once (39 MB), cached, and
loaded by the BF16 arm. Both arms therefore share an identical `y`.

21ch peaks, both including the encode:

    fp8   5286 MiB
    bf16  6611 MiB     +1325 MiB, matching the predicted 1.30 GiB

At 65 chunks with the encode, bf16 would need roughly 6276 + 1325 = 7600 MiB of
8151 MiB, i.e. 93% -- and it did OOM. **So the precondition for making BF16 the
default is to make the condition encode incremental or windowed.** Otherwise any
long-horizon production configuration will OOM.

## Verdict

    1. BF16 vs BF16 determinism   PASS  (bit-identical, 0.000e+00)
    2. 21ch paired                PASS  under the corrected definition
    3. 65ch intrinsic health      PASS  (no NaN, explosion, collapse or growth)
    4. 8 GB production memory     NOT YET -- needs its own stress gate, and the
                                  full-horizon encode is already known to OOM at
                                  65 chunks

## Naming

`LINGBOT_WEIGHT_MODE` added:

    bf16          original weights, fastest, ~6.6 GiB
    fp8_lowmem    weight-only FP8, ~5.3 GiB

Deliberately not called "fp8 fast": the low-memory tier is *slower*. FP8 here
trades capacity for speed, not the other way round. `LINGBOT_FP8` remains a
legacy alias so every existing script keeps working. The default is still
`fp8_lowmem` -- the quality gate has passed but the memory gate has not, and the
flip is deliberately the last step.

## Next

    Full interactive 8 GB memory stress gate
      bf16 weights + production interactive loop + preview/display + VAE
      + worst steady-state window
      criteria: no OOM, no allocator thrash, sensible peak margin,
                control-to-real not regressed
      known precondition: condition encoding must become incremental/windowed
                          (cf. DreamZero 603 MiB per session)
        -> PASS means BF16 becomes the production default and FP8 the fallback
