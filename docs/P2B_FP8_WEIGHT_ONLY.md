# P2b: why 8960 -> 1536 is slow on an RTX 5060 Laptop

## The question

Is it bad kernel selection, an unfriendly N=1536 shape, surrounding overhead, or
is it already near the hardware limit?

## The answer, each part measured

    not kernel selection
    not the N=1536 shape
    not surrounding overhead
    not the hardware limit
    YES: the FP8 weight-only path we enabled to save VRAM costs +83% on this GEMM

## P2b-0: production-faithful attribution (pb0_attrib.py)

Capture from a real forward:

    ffn.2 calls: 360 over 3 chunks = 120.0 per chunk  (30 blocks x 4 forwards)
    distinct call signatures: 1
      shape=(1,1881,8960) bf16 contiguous=True
      stride=(16853760, 8960, 1)     fully contiguous, standard
      autocast=True dtype=torch.bfloat16
      weight_dtype=torch.bfloat16 bias=True

So there is no shape mix and no layout/stride/transpose problem to fix.

    isolated GEMM x120           = 330.5 ms/chunk
    in-model attribution         = 333.19 ms/chunk
    unexplained by the GEMM body =   2.6 ms/chunk (1%)

The 333.19 ms IS the matrix multiply. Surrounding ops are 1%.

Hardware ceiling, square bf16 GEMMs:

    1024^3 26.5 | 2048^3 36.0 | 4096^3 37.0 | 8192^3 37.8 TFLOP/s
    4096^3 fp32-cast                              11.2 TFLOP/s

    ffn.2 at M=1881: 18.8 TFLOP/s = 50% of ceiling

## P2b-0b: pricing the FP8 path (pb0b_fp8.py)

A/B on the captured production tensor:

    shipping FP8 weight-only module   2.9391 ms   16.9 TFLOP/s
    dequantised bf16 Linear           1.6069 ms   30.8 TFLOP/s
    F.linear bf16 (same weight)       1.6056 ms   30.9 TFLOP/s

    FP8 overhead +1.3322 ms/call (+82.9%)  ->  +159.9 ms/chunk

Shape ceiling, same K/N with varying M:

    M=1881   1.566 ms  33.1 TFLOP/s
    M=3762   2.812 ms  36.8 TFLOP/s
    M=7524   5.642 ms  36.7 TFLOP/s
    M=15048 11.243 ms  36.8 TFLOP/s

bf16 reaches 33.1/36.8 = 90% of this shape's ceiling, so the kernel is fine. The
FP8 dequant-on-the-fly is what eats nearly half the throughput.

## P2b-0c: the full reclamation map (pb0c_map.py)

    family     M      K      N   fp8 ms  bf16 ms   delta  x/chunk save/chunk  +MB bit-eq
    ffn.0   1800   1536   8960   2.8721   1.4933  1.3788    120      165.5   13.1   True
    ffn.2   1800   8960   1536   2.7450   1.6036  1.1414    120      137.0   13.1   True
    ca.o    1800   1536   1536   0.3909   0.2610  0.1298    120       15.6    2.2   True
    ca.q    1800   1536   1536   0.3870   0.2589  0.1280    120       15.4    2.2   True
    ca.k     512   1536   1536   0.1998   0.0758  0.1240    120       14.9    2.2   True
    ca.v     512   1536   1536   0.1952   0.0759  0.1193    120       14.3    2.2   True
    sa.q    1800   1536   1536   0.3677   0.2590  0.1087    120       13.0    2.2   True
    sa.k    1800   1536   1536   0.3664   0.2587  0.1076    120       12.9    2.2   True
    sa.o    1800   1536   1536   0.3653   0.2592  0.1061    120       12.7    2.2   True
    sa.v    1800   1536   1536   0.3596   0.2602  0.0994    120       11.9    2.2   True
    -------------------------------------------------------------------------------
    TOTAL per chunk (all 30 blocks): 413.2 ms (26.5%)
    VRAM cost:                       1327.5 MB (1.30 GiB)

An intermediate version of this script multiplied already-whole-model numbers by
30 again and reported 12395 ms/chunk, i.e. 795% of a 1560 ms chunk. A 795% answer
should be rejected on sight. Fixed, and a hard assertion that the total must be
< 1560 ms is now in place.

## P2b-2: end-to-end verdict (pb2_fp8_ab.py, 21 chunks)

    LINGBOT_FP8=1  median chunk 1564.5 ms   VRAM peak 5243.2 MiB
    LINGBOT_FP8=0  median chunk 1208.4 ms   VRAM peak 6568.1 MiB

    -356.1 ms/chunk = -22.8%, VRAM +1325 MiB
    microbenchmark predicted 413 ms; end to end 356 ms (right order, slight
    overestimate, as expected)
    gate (>= 40 ms/chunk): 356/40 = 8.9x  PASS

## Correcting the bit-exact claim

The `bit-equal=True` in P2b-0b/0c holds because both legs use the SAME
dequantised weight -- that is a tautology, not evidence that switching storage is
bit-exact.

The real FP8=0 arm loads the ORIGINAL bf16 weights, which differ from
quantize -> dequantize:

    hash[0] b833348ca63b1bb0 (fp8)  vs  fb960196eff8805b (bf16)

So this is not a repro-safe equivalent swap; it is a different weight precision.
FP8 is a lossy quantisation of the original bf16 weights, so bf16 is strictly
more accurate. That means no quality regression, but it does change the world
trajectory, so it is not bit-exact against the current baseline.

## Conclusion

The FP8 weight-only trade is priced for the first time:

    1.30 GiB of VRAM   <->   356 ms/chunk (22.8%) of latency + weight precision

On this 8 GB card:

    FP8=1 peak 5243 MiB
    FP8=0 peak 6568 MiB   (8151 MiB total, ~1.6 GiB still free)

Recommendation: demote FP8 from "default VRAM optimisation" to "fallback tier
used only when memory is tight". The default should be bf16 -- 22.8% faster and
more accurate -- with FP8 reserved for constrained configurations. The switch
does change the latent trajectory, so a rollout quality gate is still required,
but since it is a precision improvement rather than a loss the expectation is
equal or better. Per the discipline, that must be measured, not assumed.

Against the P2b-2 candidate list:

  1. current bf16 kernel        already at 90% of the shape ceiling; no room
  2. rowwise FP8                already closed by P2a on correctness
  3. weight-only FP8/INT8       this item: default to bf16, reclaim 356 ms
  4. layout change              excluded (single clean signature)
  5. activation->down fusion    untested; surrounding ops are 1%, so low ceiling
  6. epilogue fusion            untested; same reason
  7. narrow-N Triton/CUTLASS    not needed: the shape already reaches 90%

## Methodology lesson

When a conclusion produces a number that is physically impossible, check the
units and the bound before going any further. Here 12395 ms/chunk against a 1560
ms chunk is 795%, which should have been rejected the moment it printed. Same
family as the earlier rules: mirror the operand order, require execution
evidence, mirror the tensor device, check a microbenchmark against an end-to-end
lower bound, and now check the bound.
