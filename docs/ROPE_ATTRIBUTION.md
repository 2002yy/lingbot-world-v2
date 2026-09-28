# RoPE attribution and the RoPE-1 gate: FAIL (5.4 ms/chunk against a 30 ms gate)

## The ask

RoPE is 97.1 ms/chunk over 240 calls (6.3%), where 240 = 30 layers x 4 forwards
per chunk x 2 (q and k). The task was a strictly bounded slice: attribute that
97.1 ms, and only implement if at least 30 ms/chunk of it is eliminable
materialisation, layout or launch cost. 30 ms is about 1.9% of a 1566 ms chunk.

## The first attribution was wrong: the microbenchmark put grid_sizes on CUDA

p2d_rope0.py and p2d_rope0b.py built

    grid_gpu = torch.tensor([[F, H, W]], device="cuda")

and measured a 500.2 us full call, concluding that `grid_sizes.tolist()` costs
34 us alone but 236 us inside the call, so that removing it would save
63.2 ms/chunk (-52.7%) -- bit-exact, and comfortably past the gate.

The production code is

    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])   # line 760

and `torch.tensor(tuple)` makes a **CPU** tensor. So `grid_sizes` is on the CPU
and `.tolist()` never synchronises. Corrected measurements (p2d_rope0c.py):

    CPU grid_sizes.tolist()    0.15 us   <- production
    GPU grid_sizes.tolist()   46.37 us   <- what the wrong version used
    full call, GPU grid 482.0 us vs CPU grid 416.5 us

The 65.5 us gap is the fabricated sync. The entire -63.2 ms/chunk prediction was
an artefact of that device error.

## End-to-end result (21 chunks, reps=6, single process, A/B)

    off      median 1568.6 ms  (+0.00%)  bit-identical=True  (0/21 differ)
    off2     median 1567.9 ms  (-0.05%)  bit-identical=True  (0/21 differ)
    rope     median 1567.0 ms  (-0.11%)  bit-identical=True  (0/21 differ)

Bit-exactness holds as predicted. The end-to-end gain is -0.11%
(-1.6 ms/chunk) against a -0.05% noise floor, not the -52.7% the microbenchmark
promised.

## Corrected attribution

    ref             416.5 us/call  -> 100.0 ms/chunk
    cached freqs_i  394.1 us/call  ->  94.6 ms/chunk   -22.4 us (-5.4%)
                                   => 5.4 ms/chunk exactly eliminable

A second measurement bug sits here: `_table()` was benchmarked as if it were a
pure builder, but it caches internally, so the "build (miss)" number was really
the hit path (0.11 us). The real build cost is about 22 us.

## Gate verdict

    5.38 ms/chunk  vs  30 ms/chunk  ->  FAIL

Per the stop-loss rule, RoPE is closed now, without writing a custom kernel, and
P2b becomes the next line.

## Hypotheses this refutes

- "grid_sizes.tolist() is a hard sync and the main RoPE cost" -- it is a CPU
  tensor; tolist() is 0.15 us and overlaps entirely with the GPU.
- "the per-layer freqs_i rebuild is the waste behind the 240 calls" -- worth
  about 22 us/call, 5.4 ms/chunk, and only ~1.6 ms of it shows end to end.
- "the fp64 intermediate is free money to remove" -- fp32 is indeed 3x faster
  (-81.2 ms/chunk) but differs by 4.768e-07, so it is not bit-exact and carries
  the same risk class as P2a. Not done.

What survives: the fp64 round trips are real (x.to(float64) moves 23.1 MB per
call) but they are required to stay bit-exact with the upstream reference, so
they are a cost, not a waste.

## Code

`wan/modules/model_fast.py` gained `_rope_grid_list` and `_rope_freqs_i`, both
bit-exact (0.000e+00) and valid: `LINGBOT_ROPE_CACHE` defaults to 0 so that the
default behaviour is unchanged, and `set_rope_cache()` allows single-process A/B.
Below the gate, kept as a documented default-off option; no further investment.

## Methodology lesson

**A microbenchmark must mirror the production tensor's DEVICE, not just its
shape and dtype.**

The chain: the shape, dtype and operation order were copied faithfully, but
`grid_sizes` was placed on CUDA when production keeps it on the CPU. That
invented a 46 us sync, produced a clean and bit-exact -52.7% result, and that
result *passed the gate* -- nearly buying a 1-2 hour implementation of a change
worth about 1.6 ms. Only the end-to-end number, 480x away from the prediction,
exposed it.

This is the third member of one family:

- a proxy expression must mirror the **operand order** (the r3 cam misjudgement)
- a diagnostic probe must have **execution evidence** (the Event 2004 incident)
- a proxy implementation must mirror the **tensor device**, or it fabricates sync
  costs that do not exist

New standard action: **a microbenchmark's prediction must be checked against a
cheap end-to-end lower bound BEFORE implementing.** Here one 21-chunk
off-vs-rope run would have shown within an hour that -63 ms was impossible.
Microbenchmarks rank candidates; they do not declare wins.

## Next: P2b (FFN-down, 8960 -> 1536)

Theoretical pool 333.19 ms/chunk (21.2%), far larger than RoPE. Already known:
rowwise FP8 on this narrow output is 0.73x / 0.69x / 0.60x, so it is not a
dtype swap. Investigate the actual kernel and tile, utilisation at M = 627 /
1881 / 3762, whether the N = 1536 tile tail wastes the GEMM, layout and
transpose effects on kernel selection, fusing activation into the down
projection, eating bias/residual/scale into the epilogue, and whether a narrow-N
Triton or CUTLASS kernel is justified. Attribution first, as here; a
microbenchmark ranks, the end-to-end run decides.
