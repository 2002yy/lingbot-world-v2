# Latency-3B-B0: KV-neutral replay proof

    STATUS   PASS, on the real GPU, through the production cache path.
    CLAIM    upgraded from "analysis" to "GPU-validated fact under the tested
             production cache path".

## The claim under test

> Cancelling a chunk and restarting the SAME `chunk_index` needs no KV rollback and no
> snapshot, because every KV write is a pure function of `current_start`: after the
> cancelled attempt's first forward `global_end_index == current_end`, so on the restart
> the eviction guard `current_end > global_end_index` is false, the roll does not fire
> twice, and the else branch recomputes the same slot.

The whole §Latency-3B architecture rests on this one sentence. If it were false, the stage
would need a rollback buffer, and on an 8 GB card that is the difference between feasible
and not. So it was proven before any of that stage was written.

## Result

Tested chunk `N = 8` against `local_window = 6`, `fsl = 627`, `kv_size = 3762`, bf16,
30 layers, 304x528. `N = 8` is in the saturated, evicting steady state: the ring is full
and every chunk rolls.

    cut after forward 1 .. 4, each with a CUDA synchronise and a same-index replay:

      indices (global_end, local_end)     identical to the reference
      evictions during the replay         exactly 1, never 2
      the chunk's slot (627 tokens)       bit-identical, |diff| = 0
      the WHOLE cache (661 MiB, digest)   identical
      regions differing anywhere          0

    per-forward trace, replay vs reference:
      forward 1, 2, 3 and 4               bit-identical at every step

So the roll fires once, as it would have anyway; the restart overwrites the same slot in
place; and the eviction-shifted body is left exactly as an uninterrupted run leaves it.
**No rollback buffer, no per-chunk snapshot, no steady-state pre-image.**

## Gates

    K1  final cache indices/ranges identical for every cut point            PASS
    K2a the current chunk's slot is bit-identical at every cut point         PASS
    K2b the whole cache is identical (order-sensitive int64 digest)          PASS
    K2c any difference is confined to the current chunk slot                 PASS (0 diffs)
    K3  no second eviction on replay, at any cut point, in steady state      PASS
    K4  no extra persistent VRAM allocated by the replay path                PASS
    K5  the chunk's slot is reused, not duplicated                           PASS

Supporting controls, without which the gates would mean less:

    P1/P2  the premise: the ring really is saturated and N really wants to evict
    P3/P4/P5  the reference run does exactly one eviction and advances by exactly fsl
    D1  an UNCUT re-run of the same chunk is bit-identical to the reference
    C1  a run advanced one chunk FURTHER differs, so the detector has resolution
    E   a per-forward trace, so a divergence would be located rather than guessed at

## The two harness bugs that the controls caught, and why they matter

Both produced confident, plausible, WRONG results before being found. They are recorded
because the method is part of the finding.

**Bug 1 -- a fake pass from an empty comparison.** The chunk's slot was sliced with
`chunk_index * fsl` in ABSOLUTE coordinates. The cache is indexed `0..kv_size-1` in LOCAL
coordinates, so the slice was empty and `max|diff|` was trivially `0`. K2a reported PASS
on a comparison of nothing. It was caught only because the whole-cache digest disagreed
with it, and because the localization reduced over the wrong axis and happened to span the
whole cache -- a coincidence, not a design.

**Bug 2 -- a fake failure from the wrong RNG state.** The replay was given a generator
state read at the moment of the replay, which is AFTER the reference path had already
consumed a chunk's draws. The replay therefore ran on different noise and every slot
differed by 12-18 int16 ULP. It looked like a small numerical effect of cancellation. It
was nothing of the kind. **The control that exposed it was D1**: an uncut re-run at the
same state was bit-identical, which proved the kernel was deterministic and therefore that
the difference had to come from the harness.

Without D1 the second bug would have been written up as a real preemption hazard. Without
C1 the first bug's detector would have had no demonstrated resolution. Both controls cost
seconds and both changed the conclusion.

## What this does NOT prove

- **One geometry.** 304x528, `local_window 6`, `fsl 627`, bf16. The argument is about index
  arithmetic and is geometry-independent, but the measurement is not.
- **One chunk position.** `N = 8`. Chunks 0..5 do not roll; the proof covers the rolling
  regime deliberately, since that is where the risk was.
- **One cut per run.** The four cut points are each a fresh replay from the snapshot; a
  single chunk interrupted repeatedly within one generation is not covered.
- **Nothing about cancellation itself.** No cancellation was implemented, no runtime state
  machine was touched, and `_inflight`, claims and commit were not involved. This proves
  the KV layer is recoverable, not that preemption is correct end to end.
- **Nothing about C1/C2/C3.** `_prev_pose`, the RNG-derived noise contract and the `t1`
  clobbering are untouched and remain open, exactly as §Latency-3B-A left them.

## Consequence for the roadmap

    §Latency-3B-A   CLOSED   audit, contract, gate passes with conditions
    §Latency-3B-B0  CLOSED   this document: KV-neutral replay proven on GPU
    §Latency-3B-B   NEXT     conditions C1, C3, C2 in that order, then minimal preemption

The design space has narrowed in the way that matters for an 8 GB card:

    no KV rollback buffer
    no 110 MiB per-chunk snapshot
    no ~578 MiB steady-state pre-image

Reproduce with `python kv_replay_proof.py`. It is standalone: it imports the demo's session
only to build the model, and it edits no production file.
