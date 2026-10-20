# Latency-3B-C0: real-input shakedown

    STATUS   behavioural gates PASS. NO performance conclusion is drawn, and A/B is
             NOT justified by this result.
    PURPOSE  answer one question only: with sustained real input through a real window,
             does single-rebase preemption actually fire, and does it stay correct?

## Gates

Real x11 window, bf16 performance preset, 304x528, preemption enabled, sustained input at
one discrete intent every 200 ms for 25 s.

    preemption actually fired                          PASS  31 preemptions
    several chunks really committed                    PASS  32 chunks, 124 inputs
    no chunk restarted more than once                  PASS  per-chunk max 1
    no event committed twice                           PASS  duplicates []
    no stale generation leaked into a committed chunk  PASS
    a preview never advanced committed state            PASS  0 violations
    every committed frame's t4 was actually recorded    PASS  31/31
    processed/settled explainable                       PASS  gap 0
    frames kept coming, no starvation                   PASS  31 frames shown
    the post-rebase attempt committed                   PASS  32 chunks from 31 preemptions

## What the recorded numbers say, and why A/B is not next

    preemption rate          31/32 chunks      pathological: nearly every chunk restarted
    cut point distribution   {1: 27, 2: 4}     almost all at the first boundary
    observed -> admitted     p50 84.75 ms      the admission cost, dominated by the sync
    rebase call cost         p50  0.02 ms      negligible; the sync is not

    input -> first real      p50 1010 ms       versus 994-999 ms without preemption
      of which: wait         p50  205 ms       halved, from 380-414 ms    <- the gain
      of which: own chunk    p50  740 ms       longer, from 591-595 ms    <- the cost

**The wait halves and the chunk grows by about the same amount. The net is a wash.**

The mechanism is clear and not mysterious: the `observed -> admitted` cost of 84.75 ms --
mostly the one synchronise, which by construction happens *inside* the chunk -- together
with the discarded attempt A work, consumes most of the wait that preemption removes. At
5 intents per second the two nearly cancel.

This is the case 3B-A anticipated, arriving from the opposite direction: not "preemption
never fires", but "preemption fires every time and pays for it". **The correct next slice
is therefore about the admission cost and the trigger rate, not an A/B.** Measuring A
against B now would produce two numbers that differ by noise and invite reading the wrong
thing into them.

## Two real defects C0 found, both fixed

**The t4 generation guard was wrong under preemption.** `mark_renderer_submit` required
`meta.generation_id == committed.generation_id`, which is right for `mark_real_decoded`
(called synchronously) but wrong for t4, whose timestamp arrives from the display thread
and is drained later -- by which time a preemption has moved the generation on. First C0
run: **32 refusals out of 32**, i.e. every t4 but the newest frame was silently dropped.
The correct condition is that the frame's chunk really committed, under the generation the
frame claims, which checks the historical record rather than the current cursor and is what
still makes a discarded attempt's frame un-markable.

**A single counter hid two different refusals.** "write-once" is a legitimate redraw
rejection; anything else is a real refusal. They are now counted separately, with the
reason recorded, which is how the bug above became visible at all.

**A third defect was in the shakedown's own leak check**: it compared generations globally,
so a later chunk's rebase legitimately reusing an earlier generation number flagged a chunk
that had never been preempted. A leak is a committed chunk at a generation that was
superseded *for that same chunk*.

## The GPU cleanliness contract

The free-memory gate had been lowered three times (7200 -> 6000 -> 4500 MiB) to make
experiments fit, which is the dangerous pattern rather than the ghost occupancy itself: an
A arm under 1.5 GiB of ghost and a B arm under 3 GiB differ in allocator pressure, clock
and power state, scheduling and OOM headroom, every one of which is available to be
misread as the effect under test.

`gpu_cleanliness.py` replaces it with **fixed constants** and gates on the right number:

    host used  -  WSL attributed  =  UNATTRIBUTED          limit 700 MiB (FIXED)
    free MiB                                               floor 6000 MiB (FIXED)

Observed on this machine: clean is 16-419 MiB unattributed; the ghost was 1671, then 3195
with zero attributed processes. The floor sits between the two populations with no
judgement call, and a run that cannot meet it is refused rather than accommodated.

    gpu used=23 free=7877 wsl_attributed=0 UNATTRIBUTED=23 (limit 700)
        util=0% pstate=P8 power=4.64W temp=51C   -> CLEAN

This is now the precondition for §Latency-3B-C's A/B, and it is not to be tuned per run.

## What this does not claim

- **No latency gain.** The numbers above are recorded, not asserted as a result.
- **No A/B.** Deliberately not run.
- **One input rate, one window, one geometry, one seed.**
- **The ghost occupancy's cause remains unknown.** §Runtime-GPU-1 is now a prerequisite
  for the A/B rather than a side track, but it is still not diagnosed: the fix so far is a
  refusal to sample, not a repair.

## Reproduce

    python gpu_cleanliness.py                    # the contract, one sample
    python demo_wasd.py --shakedown --input_period_ms 200 --seconds 25
