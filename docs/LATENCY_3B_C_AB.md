# Latency-3B-C: the formal A/B for preemption policy v1

    STATUS   RUN, interleaved, on the load that represents play. Policy v1 is faster at
             p50, p90 and p95, with no cadence loss and one fifth the variance.
    BUT      the first attempt at this measurement was WRONG, and how it was wrong is
             part of the result.

## The frozen policy under test

    Default preemption policy v1
      1  only a semantic control change is a candidate
      2  only after forward 1
      3  at most one preemption per chunk
      4  inputs arriving during the replay go to the next chunk
      5  no additional cooldown
      6  no stability debounce
      7  no benefit prediction

The `same_state` check stays even though the rapid load never triggers it. It is not there
for the C1 test pattern; it is there so that holding a direction cannot become a reason to
restart a chunk, which the real keyboard path mostly prevents anyway and which a future
input path might not.

## The measurement that was wrong first

Six arms run back to back, A then B per load. The result was:

    hold     +87 ms      <- B SLOWER, with ONE preemption in 50 chunks
    normal  -115 ms
    rapid   -196 ms

**One preemption in fifty chunks cannot cost 87 ms.** The tell was the device state
recorded alongside each arm: the first arm ran at 46 C / P8, and every later arm at 65-69 C
/ P4. Sequential arms on a shared machine cannot attribute a ~100 ms difference, because
the drift loads onto whichever arm ran second -- and that is true whether or not a warm-up
is added, which was tried next and made the `hold` pair worse (+133 ms) while the start
temperatures still differed by 15 C.

This is the same failure the GPU cleanliness contract exists to prevent, one level deeper:
the contract gates VRAM against fixed thresholds, and the first A/B still let the machine's
thermal state become the effect under test.

## The measurement that stands: interleaved

    order A B A B A B, so drift loads on both arms instead of one
    load  normal -- ~1 s per direction change, which is what play looks like
    thermal state recorded at the head and tail of every arm

    arm  round  p50   p90   p95   o->submit  chunks  pre  waste  cadence
    A    1       967  1354  1413       1012      38    0      0     1.52
    A    2       956  1208  1216        974      42    0      0     1.68
    A    3       821  1061  1096        840      46    0      0     1.84
    B    1       709   928   935        731      43    9      9     1.72
    B    2       693   919   932        713      43   11     11     1.72
    B    3       724  1025  1031        758      40    6      6     1.60

    median p50   A 956 ms      B 709 ms      delta -247 ms
    spread       A 146 ms      B  31 ms

**B is faster at the median and at p90 and p95, holds cadence, wastes 3.8-6.4% of forward
work, and is five times tighter run to run.** The variance reduction is the more striking
half: an interactive system that is 250 ms faster on average and far less variable is a
different product from one that is merely faster on average.

## What this does to the C1 conclusion

**The 31/37 preemption rate was a property of the 200 ms pressure load, not of play.** At
the normal load the rate is **15-26%**, which is the non-pathological band the plan was
aiming for. C1 called it pathological because it only had the rapid load; the honest
correction is that the rapid load is a stress test, not a description of a user.

Which also means the three mechanisms C1 listed -- rate limit, stability debounce, benefit
prediction -- are **not needed**. Nothing was wrong. The correct action was to measure
before limiting, and this is why the plan's insistence on an A/B before adding a rate cap
was right.

## What is still not established

- **One load, one geometry, one weight, one seed, three rounds.** The direction is
  consistent across p50, p90, p95, submit time and three interleaved rounds, which is
  stronger than the sequential mistake was, but it remains a small sample.
- **`hold` and `rapid` were measured sequentially and are therefore not attributed.** The
  hold pair is actively misleading and should be re-measured interleaved before anything is
  said about it; rapid's -196 ms agrees in direction with normal but shares the same
  sequential risk.
- **`observed` and `accepted` coincide.** `accept()` is given the input's own timestamp, so
  the two series are the same and are reported once. The cost preemption itself charges is
  `observed -> admitted`, p50 82-121 ms, and it is included in every figure above rather
  than netted out.
- **Wasted work has not been converted to anything.** 3.8-6.4% of forwards is discarded;
  what that costs in power, heat or long-run throughput on a laptop is not measured.

## Reproduce

    # the contract first; it refuses rather than being tuned
    python gpu_cleanliness.py

    # interleaved, which is the only ordering that attributes anything
    for r in 1 2 3; do
      for arm in A B; do
        python demo_wasd.py --shakedown --load normal \
          --preempt_boundaries "$([ $arm = A ] && echo '' || echo 1)" \
          --arm $arm --seconds 25 --pixel 304x528 --weight bf16 \
          --out_json output/c_ab_interleaved/r${r}_${arm}.json
      done
    done
