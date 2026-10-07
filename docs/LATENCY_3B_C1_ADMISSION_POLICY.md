# Latency-3B-C1: preemption admission policy

    STATUS   policy in place with every rejection reason counted separately.
             THE CENTRAL HYPOTHESIS WAS FALSIFIED, which is the main result.
    SCOPE    only "is this worth interrupting for". Generation, KV, RNG, commit and the
             synchronisation method are untouched.
    NEXT     the rate is structurally one per chunk at this input rate and admission
             rules cannot change that; see the bottom.

## The policy

    admit iff
        forward_index is an allowed boundary       default {1}, the conservative cut
        AND remaining forwards >= 2                default
        AND budget remains                         one per chunk
        AND the control intent actually changed

Every rejection is counted under its own name -- `rejected_same_state`,
`rejected_too_late`, `rejected_budget` -- because C0 lost a real bug behind a single
conflated counter and that is not worth repeating.

## Result at the same 200 ms input rate as C0

    peeks that found input   125
    admitted                  31
    rejected_same_state        0      <- the semantic check never fired
    rejected_too_late         70      <- boundaries 2 and 3 are not candidates
    rejected_budget           24      <- the post-rebase attempt is not re-preemptible

    preemption rate        31/37 chunks        C0 was 31/32
    observed -> admitted   p50 82.25 ms
    input -> first real    p50 878 ms          C0 1010, no-preemption 994-999

## The hypothesis was wrong, and that is the finding

The expectation going in was that `31/32` was largely the same held control state
arriving repeatedly and being treated as a fresh decision each time. **It rejected
zero.** The shakedown cycles W, D, S, A, so each arriving intent genuinely differs from
the one bound to the running chunk. **"Holding the same direction" never happens in this
input pattern**, so the semantic check cannot be what limits the rate here.

What actually sets the rate is arithmetic: a chunk takes roughly 700-800 ms and the
shakedown delivers an intent every 200 ms, so **about 3.4 inputs land per chunk**. A
pending input is therefore essentially always available at the first boundary, whatever
the admission rule says. The boundary restriction rejects 70 peeks and still admits 31,
because it is not the boundary that decides -- the arrival rate is.

**So `31/32` was never a policy defect.** It is what "input arriving faster than the
system can consume it" looks like when the budget is one per chunk.

## What this implies for the next slice

Reducing the rate to the 10-30% range the plan hoped for cannot be done by admission
rules, because at 3.4 inputs per chunk *every* chunk legitimately has a changed intent
pending. It needs one of:

    a rate limit          e.g. at most one preemption every N chunks
    a stability criterion preempt only if the pending intent has been unchanged for some
                          interval, which trades responsiveness for calm
    a larger benefit gate a numeric estimate of saved work, which the policy deliberately
                          does not have yet

All three are product decisions about the responsiveness/throughput trade, not
correctness fixes, and none of them is obviously right. **They should be decided
deliberately rather than inferred from a run.**

What the boundary rule does appear to buy, at n=1 per arm and therefore recorded rather
than announced: C0 allowed boundaries {1,2} and measured 1010 ms; C1 allows {1} and
measured 878 ms. That is consistent with the theory that a rebase later in a chunk wastes
more work than it recovers, and it is the one part of this policy that looks worth
keeping regardless of how the rate question is settled.

## Gates

    preemption fired                                  PASS  31
    several chunks committed                          PASS  37 chunks, 125 inputs
    no chunk restarted more than once                 PASS
    no event committed twice                          PASS
    no stale generation leaked                        PASS
    a preview never advanced committed state           PASS
    every committed frame's t4 recorded                PASS  36/36
    processed/settled explainable                      PASS  gap 0
    frames kept coming                                 PASS
    the post-rebase attempt committed                  PASS
    the admission policy actually rejected something    PASS  94 rejections

## What this does not claim

- **No latency gain.** 878 against 1010 and 994-999 is one run per arm on a machine whose
  state varies; the difference is recorded, not asserted. `3B-C0`'s own rule is that
  latency does not get to decide whether something is right, and its corollary is that one
  run does not get to declare a gain either.
- **The rate is not fixed.** It is bounded at one per chunk, which is the liveness
  guarantee, and it remains near one per chunk.
- **One input rate, one pattern, one geometry.** A slower input pattern would exercise the
  semantic check; this one never does, which is itself the finding.
