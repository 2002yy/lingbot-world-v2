# Preview-2A: the three-frame handoff audit

## The trajectory the user actually sees

    A = previous chunk's AUTHORITATIVE final
    P = current chunk's variant D(step0) preview
    F = current chunk's AUTHORITATIVE final

    A  --~231 ms-->  P  --~550 ms-->  F

Two handoffs, not one. `A -> P` asks whether the preview steps in the right direction
or jumps somewhere else; `P -> F` asks how much correction is left when authority takes
over; `A -> F` is the baseline, i.e. how much change would have happened anyway.

Note on provenance: A is the previous chunk's authoritative final, NOT a D-decoded
version of it. The user was looking at an authoritative frame, so that is where the
comparison must start.

## Results, 21 transitions over three scenes

    baseline change   A -> F   ssim 0.7678   L1 0.0486
    preview step      A -> P   ssim 0.7555   L1 0.0470
    remaining         P -> F   ssim 0.8524   L1 0.0295

    progress_ratio   d(A,P) / d(A,F)   0.987
    correction_ratio d(P,F) / d(A,F)   0.613
    latent direction cos(P-A, F-A)     0.867

    wrong-direction handoffs   0/21
    overshoot (correction > 1) 2/21
    gross structural jumps     0/21

    session-start  progress 0.961  correction 0.437  cos 0.915
    steady-state   progress 0.990  correction 0.622  cos 0.862

## How to read the ratios

`progress_ratio` and `correction_ratio` are the point of this stage, not raw SSIM. With
`cos = 0.867` and `|AP| ~ |AF|`, the geometry is: the preview takes a near
full-magnitude step in approximately the right direction but rotated by roughly 30
degrees, which leaves `d(P,F) ~ 0.6 |AF|`. The numbers are mutually consistent, which
is what makes them trustworthy rather than coincidental.

## Verdict: PASS on direction, MODERATE on correction

Three of the four gate conditions are clean:

- **direction is correct** -- cos 0.867, and **zero wrong-direction handoffs** across 21
  transitions and three scenes;
- **no gross structural jumps** -- 0/21;
- **session-start does not degrade** -- correction 0.437 there against 0.622 in steady
  state, so the handoff is mildest exactly where Preview-1A found the preview weakest.

The fourth is only moderate: **`correction_ratio` 0.613 means the handoff still carries
61% of the baseline visual change.** That is better than no preview at all (which would
carry 100%), but it is not the "preview already did the work, the handoff is a small
polish" picture. So a blend is not masking a wrong preview, but there is real
correction left for a blend to smooth.

Two of 21 transitions overshoot (`correction_ratio` 1.038 and 1.203), i.e. the preview
lands slightly further from F than A was. Both still have positive direction cosines
(0.864 and 0.771), so this is a magnitude overshoot rather than a direction error.

## What this decides

**Direct replacement is acceptable and should be the default.** The transition does not
go the wrong way, does not produce structural jumps, and is mildest at session-start.

**A blend is left as an open question rather than a conclusion.** The moderate
correction ratio is the case for testing one, and the clean direction is the case for
expecting a modest gain rather than a transformation. That is §Preview-2B's job, and it
should be measured rather than assumed -- the honest position is that a short crossfade
*might* help with 61% of the change still landing at the handoff, and it might just
soften a transition that was already acceptable.

## Standing caveat

All of this is offline, on nine chunks per scene across three scenes, with a single
seed per scene. It is enough to establish direction and the absence of wrong-way
handoffs; it is not enough to characterise the tail of the distribution, and
session-start here means chunks 2-3 because a transition needs a previous chunk.
