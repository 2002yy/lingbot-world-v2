# Preview-1C: variant D cross-scene release gate

Variant D (skip the last spatial upsample, then a cheap bicubic upscale) involves no
training, so it cannot overfit a training set. But "cannot overfit" only removes one
failure mode -- it does not establish that the degradation is equally graceful in
other visual domains. This measures that, reusing the holdout data already generated
for the head gate.

No training. No tuning. No new components.

## Results

    scene                 d_vs_A     dedge   start d  gross  verdict
    H1_scene04           -0.0794   -0.1116   -0.0614      0  PASS
    H2_scene01           -0.0405   -0.0639   -0.0206      0  PASS
    H2_scene03           -0.0152   -0.0338   -0.0112      0  PASS

    d_vs_A spread across scenes: 0.0642
    gross structural failures: 0 everywhere

Per-scene detail:

    scene 01   D vs final 0.8602 / 0.7912   A vs final 0.9012 / 0.8584
               D vs A directly 0.9058
    scene 03   D vs final 0.8525 / 0.7846   A vs final 0.8692 / 0.8266
               D vs A directly 0.9125
    scene 04   D vs final 0.8602 / 0.7912 (unseen seed and controls)

## Verdict

**Variant D passes.** No gross structural failures on any scene, session-start is the
mildest window rather than the worst, and the degradation sits in a narrow band
(0.015 to 0.079) rather than swinging with the scene.

One detail worth keeping: **the degradation is not monotone in scene difference.**
Scene 01, the most different scene, degrades least (-0.0405) while scene 04, the
training scene, degrades most (-0.0794). So the mechanism is not "unfamiliar content
breaks it" -- it is a roughly constant ~0.04-0.08 loss from dropping the last
upsample's learned stage, which is exactly the behaviour expected of a training-free
operation. That is a stronger statement than "it happened to work on three scenes".

    FROZEN: variant D = production preview candidate
            ~231 ms preview, ~26.7 ms decode, ~+3.5% authoritative penalty,
            zero training, zero new weights, and now measured outside scene 04.

## The learned-head line is CLOSED

Two stages of head work produced a clear negative and a clear budget:

- Preview-Head-1A: Arm T failed the capacity gate at 0.8963 against a required 0.95,
  even as a pure overfit on nine samples. A 4-9 ms tail cannot faithfully reproduce
  the decoder's second half.
- Preview-Head-1B: the head does not generalise. H1 fails on the edge metric and H2
  fails clearly, with the failure size tracking scene difference (-0.049 / -0.062 /
  -0.218).

The ROI case is now explicit. A working head would move

    variant D       ~231 ms preview, ~26.7 ms decode, ~+3.5% penalty
    ideal head      ~220 ms preview, ~15-20 ms decode, ~+2% penalty

for roughly **10 ms of preview latency and 1-1.5 points of penalty**, at the cost of a
multi-scene dataset, a training pipeline, model version management, a cross-domain
generalisation gate, speculative-semantics risk, new weight assets and long-term
calibration.

And Arm T's capacity failure says the deeper problem is not only domain generalisation:
more data would not relieve a 223K tail's capacity constraint, and fitting a harder
problem with more data makes that constraint tighter, not looser. Option 1 would in
practice mean more scenes **plus** a larger head **plus** longer training **plus** a
relaxed 15-20 ms budget -- a different research project.

**Not permanently deleted, but demoted to future research**, to be reopened only if:
a multi-scene dataset already exists for another task; variant D proves insufficient
in quality or handoff; a head budget above 20 ms becomes acceptable; or a new
structure materially changes the quality-per-millisecond frontier.

## What remains unmeasured, and it is the thing users will notice

The product experience is two moments, not one:

    231 ms   something appears
    ~790 ms  the authoritative frame replaces it

The second is completely unmeasured. With D at roughly 0.86 SSIM against the final,
the preview is good enough for early directional feedback, but that says nothing about
whether the swap at ~790 ms is visually continuous. If the world "changes its mind"
there, the preview has made the experience worse, not better.

So the next stage is §Preview-2, handoff stability: SSIM and edgeSSIM difference
distribution across the swap, camera-motion direction consistency, whether large
structures jump, whether session-start is worst, and whether a direct replacement
versus a very short blend or crossfade measurably improves it.
