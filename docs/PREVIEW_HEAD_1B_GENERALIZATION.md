# Preview-Head-1B: generalisation gate -- the head does not generalise

## Setup

Two holdout axes, so a failure is attributable rather than vague:

    H1  same scene (04), unseen seed AND unseen control script
        -> is the head just memorising the nine training pairs?
    H2  different scenes entirely
        -> does stage1-feature -> final hold across visual domains?

The head was trained **once** on the nine training pairs, then frozen and only
inferred on. It is never retrained on a holdout, because the moment a holdout is
trained on it stops being a holdout.

**The headline metric is relative, not absolute.** The baseline gets harder or easier
depending on the scene, so an absolute SSIM on a new scene cannot be compared with an
old scene's number. What is reported is `dSSIM = head - full_preview` measured on the
SAME holdout.

## Results

    H1 (scene 04, unseen seed + controls)
      all            dSSIM -0.0486   dedge -0.0553
      session-start  dSSIM -0.0285   dedge -0.0553
      steady-state   dSSIM -0.0510   dedge -0.0706
      gross structural failures 0

    H2 scene 01
      all            dSSIM -0.2183   dedge -0.1316
      session-start  dSSIM -0.1816   dedge -0.1350
      gross structural failures 0

    H2 scene 03
      all            dSSIM -0.0615   dedge -0.1259
      session-start  dSSIM -0.0624   dedge -0.1218
      gross structural failures 0

    gate: dSSIM >= -0.05, dedge >= -0.05, gross failures 0, session-start alone

    H1           FAIL (dSSIM -0.0486 passes by 0.001, dedge -0.0553 fails)
    H2 scene01   FAIL (clear)
    H2 scene03   FAIL

## Verdict

**The head does not generalise.** Against the two-axis table this is the
H1-FAIL + H2-FAIL row: the route does not hold in its current form.

Reading it honestly:

- **H1 is borderline, not a pass.** dSSIM clears the gate by 0.001, which is noise,
  and dedge fails. So even within the training scene, on unseen seeds and unseen
  controls, the head is already slightly worse than the full TAEHV preview.
- **H2 fails clearly**, and the size of the failure tracks how different the scene is:
  -0.049 same scene, -0.062 on scene 03, -0.218 on scene 01. That ordering is the
  important detail -- it means the degradation is driven by visual domain, not by
  random variation.
- **No gross structural failures anywhere.** The head degrades gracefully rather than
  collapsing, which is why the aggregate numbers look survivable while the gate still
  fails. Graceful is not the same as adequate.

**So the Preview-Head-1A result -- 0.7796 against 0.7989 at 16.33 ms -- was largely a
nine-sample fit.** It should not be integrated, and the 16.33 ms figure should not be
quoted as an available product number.

## A bug in this round's own setup, worth recording

The first H2 run used `examples/00` and `examples/05` and produced **identical numbers
for both**. That is the signature of duplicated data, not of a robust result, and md5
confirmed it: `image.jpg` and `poses.npy` are byte-identical between the two. So the
"two scenes" were one scene twice.

Identical results across supposedly different inputs should always be treated as a
data problem before being treated as a finding. The re-run used `examples/01` and
`examples/03`, both genuinely distinct.

## What this does and does not say

It does **not** say a learned preview head is impossible. Preview-Head-1A already
established that the Arm T variant is not a faithful decoder replacement either, and
§35D's 9.8M-parameter result shows the latent-to-RGB mapping is learnable at that
scale.

What it says is narrower and more useful: **a 223K-parameter tail trained on nine
samples from one scene is not a general preview component.** The gap between "learnable
at 9.8M parameters over a real corpus" and "learnable at 223K over nine samples" is
exactly what this measured.

## Options, in order of evidence

1. **More training scenes before more architecture.** The failure ordering is
   scene-driven, so data is the first hypothesis to test. This is the cheapest next
   experiment and it directly addresses the measured cause.
2. **Reconsider the budget.** Arm T showed a 4-9 ms tail cannot faithfully reproduce
   the teacher; a head that must both compress the decoder and generalise may simply
   need more capacity than the 15-20 ms budget allows.
3. **Drop the head and keep variant D.** Preview-1B's `skip last upsample` gives
   ~231 ms preview and ~+3.5% penalty with graceful quality degradation and no
   training at all. It is unexciting, but it is measured, it generalises by
   construction, and it needs no new weights.

Option 3 deserves to be stated plainly: after two stages of head work, the
training-free variant D remains the only preview path whose behaviour is established
outside its own training set.

## Recommended next action

Do **not** integrate a head. Either test option 1 (add scenes, retrain, re-run this
exact gate) or take option 3 as the shipping preview and move to a different problem.
The one thing not to do is quote 16.33 ms as if it were available.
