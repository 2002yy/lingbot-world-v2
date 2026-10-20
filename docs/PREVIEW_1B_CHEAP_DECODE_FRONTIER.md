# Preview-1B: the cheap-decode frontier

The causal chain from Preview-1A was clean: step0 latent ready ~204 ms, preview
decode 36.7 ms, preview visible 241.5 ms, authoritative penalty +43.4 ms. The
penalty is almost exactly the extra decode, so halving the decode would halve the
penalty. This stage asks how cheap the decode can get without losing the thing the
preview is for.

## Step 1: decompose the 36.7 ms

Per-child exclusive time via hooks plus CUDA events (hooks avoid reimplementing the
MemBlock memory semantics; CUDA events record asynchronously so no per-child
synchronisation is needed).

    idx module                    exclusive     pct  calls
     3-8   stage 0 (MemBlock x3, Upsample, TGrow, conv)    4.99 ms   14%   1 frame
     9-14  stage 1                                         5.85 ms   16%   2 frames
    15-20  stage 2                                        17.31 ms   48%   4 frames
    21-22  final ReLU + conv                               3.42 ms   10%   4 frames
    --------------------------------------------------------------
    total 35.84 ms, unattributed 4.14 ms

The structural reason stage 2 dominates: **temporal growth means later stages process
more frames** (1 -> 2 -> 4), so the last stage runs at the highest spatial resolution
AND at four frames. The expensive tail is real, and it is not merely a
high-resolution problem.

## Step 2: the frontier

    variant                          ms      vs A
    A full TAE                    36.01     +0.0%
    B latent /2 + upscale         10.94    -69.6%
    C latent /4 + upscale          5.14    -85.7%
    D skip last upsample          26.67    -25.9%

Quality measured against the **authoritative final frame** -- the same yardstick
Preview-1A used, because comparing a cheap decode of step0 against the full decode of
step0 would measure decoder fidelity rather than preview usability:

    window          variant   ssim p50   edgeSSIM p50
    session-start   A          0.6871       0.6352
    session-start   D          0.6324       0.5554
    steady-state    A          0.8268       0.8003
    steady-state    D          0.7410       0.6739
    all             A          0.8268       0.8003
    all             B          0.3287       0.2992
    all             C          0.3082       0.2626
    all             D          0.7402       0.6715

## What the frontier says

**B and C hit the cost target and destroy the quality.** Latent downsampling feeds
the decoder an out-of-distribution input, and because the decoder is nonlinear the
result is a structurally different image rather than a blurred version of the right
one. 0.33 / 0.30 is a collapse, not a graceful degradation. Closed.

**D degrades gracefully but does not reach the target.** Skipping the last spatial
upsample keeps every learned convolution and the input in distribution, so quality
falls from 0.83 to 0.74 in aggregate and from 0.69 to 0.63 at session start -- real,
but usable. Its cost is 26.67 ms, which misses the <= 15-20 ms target.

**So the existing TAE cannot reach ~20 ms while preserving structural quality.** That
was the stated stop condition for opening a dedicated preview head.

## Product consequence, stated precisely

D is still worth having as an interim even though it misses the decoder target,
because the authoritative penalty scales with the decode:

    full TAE   preview 241 ms   penalty +43.4 ms (+5.7%)
    D          preview ~231 ms  penalty ~26.7 ms (+3.5%)

So D buys roughly 10 ms of preview latency and 2.2 points of authoritative penalty,
at the cost of a quality drop from 0.83 to 0.74 that is graceful rather than
structural, and with session-start -- the weakest window, and the one a user sees
first -- moving from 0.69 to 0.63.

## Verdict

    decoder <= 15-20 ms          MISS (best quality-preserving option is 26.67 ms)
    preview ~220-225 ms          MISS but close (D gives ~231 ms)
    authoritative penalty <= 3%  NEAR (D gives ~3.5%)

B and C closed. D accepted as an interim with its numbers recorded rather than
described. §Preview-Head-1 is now justified by the measured frontier rather than by
preference: no amount of truncating this decoder reaches the target without breaking
structure.

## A measurement error worth recording

The first version compared cheap(step0) against full(step0), which produced 0.29-0.36
for B and read as a catastrophic result. That comparison measures how faithfully the
cheap path reproduces the full decoder, which is not the question. A preview exists to
tell the user the world is moving in the right direction, so the reference must be the
authoritative final frame. The corrected measurement still fails B, but for the right
reason and with the right number, and it is what makes D's 0.74 interpretable.
