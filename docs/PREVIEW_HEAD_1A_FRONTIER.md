# Preview-Head-1A: can a 4-9 ms tail replace TAEHV's ~21 ms learned tail?

## Setup, and what was deliberately kept apart

Two objectives, same architecture, same nine samples, same latency budget, so the
comparison is interpretable:

    Arm T  stage1_feature(step0) -> full TAEHV(step0) RGB    capacity gate
    Arm F  stage1_feature(step0) -> authoritative FINAL RGB  product value

Bundling them would conflate "how small can the decoder's second half be made" with
"is it better if the tail also predicts what the DiT has not produced yet".

**The prefix is frozen.** TAEHV stage 0+1 weights are never fine-tuned. Letting the
prefix train with the tail would give prettier loss curves while destroying the
property that matters -- a stable prefix plus a pluggable tail -- and would
reintroduce model version and weight compatibility surface.

Capacity is defined by **measured warm CUDA latency**, not parameter count: the prefix
is 10.84 ms, so a 15 ms total allows ~4.2 ms of tail and a 20 ms total ~9.2 ms.

Attach point, captured from a real decode: `child 14` output is **[1, 64, 264, 152]**,
so the tail performs 64ch @ 264x152 -> 3ch @ 528x304.

## Results

    Arm T (distil the teacher)
    size   width blocks  params    tail ms  total ms    ssim     edge   start ssim
    S         16      1   39600      0.81     11.65   0.8261   0.7831    0.8085
    M         32      1   83808      1.72     12.56   0.8619   0.8346    0.8451
    L         64      2  222912      5.14     15.98   0.8963   0.8818    0.8849

    Arm F (predict the final)
    S         16      1   39600      0.62     11.46   0.7371   0.6723    0.6613
    M         32      1   83808      1.82     12.66   0.7550   0.6905    0.6765
    L         64      2  222912      5.49     16.33   0.7796   0.7251    0.6985

    reference: full TAEHV preview(step0) vs final   0.7989 / 0.7662
               (session-start baseline from Preview-1A   0.6871 / 0.6352)
    prefix alone  10.84 ms | what a tail replaces (stage2 + final)  20.73 ms
    full TAEHV    36.01 ms

## Verdict

**Arm T fails the capacity gate.** The best is 0.8963 / 0.8818 against a required
0.95 / 0.90, even in a pure overfit setting with 600 steps on nine samples. So a
4-9 ms tail is not a faithful replacement for the decoder's second half. That is a
real, useful negative: it says the learned tail is not compressible to this budget
without loss, and it is the reason Arm F's numbers should be read as an approximation
rather than as a decoder.

**Arm F passes the product gate.** At size L, 16.33 ms total, it reaches 0.7796 /
0.7251 against the final, versus the full-TAEHV baseline of 0.7989 / 0.7662. That is
0.019 / 0.041 lower, inside the 0.03-0.05 allowance, at **45% of the decode cost**
(16.33 vs 36.01 ms).

**And at session-start it is better than the baseline.** Arm F L reaches 0.6985 on
chunks 1-2 against the baseline's 0.6871. Session-start is the window Preview-1A
identified as the weakest and the one a user sees first, so this is not a marginal
detail -- the head is strongest exactly where the pipeline was weakest.

**Runtime: 16.33 ms against the 20 ms budget passes; the preferred 15 ms misses by
1.3 ms.**

## The interpretation the two arms support

The expected shape appeared, though not in the direction first guessed: Arm T has the
higher teacher fidelity (0.8963) while Arm F has the higher final fidelity (0.7796
against 0.7989), and they measure different things.

What the pair establishes is that **this tail is not a decoder compressor -- it is a
light predictive head.** Trained against the teacher it plateaus below the capacity
gate; trained against the final frame it matches the baseline overall and exceeds it
at session-start, because predicting the final is precisely what helps where the
step0-to-final gap is largest. That is a different component from the one §35B
proposed, and it is the one the measurements support.

## Limits, stated before anyone builds on this

Nine samples and 600 steps is an **overfit test, not a generalisation test**. Nothing
here says the head generalises to unseen controls or scenes. Held-out validation is
the next step and has not been done.

Latency was measured with a single-sample batch at the exact production shape, warm,
with CUDA events. The end-to-end integration penalty has not been measured, and
Preview-1A showed that a serialised decode lands almost entirely on the authoritative
path -- so the 20 ms saving has to be re-measured in the real loop before it can be
claimed as a product number.

## Next, in order

1. Held-out validation on controls and chunks not in this set.
2. A real-loop integration measurement, since Preview-1A established that the decode
   cost transfers almost one-for-one into the authoritative penalty.
3. Only then decide whether the predictive semantics of Arm F are acceptable -- a
   head that anticipates the final frame is making a speculative claim, and that is a
   product decision rather than a measurement one.
