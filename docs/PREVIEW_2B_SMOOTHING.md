# Preview-2B: the P -> F handoff smoothing gate

The question is not "are adjacent frames closer after a crossfade", which is close to
mathematically certain. It is whether spreading the `P -> F` correction over 50-100 ms
is worth entering the authoritative image later and risking ghosting.

Preview-2A said what is being smoothed: `correction_ratio` 0.613, direction cosine
0.867, zero wrong-direction handoffs, zero gross jumps. So a blend rescues nothing --
it softens a moderate, correctly-directed correction. That is exactly the case where a
short blend may or may not be perceptible.

## Results, 21 transitions over three scenes

    arm         frames    J_peak   vs hard   J_total  vs hard  edge dip  inter edgeSSIM
    hard             2    0.0295     1.00x    0.0295    1.00x     0.000          --
    blend50          5    0.0074     0.25x    0.0295    1.00x     0.002      0.8285
    blend100         8    0.0042     0.14x    0.0295    1.00x     0.007      0.8051

    gate: peak down >= 30-40%, no gross artifact, no edge collapse,
          total not materially increased, and the 50 ms gain must be real

    blend50   PASS   |   blend100   PASS

## Why `J_total` is exactly 1.00x, and why that matters

For a linear crossfade the intermediate frames are collinear in pixel space between P
and F, and L1 distance is additive along a segment, so the path length is
mathematically identical to the hard replacement. The blend spreads the same
correction rather than routing through a longer, stranger path.

That is the ideal outcome and it is a property of the construction, not a lucky
measurement -- which also means it should not be over-claimed as evidence of quality.
What IS measured rather than assumed is the ghosting metric: the largest edge-energy
dip is 0.002 for 50 ms and 0.007 for 100 ms, i.e. negligible. A linear blend of two
images that differ mainly in low-frequency structure does not create the double-edge
artifacts the metric was there to catch.

## Decision: adopt the 50 ms blend

    peak jump      0.0295 -> 0.0074   (-75%)
    total path     1.00x               (unchanged)
    edge dip       0.002               (negligible)
    cost           50 ms of authority convergence delay

The 100 ms blend reduces the peak further, to 0.14x, but doubles the convergence delay
for a change that is already barely perceptible at 0.25x. So the answer to "is 50 ms
better than 100 ms as a product point" is yes, and it matches the prior that this is a
de-spiking operation rather than a visible dissolve.

**A display-semantics cost, stated as such:** after this the timeline is

    231 ms   preview appears
    790 ms   authoritative frame computed
    840 ms   the blend completes and the display is 100% authoritative

The extra 50 ms is not compute latency. It is a deliberate delay in when the
authoritative image fully governs the display, paid to remove a 75% peak jump.

## The Preview line is CLOSED

    Preview-1A  real preview latency measured: 241 ms, +5.7% penalty, quality 0.8268
    Preview-1B  cheap-decode frontier mapped; latent-downsample variants collapsed
                quality; skip-last-upsample (variant D) degrades gracefully at
                26.67 ms and misses the 15-20 ms target
    Preview-Head-1A/1B  learned heads: Arm T failed the capacity gate even as an
                overfit, and the head does not generalise across scenes.
                Line closed, demoted to future research.
    Preview-1C  variant D passes the cross-scene release gate; frozen as the
                production preview candidate
    Preview-2A  three-frame handoff audit: direction clean, correction moderate
    Preview-2B  handoff smoothing: 50 ms blend adopted

    SHIPPING PREVIEW PATH
      ~231 ms   variant D preview, no training, no new weights
      ~840 ms   100% authoritative, after a 50 ms blend
      cost      ~+3.5% authoritative penalty, ~26.7 ms decode, 50 ms display
                convergence delay

## What is NOT measured, honestly

Everything here is offline, on nine chunks per scene across three scenes, one seed per
scene, with a synthesized display sequence at an assumed 60 Hz cadence. There is no
t4/t5 instrumentation anywhere in this project, so none of this is a present-time
measurement; the convergence delay in particular is display semantics, not measured
latency.

The blur/ghosting test is an edge-energy and edgeSSIM proxy, not a perceptual study. It
is adequate to rule out gross double-edging, which is what it was for; it is not
adequate to claim the blend is imperceptible.

## Where attention goes next

The preview line has taken "750 ms before anything is visible" down to "about 231 ms of
causally-consistent visual feedback", and it did so with no training and no new
weights. The remaining large question is no longer in the preview layer at all: it is
the authoritative model quantum itself, where Latency-2A measured DiT steps at 71.8%
and the KV update at 22.8% of the chunk. That is where the next stage belongs.
