# RTX 5060 Optimization Methodology

This file records the methodological rules learned from the RTX 5060 Laptop 8GB LingBot work.

## Rule 1 — Mirror operand order

A proxy expression must preserve production operand ordering when ordering affects numerics, kernel choice, or semantics.

## Rule 2 — Require execution evidence

A diagnostic hypothesis must be supported by actual execution evidence. Configuration or static inspection alone is not sufficient when runtime behavior is the claim.

## Rule 3 — Mirror tensor placement

Microbenchmarks must mirror production:

- shape;
- dtype;
- device;
- stride/layout;
- contiguous/transposed state;
- autocast/accumulation mode;
- relevant stream behavior.

A GPU tensor substituted for a production CPU tensor can invent synchronization costs that do not exist.

## Rule 4 — New and authoritative does not mean applicable

Before importing an external optimization, verify that the problem it solves exists in this repository.

Example class:

- paged KV may be valuable for multi-session service runtimes;
- it is not automatically useful for a single-session implementation that already preallocates fixed-window KV storage and writes in place.

## Rule 5 — Microbenchmarks rank candidates

Do not announce optimization benefit from a microbenchmark.

Benefit requires production-faithful end-to-end validation.

## Rule 6 — Sanity-check physical magnitude

Reject a result when the claimed subcomponent time exceeds the whole request/chunk or violates a basic upper bound.

Every report must pass a dimensional and magnitude sanity check before interpretation.

## Rule 7 — Short-horizon similarity is not long-horizon safety

Causal world-model rollout can amplify tiny numerical-path differences.

Same-latent or 21-chunk visual similarity must not be generalized to long-horizon trajectory preservation without a long rollout gate.

## Rule 8 — Harness is not production

A script named `production_loop.py`, `interactive_loop.py`, etc. does not gain authority from its filename.

Authority comes from matching the actual deployment path and declared contract.

## Rule 9 — Nominal geometry is not deployment geometry

Always report geometry.

The upstream/model nominal geometry may differ from the viable 8GB deployment geometry.

## Rule 10 — Decode-complete is not display-complete

A frame being decoded does not mean it was presented to the user.

Measured input-to-display requires an actual present boundary.

## Standard experiment header

Every new performance experiment should record:

```text
hardware:
driver:
OS / WSL:
torch:
CUDA runtime/toolkit:
model/checkpoint:
weight mode:
attention backend:
compile mode:
geometry:
latent geometry:
chunk size:
local window / sink:
decoder:
condition mode:
measurement denominator:
warmup:
repetitions:
```

## Standard result gates

For candidate optimizations, report:

1. correctness;
2. latency;
3. peak/resident VRAM;
4. repeated-run stability;
5. long-horizon behavior when the numerical path changes;
6. exact measurement denominator.

## Stop-loss principle

If production-faithful attribution shows the recoverable end-to-end value is below the project's current investment threshold, close the branch instead of writing a custom kernel for completeness.
