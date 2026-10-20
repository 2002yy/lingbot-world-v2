# KV-1A: exact-path decomposition of the 174 ms KV/state update

## The question that decided the route

Is the KV update's cost O(new block), or does it actually move a large part of the
rolling window every tick?

The code does contain the pattern worth hunting for:

    num_evicted_tokens = num_new_tokens + _lei - kv_cache_size
    num_rolled_tokens  = _lei - num_evicted_tokens - sink_tokens
    kv_cache["k"][:, sink:sink+rolled] = kv_cache["k"][:, sink+evicted:...].clone()
    kv_cache["v"][:, sink:sink+rolled] = kv_cache["v"][:, sink+evicted:...].clone()

so the O(window) memmove and its clone exist. The only open question was magnitude, and
a magnitude that looks alarming at 231M elements is not necessarily alarming in
milliseconds. That is exactly the kind of estimate that has to be checked rather than
trusted.

Nothing was optimised. This is measurement only.

## Result

    denoise steps (3)             541.9 ms   (180.6 ms per step)
    KV/state update forward       170.7 ms

    KV-update / denoise-step ratio   0.945

    rolling-window traffic, steady state, one eviction:
      rolled tokens per layer     2508
      elements moved (k+v, 30 layers)  231.1 M
      bytes read + written        924.5 MB
      at 300 GB/s                 ~3.08 ms
      as a share of the KV update  1.8%

## Verdict: no large exact-path redundancy; the line closes

The KV update at 0.945x a denoise step **is essentially one more full model forward**.
Its cost is block compute -- the same attention and FFN the denoise steps perform -- and
the rolling memmove is 1.8% of it, about 3 ms.

So the 22.8% labelled "KV/state update" is not cache overhead that could be engineered
away. It is the fourth full forward of the chunk, and there is no same-math version of
it that costs materially less.

That closes KV-1B before it starts, which is the useful outcome: the branch was worth
one measurement and it does not need more.

## The structural fact this exposes

The causal design runs **four full forwards per chunk**: three denoising steps and one
state-write. The state-write cannot simply be merged into a denoising step, because the
KV must hold the CLEAN `x0` representation while the denoising steps consume noisy
latents -- different inputs, so different forwards.

Reducing the forward count therefore means changing what the KV holds, which is a
numerical change, not a same-math optimisation. Quantum-1A already showed where that
leads: monotonic unplateaued state divergence.

## Where this leaves the levers, honestly

    3 denoise forwards   541.9 ms   reducing them changes the state (Quantum-1A: FAIL)
    1 state-write        170.7 ms   structurally required to store the clean latent
    decode                36.6 ms   Preview-1B: already at the floor without a head
    control plane          2.4 ms   negligible

Every exact-path lever on the authoritative chunk is now either closed or accounted
for. The remaining candidates are not cost reductions at all:

- **early-exit**, which selects a step count per chunk. The safety predictor would have
  to be very reliable, and there is an unfavourable correlation: the chunks that most
  need low latency are the ones where a control just arrived and the world is changing
  most, which are plausibly the least safe to exit early on.
- **pipelining or scheduling**, which hides the quantum rather than shrinking it.
- **accepting the quantum**, and relying on the preview layer that is already frozen at
  ~231 ms.

The preview layer took "nothing visible for 750 ms" to "causally-consistent feedback at
231 ms". What KV-1A adds is confidence that the remaining authoritative latency is not
hiding an easy win.
