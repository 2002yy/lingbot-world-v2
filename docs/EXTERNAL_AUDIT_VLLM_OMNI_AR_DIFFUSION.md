# External audit: vLLM-Omni Realtime AR-Diffusion sessions

Source: https://docs.vllm.ai/projects/vllm-omni/en/latest/design/feature/realtime_ar_diffusion/
Date: 2026-09-15, status: experimental. Fetched and read in full, not second-hand.

## What the doc actually provides

- "LingBot World v2 is the first single-KV-branch integration."
- One autoregressive latent block per request, model state retained across
  requests.
- Runner-owned paged KV (`ARDiffusionKVCacheSpec`,
  `managed_self_attention_pages(capacity)`).
- State is committed "only when returned metadata exactly equals the submitted
  tick snapshot".
- A session-owned causal Wan VAE encoder; the opening pixel frame is followed by
  four zero pixel frames per later latent frame.
- DreamZero reports 603 MiB per-session Wan VAE causal-convolution state.
- Non-goals explicitly include "stateful streaming VAE decode".
- Single replica, `max_num_seqs == 1`, LRU eviction, Ulysses SP (strict only).

## Correction: the paged-KV premise does not apply to us

The stated motivation was that LingBot-style implementations cat/clone an entire
KV block as the window grows. Our implementation does not:

    image2video.py:835   _initialize_self_kv_cache(shape=[1, kv_size, h, d])
                         kv_size = fsl * local_attn_size   -- preallocated once
    model_fast.py:219    kv_cache["k"][:, start:end] = roped_key   -- in place
    model_fast.py:234    same for v
    model_fast.py:224    the only clone, of num_rolled_tokens, to avoid
                         overlapping memory during the rolling-window memmove

So we already have a preallocated rolling window with in-place writes. Moreover
paged KV's benefit is multi-session serving: capacity accounting, LRU eviction,
per-session residency, a `gpu_memory_fraction` expansion budget. We are a single
session on 8 GB, so those benefits largely do not transfer.

**Verdict: paged KV is not adopted. It would optimise a problem we do not have.**

## What is genuinely transferable

### 1. Transactional commit

"Reducer prepare() is speculative; reducer commit() and the session snapshot form
one logical commit." On failure the chunk index and queued events stay
uncommitted and the session goes to FAILED; the chunk is not retried in place.

We have `state_commit*.py`, but the prepare/commit split with an exact-metadata
match condition is stricter and clearer than what we have. It matters for the
same reason it does there: a control event applied at a tick must not advance
state if that tick failed or was poisoned. Suggested change: make the
interactive loop's state commit two-phase, with the commit condition being that
the returned `(chunk_index, applied_event_ids)` equal the submitted tick exactly.

### 2. Committed vs in-flight state separation

"It retains one current condition block, with separate committed and in-flight
encoder histories so a failed block cannot overwrite committed context."

This is the general form of the cam-cache poisoning bug we hit: `pipe.prewarm()`
ran a dummy forward at `current_start=0` and poisoned chunk 0 of the real
generation. Our fix was an explicit `_CAM_EPOCH` counter with
`key=(epoch, current_start)`. Their design solves the same class structurally
with two histories. Our fix is sound; the lesson is that if more cache points
appear, prefer explicit committed/in-flight separation over adding more epoch
dimensions.

## What it cannot solve

"Stateful streaming VAE decode" is a declared non-goal, so this runtime does not
provide the decode half of input-to-display. That half still needs our TAE-HV.
vLLM-Omni should therefore not be adopted as an input-to-display solution.

We also already own that measurement and do not need it from outside:

    hotswap_loop.py:19      control_to_real = t_chunk_visible_after_input - t_input
    production_loop.py:237  control-to-real p50 ~950 ms, worst ~1.6 s
    pose_warp.py:11         control-to-warp 0.60 ms and control-to-real
    preview_head.py         two-layer: preview TTFNF ~780-820 ms vs ~1058 ms full

## Usable numbers and ideas

- 603 MiB per session of Wan VAE causal-conv state (DreamZero, measured) is a
  reference anchor for our own resident-state budget.
- `effective_budget = max(configured_budget, required(1))`, i.e. always admit one
  viable session first, is a sane admission policy for an 8 GB card.
- The condition-encoding convention (one real pixel frame, then four zero pixel
  frames per later latent frame) matches ours and independently corroborates it.

## Ruling

Priority item added and immediately resolved: audit done, **vLLM-Omni is not
adopted as our runtime.** Adopt its contract ideas, not its code: two-phase
prepare/commit with an exact-metadata commit condition; committed/in-flight
history separation (already achieved via epochs); `required(1)`-first admission.
Not adopted: paged KV (single session), session migration, Ulysses SP (single
GPU). Not expected from it: streaming decode.

Orthogonal to the Sage/compile closure -- nothing here reopens the attention
backend correctness question.

## Not independently verified

The claim that no trusted consumer-card LingBot-World 1.3B measurement beats
about 3.1 FPS-equiv as of 2026-09-28, and the ~120 GB FlashDreams figure, were
not verified. Our route rests on our own measurements (repro 1566.6 ms, fast
1450.3 ms, control-to-real p50 ~950 ms), not on external rankings.

## Methodology note

New and authoritative does not mean applicable. Adopting paged KV here would have
optimised a problem we do not have while the real finding -- the transactional
commit discipline -- was easy to miss. The first step in auditing an external
design is always to verify that the problem it solves exists in our code.
