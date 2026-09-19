"""Attention backend selector for the causal-fast path.

BACKENDS
--------
    fa2     flash-attn 2 (the historical production path)
    sdpa    torch.nn.functional.scaled_dot_product_attention
    sage    SageAttention 2.2
    hybrid  per-path dispatch, chosen from MEASURED per-shape winners:
                long-window self-attention  -> sage
                cross-attention             -> sdpa
                short self-attention        -> sdpa

WHY HYBRID EXISTS
-----------------
Measured on this machine (RTX 5060, sm_120, bf16), per-call median:

    shape        FA2 ms   SDPA ms  Sage ms   best
    627x3762     0.8008   0.6273   0.4729   Sage   <- long self (window full)
    627x3135     0.6335   0.5249   0.4118   Sage
    627x2508     0.5615   0.4384   0.3351   Sage
    627x1881     0.4564   0.3416   0.3740   SDPA
    627x1254     0.3821   0.2394   0.2397   SDPA (~tie)
    627x627      0.2645   0.1393   0.1497   SDPA
    627x512      0.2570   0.1216   0.1455   SDPA   <- cross (text context)

Two things fall out of that table:

  1. flash-attn 2 is the SLOWEST option at every shape this model uses. SDPA beats
     it by 22-53%. That was the surprise of this investigation.
  2. Sage only wins where the KV window is long. At short KV and at
     cross-attention, SDPA is faster, so running Sage there is strictly worse:
     slower AND more quantisation error for nothing.

So "all-Sage" is not the optimum, it is an untuned dispatch. Hybrid sends each
path to its measured winner.

FALLBACK RULES (unchanged in spirit from the first revision)
------------------------------------------------------------
We only take over a call when the semantics match what the target kernel
supports. Anything else falls through to the original flash attention:
    * varlen/padded (k_lens not indicating a full batch) -> original
    * causal=True                                        -> original
    * window_size != (-1, -1)                            -> original
    * non-contiguous tensors                             -> original
    * dtype not fp16/bf16                                -> original
This guarantees enabling a backend can never silently change semantics for a
shape that backend cannot express.

HOT PATH DISCIPLINE
-------------------
Do NOT record/synchronise CUDA events here. An earlier revision did and it cost
~5040 device syncs per run, making the Sage arm falsely measure +1.0% slower.
The outer chunk timer is the only latency authority; this module only counts
dispatches.
"""
import os

import torch
import torch.nn.functional as F

from wan.modules.attention import attention as _orig_attention
from wan.modules.attention import flash_attention as _orig_flash_attention

VALID = ("fa2", "sdpa", "sage", "hybrid")

_state = dict(
    backend=os.environ.get("LINGBOT_ATTN_BACKEND", "fa2"),
    # Hybrid: KV length at or above which self-attention goes to Sage.
    # 627x1881 already favours SDPA, and 627x2508 favours Sage, so the measured
    # crossover sits between them. Only used by the "hybrid" backend.
    sage_min_kv=int(os.environ.get("LINGBOT_SAGE_MIN_KV", "2508")),
)
_counts = {k: 0 for k in VALID}
_fell_through = 0

_sageattn = None


def _load_sage():
    global _sageattn
    if _sageattn is None:
        from sageattention import sageattn
        _sageattn = sageattn
    return _sageattn


def set_backend(name):
    if name not in VALID:
        raise ValueError(f"unknown backend {name!r}; expected one of {VALID}")
    _state["backend"] = name


def get_backend():
    return _state["backend"]


def set_sage_min_kv(v):
    _state["sage_min_kv"] = int(v)


def stats(reset=False):
    s = dict(_state)
    s.update({f"n_{k}": v for k, v in _counts.items()})
    s["fell_through"] = _fell_through
    if reset:
        for k in _counts:
            _counts[k] = 0
        globals()["_fell_through"] = 0
    return s


def _can_offload(q, k, v, k_lens, causal, window_size):
    if causal:
        return False, "causal=True"
    if window_size is not None and tuple(window_size) != (-1, -1):
        return False, "window_size"
    if k_lens is not None:
        try:
            if not bool((k_lens == k.shape[1]).all()):
                return False, "k_lens padded"
        except Exception:
            return False, "k_lens"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return False, f"dtype {q.dtype}"
    for t in (q, k, v):
        if not t.is_contiguous():
            return False, "non-contiguous"
    return True, ""


def _sdpa(q, k, v):
    # q,k,v are [B, L, H, D] (NHD); SDPA wants [B, H, L, D].
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        is_causal=False).transpose(1, 2)


def _choose(backend, q, k, v, k_lens, causal, window_size, orig, kw, is_cross):
    """Return the callable to use, or None to fall through."""
    ok, why = _can_offload(q, k, v, k_lens, causal, window_size)
    if not ok:
        return None, why
    if backend == "fa2":
        return None, "fa2 passthrough"
    if backend == "sdpa":
        return (lambda: _sdpa(q, k, v)), ""
    if backend == "sage":
        sageattn = _load_sage()
        return (lambda: sageattn(q, k, v, tensor_layout="NHD",
                                 is_causal=False, smooth_k=True)), ""
    if backend == "hybrid":
        # Measured dispatch, by semantic path -- deliberately NOT a generic
        # shape threshold. Cross-attention is always SDPA because SDPA beats
        # Sage there by 19%; short self-attention likewise.
        if is_cross:
            return (lambda: _sdpa(q, k, v)), "hybrid:cross->sdpa"
        if k.shape[1] >= _state["sage_min_kv"]:
            sageattn = _load_sage()
            return (lambda: sageattn(q, k, v, tensor_layout="NHD",
                                     is_causal=False, smooth_k=True)), \
                "hybrid:longself->sage"
        return (lambda: _sdpa(q, k, v)), "hybrid:shortself->sdpa"
    return None, "unknown"


def _dispatch(q, k, v, q_lens, k_lens, causal, window_size, orig, kw,
              is_cross=False):
    global _fell_through
    if q_lens is not None:
        _fell_through += 1
        return orig(q, k, v, **kw)
    backend = _state["backend"]
    fn, why = _choose(backend, q, k, v, k_lens, causal, window_size,
                      orig, kw, is_cross)
    if fn is None:
        _fell_through += 1
        return orig(q, k, v, **kw)
    _counts[backend] += 1
    return fn()


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
              softmax_scale=None, q_scale=None, causal=False,
              window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16,
              fa_version=None):
    """Drop-in for wan.modules.attention.attention (self-attention path)."""
    kw = dict(q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
              softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
              window_size=window_size, deterministic=deterministic,
              dtype=dtype, fa_version=fa_version)
    return _dispatch(q, k, v, q_lens, k_lens, causal, window_size,
                     _orig_attention, kw, is_cross=False)


def flash_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                    softmax_scale=None, q_scale=None, causal=False,
                    window_size=(-1, -1), deterministic=False,
                    dtype=torch.bfloat16, version=None):
    """Drop-in for wan.modules.attention.flash_attention (cross-attn path)."""
    kw = dict(q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
              softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
              window_size=window_size, deterministic=deterministic,
              dtype=dtype, version=version)
    return _dispatch(q, k, v, q_lens, k_lens, causal, window_size,
                     _orig_flash_attention, kw, is_cross=True)
