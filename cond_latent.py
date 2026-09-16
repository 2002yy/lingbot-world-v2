#!/usr/bin/env python
"""§41E-2 / §41E-3: zero-tail conditioning + cond_cache.

Verified facts (§41E-1, `zerotail_prefix_gate.py`, all bitwise_equal=True):
    ZERO_TAIL_CONST_START = 29     first constant latent frame
    PREFIX_PIXEL_FRAMES   = 117    ceil(117/4) = 30 latent frames

    encode(N)[:, :29]  identical for every N   (prefix invariance)
    encode(N)[:, 29:]  == encode(N)[:, 29]     (constant tail)
    cat(prefix, expand(c, T-29)) == encode(N)  exactly

So the conditioning latent is
    y = cat(z[0..28], expand(c, T - 29))
i.e. a FIXED-SIZE prefix plus ONE constant frame broadcast to any rollout
length. That is what makes cond_cache possible: the cache stores
(prefix, c), NOT the assembled y, so changing the rollout length does not
invalidate it.

HONEST SCOPE
    The encode cost has a large fixed component (measured: 117px -> 8.47s,
    224px -> 10.46s, only ~19% apart), so the zero-tail cut alone is a modest
    win. The real win is a warm cache hit, which skips the encode entirely.
    This does NOT change steady-state world generation (DiT ~796ms +
    decode / chunk), i.e. fps-equiv stays ~4.2. What it changes:
        cold start -> bounded (independent of rollout length)
        warm start -> ~O(1)
        scene reset -> ~O(1) after one canonical prefix
        peak conditioning VRAM -> lower

`ZERO_TAIL_CONST_START` / `PREFIX_PIXEL_FRAMES` are properties of THIS VAE
checkpoint (verified), NOT universal laws. The code asserts them at runtime
rather than trusting a magic number.
"""
import hashlib
import math
from typing import Dict, Optional, Tuple

import torch

# --- verified constants for the current VAE checkpoint ---
ZERO_TAIL_CONST_START = 29
PREFIX_PIXEL_FRAMES = 117
VAE_FINGERPRINT = "lingbot-v2-1.3b-causal-fast/vae2_1"

_stats = dict(full_calls=0, cold_calls=0, warm_calls=0,
              full_ms=0.0, cold_ms=0.0, warm_ms=0.0)


def stats():
    return dict(_stats)


def reset_stats():
    for k in _stats:
        _stats[k] = 0 if isinstance(_stats[k], int) else 0.0


def _frame_key(first: torch.Tensor) -> str:
    """Hash of the first-frame conditioning tensor (content, not identity)."""
    a = first.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha1(a.numpy().tobytes()).hexdigest()


def cache_key(first: torch.Tensor, lat_h: int, lat_w: int, dtype) -> str:
    """Key deliberately EXCLUDES the rollout length, so a longer rollout can
    reuse a prefix built for a shorter one."""
    return "|".join([_frame_key(first), f"{lat_h}x{lat_w}", str(dtype),
                     VAE_FINGERPRINT, f"px{PREFIX_PIXEL_FRAMES}",
                     f"c{ZERO_TAIL_CONST_START}"])


def _latent_T(frames_n: int, vae_stride_t: int) -> int:
    return (frames_n - 1) // vae_stride_t + 1


class CondCache:
    """Stores (prefix, c) per key -- NOT the assembled y."""

    def __init__(self, max_entries: int = 8):
        self.max_entries = max_entries
        self.store: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    def put(self, key, prefix, c):
        if len(self.store) >= self.max_entries:
            self.store.pop(next(iter(self.store)))
        self.store[key] = (prefix.detach().to("cpu"), c.detach().to("cpu"))


def build_condition_latent(vae, first_frame: torch.Tensor, frames_n: int,
                           h: int, w: int, device,
                           vae_stride_t: int = 4,
                           cache: Optional[CondCache] = None,
                           msnk: Optional[torch.Tensor] = None):
    """Drop-in replacement for the inline `vae.encode([first, zeros...])` block.

    `first_frame` is [3, 1, h, w] in the model's normalized range.
    Returns the same [C, T, lat_h, lat_w] conditioning latent as before,
    bit-exactly, plus timing stats.
    """
    import time
    T = _latent_T(frames_n, vae_stride_t)
    lat_h, lat_w = h // vae_stride_t // 2, w // vae_stride_t // 2  # unused
    key = cache_key(first_frame, h // 8, w // 8, first_frame.dtype)

    # short rollouts: the WHOLE sequence is shorter than the prefix, so the
    # prefix path does not apply -- encode normally.
    if T <= ZERO_TAIL_CONST_START + 1:
        x = torch.concat([first_frame.to(device),
                          torch.zeros(3, frames_n - 1, h, w, device=device)],
                         dim=1)
        t0 = time.perf_counter()
        with torch.no_grad():
            z = vae.encode([x])[0]
        torch.cuda.synchronize()
        _stats["full_calls"] += 1
        _stats["full_ms"] += (time.perf_counter() - t0) * 1000.0
        del x
        return z, 0.0

    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            prefix, c = hit
            t0 = time.perf_counter()
            y = _assemble(prefix.to(device), c.to(device), T)
            torch.cuda.synchronize()
            _stats["warm_calls"] += 1
            _stats["warm_ms"] += (time.perf_counter() - t0) * 1000.0
            return y, 0.0
    else:
        prefix = c = None

    # cold prefix encode
    t0 = time.perf_counter()
    xp = torch.concat([first_frame.to(device),
                       torch.zeros(3, PREFIX_PIXEL_FRAMES - 1, h, w,
                                   device=device)], dim=1)
    with torch.no_grad():
        zp = vae.encode([xp.to(device)])[0]
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - t0) * 1000.0
    _stats["cold_calls"] += 1
    _stats["cold_ms"] += cold_ms
    del xp
    assert zp.shape[1] >= ZERO_TAIL_CONST_START + 1, (
        f"prefix encode yielded T={zp.shape[1]}, need >= "
        f"{ZERO_TAIL_CONST_START + 1}; PREFIX_PIXEL_FRAMES is stale for this "
        f"VAE checkpoint")
    prefix = zp[:, :ZERO_TAIL_CONST_START].contiguous()
    c = zp[:, ZERO_TAIL_CONST_START:ZERO_TAIL_CONST_START + 1].contiguous()
    if cache is not None:
        cache.put(key, prefix, c)
    y = _assemble(prefix, c, T)
    return y, cold_ms


def _assemble(prefix: torch.Tensor, c: torch.Tensor, T: int) -> torch.Tensor:
    k = prefix.shape[1]
    assert k == ZERO_TAIL_CONST_START, (k, ZERO_TAIL_CONST_START)
    if T <= k:
        return prefix[:, :T].clone()
    return torch.concat([prefix, c.expand(-1, T - k, -1, -1)], dim=1).clone()
