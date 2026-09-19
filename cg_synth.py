#!/usr/bin/env python
"""P3 step 1: minimal synthetic isolation -- eager-input cache vs registered buffer.

QUESTION (one question only)
----------------------------
PyTorch skips CUDA Graph capture when a *mutated eager input* is present:

    skipping cudagraphs due to mutated inputs (2 instances)
        at crossattn_cache["k"].copy_(k)

CUDAGraph Trees documents that mutation of *parameters/buffers* is a supported
scenario while mutation of *eager inputs* is not. Before touching the real model
(30+ layers, invasive) we verify that distinction in the smallest possible case,
with a real matmul/softmax attention so Inductor actually generates kernels.

THREE ARMS
----------
  eager_in : cache passed as a function argument, mutated in place
  buffer   : cache held as a registered nn.Module buffer, mutated in place
  eager_out: cache passed in, NOT mutated -- new k/v returned instead
             (the "functional" pattern, the invasive option we are deferring;
              included because it costs one line here and tells us the answer)

For each arm we report
  * whether "skipping cudagraphs" was emitted (and why)
  * whether a CUDAGraph was actually captured (inspected via the inductor
    cudagraph manager, not merely "the run did not raise")
  * median latency
  * numerical agreement against the eager reference

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCH_LOGS=perf_hints python cg_synth.py
"""
import os
import statistics
import time

import torch
import torch.nn as nn

D = 512
L = 64          # tokens added per call
CACHE = 256     # cache capacity


class AttnEagerIn(nn.Module):
    """cache arrives as an eager input and is mutated in place."""
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(D, 3 * D)

    def forward(self, x, cache_k, cache_v):
        q, k, v = self.w(x).chunk(3, -1)
        cache_k[:, :L] = k          # <-- mutate an EAGER INPUT
        cache_v[:, :L] = v
        att = torch.softmax(q @ cache_k.transpose(-1, -2), -1) @ cache_v
        return x + att


class AttnBuffer(nn.Module):
    """cache is a registered buffer, mutated in place (supported scenario)."""
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(D, 3 * D)
        self.register_buffer("cache_k", torch.zeros(1, CACHE, D))
        self.register_buffer("cache_v", torch.zeros(1, CACHE, D))

    def forward(self, x):
        q, k, v = self.w(x).chunk(3, -1)
        self.cache_k[:, :L] = k     # <-- mutate a BUFFER
        self.cache_v[:, :L] = v
        att = torch.softmax(q @ self.cache_k.transpose(-1, -2), -1) @ self.cache_v
        return x + att


class AttnEagerOut(nn.Module):
    """cache arrives as an eager input; NOT mutated -- new k/v are returned."""
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(D, 3 * D)

    def forward(self, x, cache_k, cache_v):
        q, k, v = self.w(x).chunk(3, -1)
        att = torch.softmax(q @ cache_k.transpose(-1, -2), -1) @ cache_v
        # functional: caller owns the update, graph never mutates an input
        return x + att, k, v


def captured_info(fn):
    """Best-effort: did inductor actually build a CUDAGraph for this callable?"""
    try:
        from torch._inductor.cudagraph_trees import get_manager
        m = get_manager(0)
        return dict(manager=True,
                    ids_exist=len(getattr(m, "id_to_stack", {})) if m else 0)
    except Exception as e:
        return dict(err=str(e)[:120])


def bench(mod, args_fn, n=12, warm=4):
    for _ in range(warm):
        out = mod(*args_fn())
    torch.cuda.synchronize()
    lat = []
    outs = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = mod(*args_fn())
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        outs.append(out[0] if isinstance(out, tuple) else out)
    return lat, outs


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print(f"torch {torch.__version__}  expandable_segments="
          f"{os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)

    x = torch.randn(1, L, D, device=dev)
    # Shared init so all arms see identical weights
    base = AttnEagerIn().to(dev).eval()
    with torch.no_grad():
        base.w.weight.normal_(); base.w.bias.zero_()

    ref_k = torch.zeros(1, CACHE, D, device=dev)
    ref_v = torch.zeros(1, CACHE, D, device=dev)
    with torch.no_grad():
        ref = base(x, ref_k, ref_v).clone()

    results = {}
    for arm in ("eager_in", "buffer", "eager_out"):
        print(f"\n===== arm={arm} =====", flush=True)
        torch._dynamo.reset()
        torch.cuda.empty_cache()
        if arm == "eager_in":
            m = AttnEagerIn().to(dev).eval()
            m.load_state_dict(base.state_dict())
            ck = torch.zeros(1, CACHE, D, device=dev)
            cv = torch.zeros(1, CACHE, D, device=dev)
            args_fn = lambda: (x, ck, cv)
        elif arm == "buffer":
            m = AttnBuffer().to(dev).eval()
            with torch.no_grad():
                m.w.weight.copy_(base.w.weight); m.w.bias.copy_(base.w.bias)
            args_fn = lambda: (x,)
        else:
            m = AttnEagerOut().to(dev).eval()
            m.load_state_dict(base.state_dict(), strict=False)
            with torch.no_grad():
                m.w.weight.copy_(base.w.weight); m.w.bias.copy_(base.w.bias)
            ck = torch.zeros(1, CACHE, D, device=dev)
            cv = torch.zeros(1, CACHE, D, device=dev)
            args_fn = lambda: (x, ck, cv)

        cm = torch.compile(m, mode="reduce-overhead", fullgraph=False)
        try:
            lat, outs = bench(cm, args_fn)
            got = outs[-1]
            err = (got - ref).abs().max().item()
            med = statistics.median(lat) * 1000
            print(f"[synth] {arm:10s} median {med:7.3f} ms   "
                  f"max|Δ vs ref| {err:.3e}", flush=True)
            results[arm] = dict(median_ms=med, max_err=err, ok=True)
        except Exception as e:
            print(f"[synth] {arm:10s} FAILED: {type(e).__name__}: "
                  f"{str(e)[:200]}", flush=True)
            results[arm] = dict(ok=False, err=f"{type(e).__name__}: {str(e)[:200]}")

        info = captured_info(cm)
        print(f"[synth]   cudagraph manager: {info}", flush=True)
        results[arm]["cudagraph"] = info

    print("\n[synth] ===== verdict =====")
    print("[synth] 'skipping cudagraphs' messages should appear above for "
          "eager_in. If 'buffer' shows none and is faster, the mechanism is "
          "confirmed: eager-input mutation is the blocker, buffer mutation is "
          "supportable.")
    import json
    json.dump(results, open("output/cg/synth.json", "w"), indent=1, default=str)
    print("[synth] wrote output/cg/synth.json")


if __name__ == "__main__":
    os.makedirs("output/cg", exist_ok=True)
    main()
