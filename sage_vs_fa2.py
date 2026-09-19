#!/usr/bin/env python
"""S3-A step 2: SageAttention 2.2 vs the ACTUAL production kernel (flash_attn).

WHY THIS IS THE DECISIVE MEASUREMENT
------------------------------------
The first microbench compared SageAttention against
`torch.nn.functional.scaled_dot_product_attention`. But LingBot's production
attention does not call SDPA -- `wan/modules/attention.py::attention()` has
`if FLASH_ATTN_2_AVAILABLE ... return flash_attention(...)`, and we have
flash-attn 2.8.3 installed. So the incumbent is FA2, not SDPA.

If FA2 is already faster than SDPA at these shapes, SageAttention's real
advantage could be much smaller -- or negative. That number decides whether the
whole SageAttention line is worth it, so we measure it directly, with the same
tensors, at the exact shapes the profiler observed:

    self-attn  : Lq=627, Lkv=3762, H=12, D=128   (360 calls / 8 chunks)
    cross-attn : Lq=627, Lkv= 512, H=12, D=128   (960 calls / 8 chunks)
    + the intermediate window sizes 1254/1881/2508/3135 and the 627 square

Budget model
------------
For each shape we report per-call ms for FA2 and for Sage, then convert to
ms/chunk using the profiler's real call counts, so we can state the ceiling
speedup of the whole chunk BEFORE integrating anything.
"""
import collections
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/ai/lerobot"))  # noqa: F401 (harmless)
from wan.modules.attention import attention as prod_attention, flash_attention  # noqa: E402

torch.manual_seed(0)


def bench(fn, n=100, warm=20):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000, out


def main():
    from sageattention import sageattn
    print(f"torch {torch.__version__}  cap {torch.cuda.get_device_capability()}",
          flush=True)
    dev, dtype = "cuda", torch.bfloat16

    # (Lq, Lkv, H, D, calls_per_chunk)  -- call counts from attn_profile.py
    shapes = [
        (627, 3762, 12, 128, 45, "self  steady(6-frame window)"),
        (627, 3135, 12, 128, 15, "self  window rolloff"),
        (627, 2508, 12, 128, 15, "self  window rolloff"),
        (627, 1881, 12, 128, 15, "self  window rolloff"),
        (627, 1254, 12, 128, 15, "self  window rolloff"),
        (627, 627, 12, 128, 15, "self  cache filling"),
        (627, 512, 12, 128, 120, "cross (text context)"),
    ]
    print("\n{:<30} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}".format(
        "shape / role", "FA2 ms", "SDPA ms", "Sage ms", "Sage/FA2",
        "cos vs FA2", "Δms/chunk"))
    print("-" * 100)
    rows = []
    for Lq, Lkv, H, D, calls, role in shapes:
        q = torch.randn(1, Lq, H, D, device=dev, dtype=dtype)
        k = torch.randn(1, Lkv, H, D, device=dev, dtype=dtype)
        v = torch.randn(1, Lkv, H, D, device=dev, dtype=dtype)

        def fa2():
            return prod_attention(q, k, v)

        def sdpa():
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=False).transpose(1, 2)

        def sage():
            return sageattn(q, k, v, tensor_layout="NHD",
                            is_causal=False, smooth_k=True)

        res = {}
        for name, fn in (("fa2", fa2), ("sdpa", sdpa), ("sage", sage)):
            try:
                res[name] = bench(fn)
            except Exception as e:
                res[name] = (float("nan"), None)
                print(f"  {name} failed at {Lq}x{Lkv}: "
                      f"{type(e).__name__}: {str(e)[:60]}")

        t_fa2, o_fa2 = res["fa2"]
        t_sdpa, _ = res["sdpa"]
        t_sage, o_sage = res["sage"]
        cos = float("nan")
        if o_fa2 is not None and o_sage is not None and \
                o_fa2.shape == o_sage.shape:
            cos = torch.nn.functional.cosine_similarity(
                o_sage.float().reshape(-1), o_fa2.float().reshape(-1),
                dim=0).item()
        ratio = t_sage / t_fa2 if t_fa2 else float("nan")
        delta_chunk = (t_sage - t_fa2) * calls   # negative = Sage wins
        print("{:<30} {:>9.4f} {:>9.4f} {:>9.4f} {:>8.2f}x {:>9.5f} {:>+10.2f}"
              .format(f"{Lq}x{Lkv}", t_fa2, t_sdpa, t_sage, ratio, cos,
                      delta_chunk), flush=True)
        rows.append(dict(role=role, Lq=Lq, Lkv=Lkv, H=H, D=D, calls=calls,
                         fa2_ms=t_fa2, sdpa_ms=t_sdpa, sage_ms=t_sage,
                         sage_over_fa2=ratio, cosine_vs_fa2=cos,
                         delta_ms_per_chunk=delta_chunk))

    tot = sum(r["delta_ms_per_chunk"] for r in rows)
    # steady-state subset: only the shapes that occur once the window is full
    steady = [r for r in rows if r["Lkv"] >= 3762 or r["Lkv"] == 512]
    tot_steady = sum(r["delta_ms_per_chunk"] for r in steady)
    print("\n[mb] Σ Δms/chunk (all listed shapes) = {:+.2f} ms".format(tot))
    print("[mb] Σ Δms/chunk (steady state only) = {:+.2f} ms".format(tot_steady))
    print("[mb] chunk baseline ≈ 827 ms → "
          "ceiling = {:+.2f}%".format(100 * tot_steady / 827))
    json.dump(dict(rows=rows, total_delta_all=tot,
                   total_delta_steady=tot_steady,
                   chunk_baseline_ms=827.0,
                   pct_steady=100 * tot_steady / 827),
              open("output/attnprof/sage_vs_fa2.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
