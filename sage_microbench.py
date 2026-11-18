#!/usr/bin/env python
"""S2: SageAttention 2.2 microbench on the RTX 5060 (sm_120).

Verifies the freshly built extension actually loads and runs, then measures it
against PyTorch SDPA at the real shapes LingBot's causal-fast attention uses.

LingBot self-attention shape (causal_fast, local_attn_size=6):
  q: [B, L, H, D] with L = frame_seqlen (one frame of latent tokens)
  k/v: the local window, up to local_attn_size * frame_seqlen tokens
i.e. a small number of query tokens against a longer KV window, causal.

We sweep a few (L, KV) pairs rather than only the exact production shape, so we
can see how the speedup scales, and we report error metrics (max/mean abs,
cosine) because SageAttention quantises -- it is NOT bit-exact by construction.

Run:
  LINGBOT_FP8=1 python sage_microbench.py
"""
import os
import statistics
import time

import torch

torch.manual_seed(0)


def bench(fn, n=50, warm=10):
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
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"cap {torch.cuda.get_device_capability()}", flush=True)
    try:
        import sageattention
        print(f"sageattention {getattr(sageattention, '__version__', '?')} "
              f"from {sageattention.__file__}", flush=True)
        from sageattention import sageattn
    except Exception as e:
        print(f"IMPORT FAILED: {type(e).__name__}: {e}")
        return
    # which kernels got built?
    try:
        import sageattention._qattn_sm80  # noqa: F401
        print("  _qattn_sm80: loaded")
    except Exception as e:
        print(f"  _qattn_sm80: {type(e).__name__}: {str(e)[:80]}")
    try:
        import sageattention._qattn_sm89  # noqa: F401
        print("  _qattn_sm89: loaded")
    except Exception as e:
        print(f"  _qattn_sm89: {type(e).__name__}: {str(e)[:80]}")
    try:
        import sageattention._fused  # noqa: F401
        print("  _fused: loaded")
    except Exception as e:
        print(f"  _fused: {type(e).__name__}: {str(e)[:80]}")

    dev = "cuda"
    dtype = torch.bfloat16
    # (seq_len_q, seq_len_kv, heads, head_dim)
    shapes = [
        (1026, 1026, 12, 128),   # roughly production: frame_seqlen ~38*66/4
        (1026, 6156, 12, 128),   # local window = 6 frames
        (627, 3762, 12, 128),
        (2048, 2048, 12, 128),
    ]
    print("\n{:<28} {:>10} {:>10} {:>8} {:>10} {:>10} {:>9}".format(
        "shape (Lq, Lkv, H, D)", "sdpa ms", "sage ms", "speedup",
        "max|d|", "mean|d|", "cosine"))
    print("-" * 96)
    for Lq, Lkv, H, D in shapes:
        q = torch.randn(1, Lq, H, D, device=dev, dtype=dtype)
        k = torch.randn(1, Lkv, H, D, device=dev, dtype=dtype)
        v = torch.randn(1, Lkv, H, D, device=dev, dtype=dtype)

        def sdpa():
            # [B, H, L, D]
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                is_causal=False).transpose(1, 2)

        def sage():
            return sageattn(q, k, v, tensor_layout="HND",
                            is_causal=False, smooth_k=True)

        try:
            t_sdpa, o_sdpa = bench(sdpa)
        except Exception as e:
            print(f"  sdpa failed: {type(e).__name__}: {str(e)[:60]}")
            continue
        try:
            t_sage, o_sage = bench(sage)
        except Exception as e:
            print(f"{str((Lq,Lkv,H,D)):<28} {t_sdpa:10.3f}  SAGE FAILED: "
                  f"{type(e).__name__}: {str(e)[:60]}")
            continue

        d = (o_sage.float() - o_sdpa.float())
        maxd = d.abs().max().item()
        meand = d.abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(
            o_sage.float().reshape(-1), o_sdpa.float().reshape(-1), dim=0).item()
        print("{:<28} {:>10.3f} {:>10.3f} {:>7.2f}x {:>10.3e} {:>10.3e} {:>9.5f}"
              .format(f"{Lq},{Lkv},{H},{D}", t_sdpa, t_sage, t_sdpa / t_sage,
                      maxd, meand, cos), flush=True)


if __name__ == "__main__":
    main()
