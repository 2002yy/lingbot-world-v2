#!/usr/bin/env python
"""S2 (fixed): SageAttention 2.2 microbench on RTX 5060 / sm_120.

Layout contract (this is what the first run got wrong):
  sageattn(..., tensor_layout="HND") expects q,k,v as [B, H, N, D]
  sageattn(..., tensor_layout="NHD") expects q,k,v as [B, N, H, D]
Passing [B, N, H, D] while claiming HND silently reinterprets N as heads, which
is why the first run reported cosine ~0.008 (garbage, not quantisation error).

LingBot causal-fast self-attention is: a short block of query tokens against a
local KV window of up to local_attn_size frames, non-causal within the window.
We benchmark both that pattern and a few plain square shapes.

We report error metrics honestly: SageAttention quantises QK^T, so it is NOT
bit-exact and cosine will not be 1.0.
"""
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
    print(f"torch {torch.__version__}  cap {torch.cuda.get_device_capability()}",
          flush=True)
    from sageattention import sageattn
    import sageattention
    print(f"sageattention {getattr(sageattention, '__version__', '?')}", flush=True)

    dev = "cuda"
    dtype = torch.bfloat16
    # (Lq, Lkv, H, D)  -- all in NHD semantics
    shapes = [
        (627, 627, 12, 128),      # one frame worth of latent tokens, square
        (627, 3762, 12, 128),     # production-ish: 1 frame query vs 6-frame window
        (1254, 3762, 12, 128),    # 2 frames query vs 6-frame window
        (2048, 2048, 12, 128),    # large square
        (4096, 4096, 12, 128),    # stress
    ]
    print("\n{:<26} {:>9} {:>9} {:>9} {:>10} {:>10} {:>8}".format(
        "shape (Lq,Lkv,H,D)", "sdpa ms", "sage ms", "speedup",
        "max|d|", "mean|d|", "cosine"))
    print("-" * 92)
    for Lq, Lkv, H, D in shapes:
        # NHD layout for the model's mental model
        q = torch.randn(1, Lq, H, D, device=dev, dtype=dtype)
        k = torch.randn(1, Lkv, H, D, device=dev, dtype=dtype)
        v = torch.randn(1, Lkv, H, D, device=dev, dtype=dtype)

        # SDPA wants [B, H, N, D]
        qh, kh, vh = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        def sdpa():
            return torch.nn.functional.scaled_dot_product_attention(
                qh, kh, vh, is_causal=False)

        # sageattn HND wants [B, H, N, D] -- identical to qh/kh/vh
        def sage_hnd():
            return sageattn(qh, kh, vh, tensor_layout="HND",
                            is_causal=False, smooth_k=True)

        # sageattn NHD wants [B, N, H, D] -- the raw tensors
        def sage_nhd():
            return sageattn(q, k, v, tensor_layout="NHD",
                            is_causal=False, smooth_k=True)

        try:
            t_sdpa, o_sdpa = bench(sdpa)
        except Exception as e:
            print(f"  sdpa failed: {type(e).__name__}: {str(e)[:70]}")
            continue
        for label, fn in (("HND", sage_hnd), ("NHD", sage_nhd)):
            try:
                t_sage, o_sage = bench(fn)
            except Exception as e:
                print("{:<26} {:>9.3f}  sage[{}] FAILED: {}".format(
                    f"{Lq},{Lkv},{H},{D}", t_sdpa, label,
                    f"{type(e).__name__}: {str(e)[:50]}"))
                continue
            # both return [B, H, N, D] for HND and [B, N, H, D] for NHD
            a = o_sage.float()
            b = o_sdpa.float()
            if a.shape != b.shape:
                a = a.transpose(1, 2).contiguous() if a.shape[1] == H else a
            if a.shape != b.shape:
                print(f"    shape mismatch sage={tuple(o_sage.shape)} "
                      f"sdpa={tuple(o_sdpa.shape)}")
                continue
            d = (a - b)
            print("{:<26} {:>9.3f} {:>9.3f} {:>8.2f}x {:>10.3e} {:>10.3e} {:>8.5f}  [{}]"
                  .format(f"{Lq},{Lkv},{H},{D}", t_sdpa, t_sage,
                          t_sdpa / t_sage, d.abs().max().item(),
                          d.abs().mean().item(),
                          torch.nn.functional.cosine_similarity(
                              a.reshape(-1), b.reshape(-1), dim=0).item(),
                          label), flush=True)


if __name__ == "__main__":
    main()
