#!/usr/bin/env python
"""RoPE-0: attribute the 97.1 ms / 240 calls across the six buckets.

240 calls per chunk = 30 layers x 4 forwards per chunk x 2 (q and k).

The implementation (wan/modules/model_fast.py:100) is:

    freqs = freqs.split([c - 2*(c//3), c//3, c//3], dim=1)          # (1) split
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):             # (2) tolist
        x_i = torch.view_as_complex(
            x[i,:seq_len].to(torch.float64).reshape(seq_len,n,-1,2))# (3) fp64 cast
        freqs_i = torch.cat([...expand...], dim=-1).reshape(...)     # (4) materialise
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)           # (5) complex mul
        x_i = torch.cat([x_i, x[i, seq_len:]])                       # (6) cat
    return torch.stack(output).type_as(x)                            # (7) stack + cast

Primary suspect: (3)+(5). Consumer Blackwell runs fp64 at roughly 1/64 of fp32,
and `freqs` is complex64, so `x_i * freqs_i` is evaluated as complex128.

This measures each bucket separately with CUDA events, and then measures an
otherwise-identical fp32 variant to price the fp64 decision on its own.
"""
import time

import torch

dev = "cuda"
torch.manual_seed(0)

# --- real geometry ----------------------------------------------------------
# area 512x320 -> h=304, w=528 -> latent 38x66 -> patch 2x2 -> 19x33 = 627/frame
# chunk_size 3 -> f=3, seq_len = 3*19*33 = 1881 (= M)
N_HEADS = 12
HEAD_DIM = 128
F, H, W = 3, 19, 33
SEQ = F * H * W
print(f"geometry: f={F} h={H} w={W}  seq_len={SEQ}  heads={N_HEADS} "
      f"head_dim={HEAD_DIM}  x={[1, SEQ, N_HEADS, HEAD_DIM]}")
print(f"x bytes fp32 = {SEQ*N_HEADS*HEAD_DIM*4/1e6:.1f} MB, "
      f"fp64 = {SEQ*N_HEADS*HEAD_DIM*8/1e6:.1f} MB")
print()

x = torch.randn(1, SEQ, N_HEADS, HEAD_DIM, dtype=torch.float32, device=dev)
grid_sizes = torch.tensor([[F, H, W]], device=dev)
c = HEAD_DIM // 2
# rope_params returns complex64 of shape (max_seq_len, dim/2)
max_len = 1024
theta = 10000.0
fr = torch.outer(
    torch.arange(max_len, device=dev, dtype=torch.float64),
    1.0 / torch.pow(theta,
                    torch.arange(0, HEAD_DIM, 2, device=dev, dtype=torch.float64)
                    .div(HEAD_DIM)))
freqs_all = torch.polar(torch.ones_like(fr), fr).to(torch.complex64)
print(f"freqs: shape={list(freqs_all.shape)} dtype={freqs_all.dtype} "
      f"(complex64 -> x_i*freqs_i promotes to complex128)")
print()


def ev():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def ms(a, b):
    torch.cuda.synchronize()
    return a.elapsed_time(b)


def bench(fn, n=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000.0     # us


def rope_full(xx, grid, freqs, start_frame, dt=torch.float64):
    """The production function, parameterised on the intermediate dtype."""
    n, cc = xx.size(2), xx.size(3) // 2
    freqs = freqs.split([cc - 2 * (cc // 3), cc // 3, cc // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            xx[i, :seq_len].to(dt).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1)
            .expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, xx[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(xx)


print("=== bucket attribution (per single call, us) ===")
t_full = bench(lambda: rope_full(x, grid_sizes, freqs_all, 0))
print(f"  full call                       {t_full:8.1f} us")
print()

# time each bucket in isolation at the real size
n_, cc_ = N_HEADS, HEAD_DIM // 2
frs = freqs_all.split([cc_ - 2 * (cc_ // 3), cc_ // 3, cc_ // 3], dim=1)
t_split = bench(lambda: freqs_all.split([cc_ - 2 * (cc_ // 3), cc_ // 3,
                                         cc_ // 3], dim=1))
print(f"  (1) freqs.split                 {t_split:8.1f} us")

t_tolist = bench(lambda: grid_sizes.tolist())
print(f"  (2) grid_sizes.tolist()         {t_tolist:8.1f} us  <- CPU sync")

t_cast64 = bench(lambda: x[0, :SEQ].to(torch.float64))
print(f"  (3) x.to(float64)               {t_cast64:8.1f} us  <- "
      f"{SEQ*N_HEADS*HEAD_DIM*8/1e6:.1f} MB out")


def mk_freqs_i():
    return torch.cat([
        frs[0][0:F].view(F, 1, 1, -1).expand(F, H, W, -1),
        frs[1][:H].view(1, H, 1, -1).expand(F, H, W, -1),
        frs[2][:W].view(1, 1, W, -1).expand(F, H, W, -1)
    ], dim=-1).reshape(SEQ, 1, -1)


t_mat = bench(mk_freqs_i)
fi = mk_freqs_i()
print(f"  (4) freqs_i materialise         {t_mat:8.1f} us  <- "
      f"expand + cat forces real alloc {SEQ*cc_*8/1e6:.2f} MB")

x64 = torch.view_as_complex(
    x[0, :SEQ].to(torch.float64).reshape(SEQ, N_HEADS, -1, 2))
fi64 = fi
t_mul64 = bench(lambda: x64 * fi64)
print(f"  (5) complex mul (complex128)    {t_mul64:8.1f} us  <- PRIMARY SUSPECT")

x32 = torch.view_as_complex(x[0, :SEQ].reshape(SEQ, N_HEADS, -1, 2))
fi32 = fi.to(torch.complex64)
t_mul32 = bench(lambda: x32 * fi32)
print(f"  (5f) same mul in complex64      {t_mul32:8.1f} us  <- for contrast")
print(f"       fp64/fp32 mul ratio        {t_mul64/t_mul32:8.1f}x")

t_lay = bench(lambda: torch.view_as_real(x64 * fi64).flatten(2))
print(f"  (6) view_as_real+flatten        {t_lay:8.1f} us")

prod = torch.view_as_real(x64 * fi64).flatten(2)
t_cat = bench(lambda: torch.cat([prod, x[0, SEQ:]]))
print(f"  (7) cat with the tail           {t_cat:8.1f} us  (tail is empty here)")

t_stack = bench(lambda: torch.stack([prod]).type_as(x))
print(f"  (8) stack + type_as             {t_stack:8.1f} us")

print()
print("=== fp32 variant of the whole function (the fp64 decision priced) ===")
t_fp32 = bench(lambda: rope_full(x, grid_sizes, freqs_all, 0, dt=torch.float32))
# correctness of the fp32 variant against the fp64 reference
ref = rope_full(x, grid_sizes, freqs_all, 0, dt=torch.float64)
alt = rope_full(x, grid_sizes, freqs_all, 0, dt=torch.float32)
d = (ref.float() - alt.float()).abs().max().item()
print(f"  full call fp64 {t_full:8.1f} us   fp32 {t_fp32:8.1f} us   "
      f"delta {t_fp32-t_full:+8.1f} us  ({(t_fp32-t_full)/t_full*100:+6.1f}%)")
print(f"  fp32 vs fp64 max|diff| = {d:.3e}   bit-identical="
      f"{torch.equal(ref, alt)}")

print()
print("=== scaled to 240 calls per chunk ===")
for name, t in (("full (production, fp64)", t_full),
                ("full (fp32 variant)", t_fp32),
                ("(5) complex128 mul only", t_mul64),
                ("(5f) complex64 mul only", t_mul32),
                ("(4) freqs_i materialise", t_mat),
                ("(3) fp64 cast", t_cast64)):
    print(f"  {name:<28} {t*240/1000:8.1f} ms/chunk")
print()
print(f"  reported RoPE total: 97.1 ms/chunk")
print(f"  measurement here    : {t_full*240/1000:.1f} ms/chunk "
      f"(isolation, no other work interleaved)")
