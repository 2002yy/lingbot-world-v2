#!/usr/bin/env python
"""RoPE-0c: correct the attribution. grid_sizes is a CPU tensor.

p2d_rope0.py/0b.py put grid_sizes on CUDA, which invented a hard sync that does
not exist. The real construction is:

    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long)
                              for u in x])                       # model_fast.py:760

`torch.tensor(tuple)` gives a CPU tensor, so `.tolist()` is pure CPU and never
synchronises. That is why the microbenchmark predicted -263 us/call (-63 ms per
chunk) while the 21-chunk rollout measured only -0.11%.

This re-measures with the correct device, and also measures the CPU-side cost of
the two caches on their own, so the honest split is visible:

  - what the caches remove that is real and on the critical path
  - what is CPU-only work that simply overlaps with the GPU
"""
import torch

dev = "cuda"
torch.manual_seed(0)
N_HEADS, HEAD_DIM = 12, 128
F, H, W = 3, 19, 33
SEQ = F * H * W
c = HEAD_DIM // 2

x = torch.randn(1, SEQ, N_HEADS, HEAD_DIM, dtype=torch.float32, device=dev)
grid_gpu = torch.tensor([[F, H, W]], device=dev)          # WRONG: what 0/0b used
grid_cpu = torch.tensor([[F, H, W]], dtype=torch.long)    # RIGHT: production
max_len = 1024
fr = torch.outer(
    torch.arange(max_len, device=dev, dtype=torch.float64),
    1.0 / torch.pow(10000.0,
                    torch.arange(0, HEAD_DIM, 2, device=dev, dtype=torch.float64)
                    .div(HEAD_DIM)))
freqs_all = torch.polar(torch.ones_like(fr), fr).to(torch.complex64)

_TC = {}


def _table(freqs, cc, start_frame, f, h, w, split):
    key = (id(freqs), cc, start_frame, f, h, w)
    got = _TC.get(key)
    if got is None:
        got = torch.cat([
            split[0][start_frame:start_frame + f].view(f, 1, 1, -1)
            .expand(f, h, w, -1),
            split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1)
        _TC[key] = got
    return got


def rope(xx, grid, freqs, start_frame, dt=torch.float64, cached=False):
    n, cc = xx.size(2), xx.size(3) // 2
    fs = freqs.split([cc - 2 * (cc // 3), cc // 3, cc // 3], dim=1)
    out = []
    for i, (f, h, w) in enumerate(grid.tolist()):
        sl = f * h * w
        x_i = torch.view_as_complex(
            xx[i, :sl].to(dt).reshape(sl, n, -1, 2))
        if cached:
            fi = _table(freqs, cc, start_frame, f, h, w, fs)
        else:
            fi = torch.cat([
                fs[0][start_frame:start_frame + f].view(f, 1, 1, -1)
                .expand(f, h, w, -1),
                fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(sl, 1, -1)
        x_i = torch.view_as_real(x_i * fi).flatten(2)
        x_i = torch.cat([x_i, xx[i, sl:]])
        out.append(x_i)
    return torch.stack(out).type_as(xx)


def bench(fn, n=300):
    for _ in range(30):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000.0


print("=== is grid_sizes.tolist() a sync? ===")
t_cpu = bench(lambda: grid_cpu.tolist(), n=2000)
t_gpu = bench(lambda: grid_gpu.tolist(), n=2000)
print(f"  CPU grid_sizes.tolist()  {t_cpu:8.2f} us   <- production")
print(f"  GPU grid_sizes.tolist()  {t_gpu:8.2f} us   <- what 0/0b wrongly used")
print()

print("=== full call, correct device (CPU grid_sizes) ===")
t_ref = bench(lambda: rope(x, grid_cpu, freqs_all, 0))
t_cac = bench(lambda: rope(x, grid_cpu, freqs_all, 0, cached=True))
print(f"  ref              {t_ref:8.1f} us")
print(f"  cached freqs_i   {t_cac:8.1f} us   delta {t_cac-t_ref:+8.1f} us "
      f"({(t_cac-t_ref)/t_ref*100:+6.1f}%)")
print(f"  per chunk (240)  ref {t_ref*240/1000:.1f} ms -> "
      f"cached {t_cac*240/1000:.1f} ms   save {(t_ref-t_cac)*240/1000:+.1f} ms")
print()

print("=== for contrast: the same call with grid_sizes on GPU ===")
t_g_ref = bench(lambda: rope(x, grid_gpu, freqs_all, 0))
print(f"  ref (GPU grid)   {t_g_ref:8.1f} us   "
      f"vs CPU grid {t_ref:8.1f} us   diff {t_g_ref-t_ref:+.1f} us")
print(f"  -> that diff is the fabricated sync that produced the -63 ms/chunk")
print(f"     prediction; the real end-to-end 21-chunk rollout saw -0.11%")
print()

print("=== what the freqs_i cache actually removes (CPU+GPU split) ===")
split = freqs_all.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
t_build = bench(lambda: _table(freqs_all, c, 0, F, H, W, split), n=500)
_TC.clear()
_table(freqs_all, c, 0, F, H, W, split)
t_hit = bench(lambda: _table(freqs_all, c, 0, F, H, W, split), n=2000)
print(f"  freqs_i build (miss)  {t_build:8.2f} us")
print(f"  freqs_i lookup (hit)  {t_hit:8.2f} us")
print(f"  saved per hit         {t_build-t_hit:8.2f} us")
print(f"  per chunk if 7 of 8 calls hit: {(t_build-t_hit)*240*0.875/1000:.2f} ms")
print()
print(f"=== verdict against the >=30 ms/chunk gate ===")
print(f"  exactly-eliminable, corrected: {(t_ref-t_cac)*240/1000:8.2f} ms/chunk")
print(f"  gate: 30 ms/chunk -> "
      f"{'PASS' if (t_ref-t_cac)*240/1000 >= 30 else 'FAIL'}")
