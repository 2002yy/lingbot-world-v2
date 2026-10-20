#!/usr/bin/env python
"""RoPE-0b: clean A/B deltas instead of isolated bucket timings.

The first pass summed to 634.9 us against a 488.7 us full call, because each
bucket was timed in isolation and carried its own launch overhead. This measures
variants of the WHOLE function instead, so each delta is attributable.

Variants:
  ref      the current implementation, verbatim
  cached   tolist() hoisted and the freqs_i table precomputed per (start_frame,
           f, h, w) -- pure CPU/setup, no arithmetic change, so it must stay
           bit-exact
  fp32     the fp64 intermediate replaced by fp32 -- NOT expected to be exact,
           measured only to price the precision decision
"""
import torch

dev = "cuda"
torch.manual_seed(0)
N_HEADS, HEAD_DIM = 12, 128
F, H, W = 3, 19, 33
SEQ = F * H * W
c = HEAD_DIM // 2

x = torch.randn(1, SEQ, N_HEADS, HEAD_DIM, dtype=torch.float32, device=dev)
grid_sizes = torch.tensor([[F, H, W]], device=dev)
max_len = 1024
fr = torch.outer(
    torch.arange(max_len, device=dev, dtype=torch.float64),
    1.0 / torch.pow(10000.0,
                    torch.arange(0, HEAD_DIM, 2, device=dev, dtype=torch.float64)
                    .div(HEAD_DIM)))
freqs_all = torch.polar(torch.ones_like(fr), fr).to(torch.complex64)


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


# ---------------------------------------------------------------- reference
def rope_ref(xx, grid, freqs, start_frame, dt=torch.float64):
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


# ------------------------------------------------------------------- cached
_freqs_split_cache = {}


def _freqs_split(freqs, cc):
    key = id(freqs)
    got = _freqs_split_cache.get(key)
    if got is None:
        got = freqs.split([cc - 2 * (cc // 3), cc // 3, cc // 3], dim=1)
        _freqs_split_cache[key] = got
    return got


_table_cache = {}


def _freqs_i_table(freqs, cc, start_frame, f, h, w):
    key = (id(freqs), cc, start_frame, f, h, w)
    got = _table_cache.get(key)
    if got is None:
        fs = _freqs_split(freqs, cc)
        got = torch.cat([
            fs[0][start_frame:start_frame + f].view(f, 1, 1, -1)
            .expand(f, h, w, -1),
            fs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            fs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).contiguous()
        _table_cache[key] = got
    return got


def rope_cached(xx, grid, freqs, start_frame, dt=torch.float64):
    n, cc = xx.size(2), xx.size(3) // 2
    output = []
    for i, (f, h, w) in enumerate(grid.tolist()):     # tolist still a sync here
        seq_len = f * h * w
        x_i = torch.view_as_complex(
            xx[i, :seq_len].to(dt).reshape(seq_len, n, -1, 2))
        freqs_i = _freqs_i_table(freqs, cc, start_frame, f, h, w)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, xx[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(xx)


# ------------------------------------------------------------------ geometry
GEO = tuple(grid_sizes.tolist()[0])       # the tolist() result, computed once


def rope_geo(xx, grid, freqs, start_frame, dt=torch.float64):
    """tolist hoisted to a module constant -- the sync is gone entirely."""
    n, cc = xx.size(2), xx.size(3) // 2
    f, h, w = GEO
    seq_len = f * h * w
    x_i = torch.view_as_complex(
        xx[0, :seq_len].to(dt).reshape(seq_len, n, -1, 2))
    freqs_i = _freqs_i_table(freqs, cc, start_frame, f, h, w)
    x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
    if xx.size(1) > seq_len:
        x_i = torch.cat([x_i, xx[0, seq_len:]])
    return x_i.unsqueeze(0).type_as(xx)


def check(name, ref, alt):
    eq = torch.equal(ref, alt)
    d = (ref.float() - alt.float()).abs().max().item()
    print(f"  {name:<10} bit-identical={str(eq):<5}  max|diff|={d:.3e}")
    return eq


print("=== correctness ===")
R = rope_ref(x, grid_sizes, freqs_all, 0)
C = rope_cached(x, grid_sizes, freqs_all, 0)
G = rope_geo(x, grid_sizes, freqs_all, 0)
P = rope_ref(x, grid_sizes, freqs_all, 0, dt=torch.float32)
check("ref", R, R)
check("cached", R, C)
check("geo", R, G)
check("fp32", R, P)
print()

print("=== full-call timing, A/B deltas (us per call, 300 iters) ===")
t_ref = bench(lambda: rope_ref(x, grid_sizes, freqs_all, 0))
t_cac = bench(lambda: rope_cached(x, grid_sizes, freqs_all, 0))
t_geo = bench(lambda: rope_geo(x, grid_sizes, freqs_all, 0))
t_f32 = bench(lambda: rope_ref(x, grid_sizes, freqs_all, 0, dt=torch.float32))
for nm, t in (("ref", t_ref), ("cached", t_cac), ("geo", t_geo),
              ("fp32", t_f32)):
    print(f"  {nm:<8} {t:8.1f} us   delta vs ref {t-t_ref:+8.1f} us "
          f"({(t-t_ref)/t_ref*100:+6.1f}%)")
print()

# --- also: what the second and later calls of a chunk look like -------------
# within a chunk start_frame is constant, so cached/geo hit their caches
print("=== steady-state (cache warm, as in forwards 2..4 of a chunk) ===")
_c = {}
for nm, fn in (("cached", lambda: rope_cached(x, grid_sizes, freqs_all, 0)),
               ("geo", lambda: rope_geo(x, grid_sizes, freqs_all, 0))):
    fn()
t_cac2 = bench(lambda: rope_cached(x, grid_sizes, freqs_all, 0))
t_geo2 = bench(lambda: rope_geo(x, grid_sizes, freqs_all, 0))
print(f"  cached   {t_cac2:8.1f} us   delta vs ref {t_cac2-t_ref:+8.1f} us "
      f"({(t_cac2-t_ref)/t_ref*100:+6.1f}%)")
print(f"  geo      {t_geo2:8.1f} us   delta vs ref {t_geo2-t_ref:+8.1f} us "
      f"({(t_geo2-t_ref)/t_ref*100:+6.1f}%)")
print()

print("=== per chunk (240 calls) ===")
for nm, t in (("ref (production)", t_ref), ("cached", t_cac2),
              ("geo", t_geo2), ("fp32", t_f32)):
    print(f"  {nm:<18} {t*240/1000:8.1f} ms/chunk   "
          f"save {(t_ref-t)*240/1000:+8.1f} ms")
print()
print(f"  reported RoPE total     97.1 ms/chunk")
print(f"  this measurement, ref   {t_ref*240/1000:.1f} ms/chunk")
print()
print("=== the gate: >= 30 ms/chunk exactly-eliminable? ===")
print(f"  cached+geo (bit-exact)  {(t_ref-t_geo2)*240/1000:8.1f} ms/chunk")
print(f"  fp32 (NOT exact)        {(t_ref-t_f32)*240/1000:8.1f} ms/chunk")
