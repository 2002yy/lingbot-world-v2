#!/usr/bin/env python
"""P2d-1b1 ceiling: what can addcmul possibly buy at the real shape?

The 21-chunk rollout A/B was inconclusive: off vs off2 (identical configs) drifted
by 1.34%, larger than the 0.04% we measured for the full fusion. CUPTI is
unavailable under WSL (CUPTI_ERROR_INVALID_DEVICE), so kernel counts are not
available either.

This measures the op in isolation with CUDA events at the production shape, which
gives the ceiling: if the fused form is not faster here, it cannot be faster in
the rollout, and the conclusion is settled without needing a 1%-resolution
end-to-end timer.

Traffic argument, for reference:
  unfused  tmp = b*c  (read b,c; write tmp)  then  a + tmp  (read a,tmp; write out)
           = 3 reads + 2 writes
  addcmul  a + b*c    (read a,b,c; write out) = 3 reads + 1 write
so the fused form removes one write plus one read of an [1,1881,1536] tensor.
Whether that matters depends on whether the intermediate stays in L2 -- at
11.5 MB for fp32 it may well do.
"""
import torch

dev = "cuda"
L, D = 1881, 1536


def bench(fn, n=300):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000.0     # microseconds


print(f"shape [1, {L}, {D}]   fp32 = {L*D*4/1e6:.1f} MB per tensor")
print()

# --- the mod1 chain, exactly as the model has it ---------------------------
n1 = torch.randn(1, L, D, dtype=torch.bfloat16, device=dev)
e0 = torch.randn(1, L, D, dtype=torch.float32, device=dev)
e1 = torch.randn(1, L, D, dtype=torch.float32, device=dev)

ref = bench(lambda: n1.float() * (1 + e1) + e0)
fus = bench(lambda: torch.addcmul(e0, n1.float(), 1 + e1))
print(f"  mod1  unfused {ref:8.1f} us   addcmul {fus:8.1f} us   "
      f"delta {fus-ref:+8.1f} us  ({(fus-ref)/ref*100:+6.2f}%)")

# --- resA, the cleanest case (all fp32, no cast) ---------------------------
x = torch.randn(1, L, D, dtype=torch.float32, device=dev)
yv = torch.randn(1, L, D, dtype=torch.float32, device=dev)
e2 = torch.randn(1, L, D, dtype=torch.float32, device=dev)
ref2 = bench(lambda: x + yv * e2)
fus2 = bench(lambda: torch.addcmul(x, yv, e2))
print(f"  resA  unfused {ref2:8.1f} us   addcmul {fus2:8.1f} us   "
      f"delta {fus2-ref2:+8.1f} us  ({(fus2-ref2)/ref2*100:+6.2f}%)")

# --- the whole five-chain sequence per block, and what a block costs -------
def five_unfused():
    a = n1.float() * (1 + e1) + e0
    a = a + yv * e2
    a = (1.0 + n1) * a + e0
    a = a + yv * e2
    a = a + yv * e1
    return a


def five_fused():
    a = torch.addcmul(e0, n1.float(), 1 + e1)
    a = torch.addcmul(a, yv, e2)
    a = torch.addcmul(e0, a, 1.0 + n1)
    a = torch.addcmul(a, yv, e2)
    a = torch.addcmul(a, yv, e1)
    return a


u = bench(five_unfused, n=200)
f = bench(five_fused, n=200)
print()
print(f"  five-chain block  unfused {u:8.1f} us   fused {f:8.1f} us   "
      f"delta {f-u:+8.1f} us  ({(f-u)/u*100:+6.2f}%)")

# per chunk there are 30 blocks x 4 forwards = 120 block passes
print()
print(f"  extrapolated per chunk (120 block passes):")
print(f"    unfused {u*120/1000:8.2f} ms   fused {f*120/1000:8.2f} ms   "
      f"delta {(f-u)*120/1000:+8.2f} ms")
print(f"    as a fraction of a 1660 ms chunk: {(f-u)*120/1000/1660*100:+6.2f}%")

# --- memory bandwidth sanity: a pure copy of one tensor --------------------
src = torch.randn(1, L, D, dtype=torch.float32, device=dev)
cpy = bench(lambda: src.clone())
print()
print(f"  reference: clone of one fp32 tensor = {cpy:.1f} us  "
      f"({L*D*4*2/cpy/1e3:.0f} GB/s effective, read+write)")
