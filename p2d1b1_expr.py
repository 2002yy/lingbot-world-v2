"""
P2d-1b1 step 0 -- validate the model's ACTUAL expressions, not the r3 proxy.

r3's matrix passed `a=n1, b=(1+e1), c=e0` to a helper computing `a + b*c`, i.e.
it tested  n1 + (1+e1)*e0 .  The model computes  n1*(1+e1) + e0 .  Those are
different expressions, so r3's 40/40 for mod1/mod2 says nothing about the line we
actually intend to replace.  Same story for cam.

This script tests the real lines with the real dtypes, as pure tensors, so the
result is decisive without loading the model.

  model 349:  self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)
  model 353:  x = x + y * e[2].squeeze(2)
  model 382:  x = (1.0 + cam_scale) * x + cam_shift
  model 391:  self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)
  model 393:  x = x + y * e[5].squeeze(2)
"""
import torch

torch.manual_seed(0)
dev = "cuda"
D, L = 1536, 1881


def rnd(dtype):
    return torch.randn(L, D, dtype=dtype, device=dev)


def rep(name, ref, cand):
    eq = torch.equal(ref, cand)
    d = (ref.float() - cand.float()).abs().max().item()
    print(f"  {name:<6} {'EXACT' if eq else 'DIFF '}  max|d|={d:.3e}  "
          f"dtype={ref.dtype}")
    return eq


print("=== the model's ACTUAL expressions, real dtypes ===")
print("(dtypes mirror the hooks: norm outputs bf16, modulation e[] f32,")
print(" attn/ffn outputs f32 under autocast, cam_scale/shift bf16)")

# --- mod1 / model line 349 -------------------------------------------------
n1 = rnd(torch.bfloat16)          # norm1(x) output, bf16
e0, e1 = rnd(torch.float32), rnd(torch.float32)
ref = n1.float() * (1 + e1) + e0
cand = torch.addcmul(e0, n1.float(), 1 + e1)
m1 = rep("mod1", ref, cand)

# --- mod2 / model line 391 -------------------------------------------------
n2 = rnd(torch.bfloat16)          # norm2(x) output, bf16
e3, e4 = rnd(torch.float32), rnd(torch.float32)
ref2 = n2.float() * (1 + e4) + e3
cand2 = torch.addcmul(e3, n2.float(), 1 + e4)
m2 = rep("mod2", ref2, cand2)

# --- resA / model line 353 -------------------------------------------------
# x here is the mod1 output (f32); y is the self-attn output (f32)
x = ref.clone()
y = rnd(torch.float32)
e2 = rnd(torch.float32)
refA = x + y * e2
candA = torch.addcmul(x, y, e2)
rA = rep("resA", refA, candA)

# --- resB / model line 393 -------------------------------------------------
x2 = rnd(torch.float32)
y2 = rnd(torch.float32)
e5 = rnd(torch.float32)
refB = x2 + y2 * e5
candB = torch.addcmul(x2, y2, e5)
rB = rep("resB", refB, candB)

# --- cam / model line 382 --------------------------------------------------
# cam_scale / cam_shift come out of Linear layers in bf16; x is f32 here.
cs = rnd(torch.bfloat16)
ct = rnd(torch.bfloat16)
xc = rnd(torch.float32)
refC = (1.0 + cs) * xc + ct
candC = torch.addcmul(ct, xc, 1.0 + cs)
rC = rep("cam", refC, candC)

# --- what r3 actually tested for cam, for contrast -------------------------
print()
print("=== contrast: what r3's cam call actually measured ===")
refW = xc + (1.0 + cs) * ct          # a + b*c with a=xc, b=1+cs, c=ct
candW = torch.addcmul(xc, 1.0 + cs, ct)
dW = (refW.float() - candW.float()).abs().max().item()
print(f"  r3-cam  {'EXACT' if torch.equal(refW, candW) else 'DIFF '}  "
      f"max|d|={dW:.3e}   <- matches the 1.953e-03 r3 reported")

print()
print("=== verdict ===")
for nm, ok in (("mod1", m1), ("mod2", m2), ("resA", rA), ("resB", rB),
               ("cam", rC)):
    print(f"  {nm:<6} {'repro-safe candidate' if ok else 'NOT exact -> close'}")
