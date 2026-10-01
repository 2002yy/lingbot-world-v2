#!/usr/bin/env python
"""Capture TAEHV stage boundaries during a real decode_video call.

Directly calling the decoder children is not possible: MemBlock.forward requires a
`past` argument whose bookkeeping lives in apply_model_with_memblocks. Hooks observe
the real call instead, which is also the more faithful measurement.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

dev = "cuda"
tae = TAEHV(checkpoint_path=os.path.expanduser("~/ai/taehv/taew2_1.pth")).to(dev).eval()
dec = tae.decoder

print("=" * 80)
print("  TAEHV decoder geometry (captured from a real decode)")
print("=" * 80)
print(f"  decoder children: {len(list(dec))}")

shapes = {}
hs = []
for i, m in enumerate(dec):
    def mk(i):
        def h(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t):
                shapes.setdefault(i, (type(mod).__name__, list(t.shape)))
        return h
    hs.append(m.register_forward_hook(mk(i)))

z = torch.randn(16, 1, 66, 38, device=dev)
x = z.permute(1, 0, 2, 3).unsqueeze(0)
with torch.no_grad():
    out = tae.decode_video(x, parallel=False, show_progress_bar=False)
for h in hs:
    h.remove()

for i in sorted(shapes):
    name, sh = shapes[i]
    print(f"    child {i:>2} {name:<12} -> {sh}")
print()
print(f"  decode output {list(out.shape)}")

# stage boundaries by child index, as established by the earlier decomposition
print()
print("  stage 0 = children 0-8   stage 1 = 9-14   stage 2 = 15-20   final = 21-22")
if 14 in shapes:
    print(f"  TAIL ATTACH POINT: child 14 output = {shapes[14][1]}")
