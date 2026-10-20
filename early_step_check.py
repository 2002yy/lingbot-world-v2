#!/usr/bin/env python
"""§Latency-2A part 2: does an early DiT step already carry a usable picture?

The decomposition says the DiT steps plus the KV update are 94.6% of the chunk,
so a cheaper decoder would address almost nothing. The remaining question is
whether the pipeline could show something sooner: if step 1 already produces a
directionally correct image, an interaction architecture of "~250 ms preview,
~750 ms authoritative" is available without inventing anything.

This decodes step0 / step1 / step2 with the SAME decoder and compares each to the
final, plus against the first frame, so the comparison is against the actual
trajectory rather than an absolute.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402


def edge_map(x):
    g = x.mean(0, keepdim=True).unsqueeze(0)
    k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                     device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(g, k, padding=1)
    gy = F.conv2d(g, k.transpose(-1, -2), padding=1)
    e = (gx * gx + gy * gy).sqrt()
    return (e / e.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6))[0]


def ssim(a, b, win=11):
    c = torch.arange(win, dtype=torch.float32, device=a.device) - (win - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * 1.5 ** 2)); g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    a, b = a.unsqueeze(0).float(), b.unsqueeze(0).float()
    p = win // 2
    w3 = w.expand(3, 1, win, win)
    ma = F.conv2d(a, w3, padding=p, groups=3)
    mb = F.conv2d(b, w3, padding=p, groups=3)
    va = F.conv2d(a * a, w3, padding=p, groups=3) - ma * ma
    vb = F.conv2d(b * b, w3, padding=p, groups=3) - mb * mb
    vab = F.conv2d(a * b, w3, padding=p, groups=3) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * vab + C2)) / \
        ((ma * ma + mb * mb + C1) * (va + vb + C2))
    return s.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dev = "cuda"
    d = torch.load(args.latents, map_location="cpu", weights_only=False)
    keys = sorted(d.keys())
    print("=" * 78)
    print(f"  early-step usability   steps={keys}")
    print("=" * 78)
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()

    frames = {}
    for k in keys:
        z = d[k].to(dev)
        with torch.no_grad():
            fr = tae.decode_video(z.permute(1, 0, 2, 3).unsqueeze(0),
                                  parallel=False, show_progress_bar=False)
        f = fr[0] if isinstance(fr, (list, tuple)) else fr
        # normalise to [3, H, W]: drop leading batch dims and pick a middle frame
        while f.dim() > 3:
            if f.dim() == 4 and f.shape[0] in (1, 3):
                # [C, T, H, W] -> middle frame
                f = f[:, f.shape[1] // 2]
            elif f.dim() == 4:
                f = f[f.shape[0] // 2]
            else:
                f = f[0]
        if f.dim() == 3 and f.shape[0] not in (1, 3):
            f = f.permute(2, 0, 1)
        if f.dim() == 3 and f.shape[0] == 1:
            f = f.repeat(3, 1, 1)
        frames[k] = f.float().clamp(0, 1)
        print(f"  {k}: decoded {list(frames[k].shape)}")

    final = frames[keys[-1]]
    print()
    print(f"  {'step':<8} {'vs FINAL ssim':>14} {'edgeSSIM':>10} "
          f"{'mean|d|':>10} {'std':>8} {'edgeE':>8}")
    print("  " + "-" * 62)
    out = {}
    for k in keys:
        f = frames[k]
        e_final = edge_map(final).repeat(3, 1, 1)
        e_k = edge_map(f).repeat(3, 1, 1)
        row = dict(ssim_vs_final=ssim(f, final),
                   edge_ssim=ssim(e_k, e_final),
                   mean_abs=(f - final).abs().mean().item(),
                   std=f.std().item(),
                   edge_energy=edge_map(f).mean().item())
        out[k] = row
        print(f"  {k:<8} {row['ssim_vs_final']:>14.4f} {row['edge_ssim']:>10.4f} "
              f"{row['mean_abs']:>10.4f} {row['std']:>8.4f} "
              f"{row['edge_energy']:>8.4f}")

    print()
    print("  INTERPRETATION")
    print("    a high step0/step1 ssim against the FINAL frame means the early")
    print("    step already carries the structure, so a preview at that point")
    print("    would be directionally correct rather than noise.")
    print("    edgeSSIM is the more relevant figure for 'does it look like the")
    print("    same scene', since it ignores low-frequency colour drift.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/early_step.json", "w") as f:
        json.dump(out, f, indent=2)
    for k in keys:
        torch.save(frames[k].cpu(), f"{args.out_dir}/frame_{k}.pt")
    print(f"\n  wrote {args.out_dir}/early_step.json and per-step frames")


if __name__ == "__main__":
    main()
