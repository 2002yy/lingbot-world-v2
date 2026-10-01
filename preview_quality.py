#!/usr/bin/env python
"""Preview-1A quality: is step0 preview structure consistent across real controls?

The earlier figure (SSIM 0.687 / edgeSSIM 0.635) came from one sample. Before
treating the preview as a product path it must be characterised across a set of real
controls, and this reports per-control values plus a gross-failure count rather than
a single average that could hide a bimodal result.

This is not a new benchmark. It answers one question: was that sample a fluke?
"""
import argparse
import json
import os
import statistics
import sys

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
    return (((2 * ma * mb + C1) * (2 * vab + C2)) /
            ((ma * ma + mb * mb + C1) * (va + vb + C2))).mean().item()


def decode(tae, z, dev):
    with torch.no_grad():
        fr = tae.decode_video(z.permute(1, 0, 2, 3).unsqueeze(0).to(dev),
                              parallel=False, show_progress_bar=False)
    f = fr[0] if isinstance(fr, (list, tuple)) else fr
    while f.dim() > 3:
        if f.dim() == 4 and f.shape[0] in (1, 3):
            f = f[:, f.shape[1] // 2]
        elif f.dim() == 4:
            f = f[f.shape[0] // 2]
        else:
            f = f[0]
    if f.dim() == 3 and f.shape[0] not in (1, 3):
        f = f.permute(2, 0, 1)
    if f.dim() == 3 and f.shape[0] == 1:
        f = f.repeat(3, 1, 1)
    return f.float().clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--gross_ssim", type=float, default=0.30,
                    help="below this, the preview is not directionally correct")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dev = "cuda"
    d = torch.load(args.latents, map_location="cpu", weights_only=False)
    chunks = sorted({int(k[1:].split("_")[0]) for k in d if k.startswith("c")})
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()

    print("=" * 78)
    print(f"  Preview-1A quality across {len(chunks)} real controls")
    print("=" * 78)
    print(f"  {'chunk':>6} {'ssim vs final':>14} {'edgeSSIM':>10} "
          f"{'mean|d|':>9} {'std':>8} {'verdict':>10}")
    print("  " + "-" * 66)
    rows = []
    for c in chunks:
        s0 = decode(tae, d[f"c{c}_step0"], dev)
        fin = decode(tae, d[f"c{c}_final"], dev)
        v = dict(chunk=c, ssim=ssim(s0, fin),
                 edge_ssim=ssim(edge_map(s0).repeat(3, 1, 1),
                                edge_map(fin).repeat(3, 1, 1)),
                 mean_abs=(s0 - fin).abs().mean().item(),
                 std=s0.std().item())
        v["gross_failure"] = v["ssim"] < args.gross_ssim
        rows.append(v)
        print(f"  {c:>6} {v['ssim']:>14.4f} {v['edge_ssim']:>10.4f} "
              f"{v['mean_abs']:>9.4f} {v['std']:>8.4f} "
              f"{'GROSS FAIL' if v['gross_failure'] else 'ok':>10}")

    ss = [r["ssim"] for r in rows]
    es = [r["edge_ssim"] for r in rows]
    nf = sum(1 for r in rows if r["gross_failure"])
    print()
    print(f"  ssim       p50 {statistics.median(ss):.4f}  min {min(ss):.4f}  "
          f"max {max(ss):.4f}  spread {max(ss)-min(ss):.4f}")
    print(f"  edgeSSIM   p50 {statistics.median(es):.4f}  min {min(es):.4f}  "
          f"max {max(es):.4f}")
    print(f"  gross structural failures (ssim < {args.gross_ssim}): {nf}/{len(rows)}")
    print()
    print(f"  single-sample reference was 0.6871 / 0.6352")
    print(f"  -> across {len(rows)} controls the p50 is "
          f"{statistics.median(ss):.4f} / {statistics.median(es):.4f}")
    if nf == 0 and statistics.median(ss) > 0.5:
        print("  VERDICT: step0 preview is consistently directionally correct;")
        print("           the single sample was not a fluke.")
    else:
        print("  VERDICT: the preview is NOT consistently usable; see the")
        print("           failures above before productising.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/preview_quality.json", "w") as f:
        json.dump(dict(rows=rows, n=len(rows), gross_failures=nf,
                       ssim_p50=statistics.median(ss),
                       edge_ssim_p50=statistics.median(es)), f, indent=2)
    print(f"\n  wrote {args.out_dir}/preview_quality.json")


if __name__ == "__main__":
    main()
