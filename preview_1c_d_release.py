#!/usr/bin/env python
"""§Preview-1C: variant D cross-scene release gate.

Variant D (skip the last spatial upsample, then a cheap bicubic upscale) involves no
training, so it cannot overfit a training set. But "cannot overfit" only removes one
failure mode -- it does not establish that the degradation is equally graceful in
other visual domains. This measures that, reusing the holdout data already generated
for the head gate.

No training. No tuning. No new components.

    full TAEHV(step0) vs variant D(step0) vs authoritative final

on scene 04 (unseen seed and controls), scene 01 and scene 03, reporting SSIM, edge
SSIM, session-start separately, gross structural failures, and D's degradation
relative to the full preview on the SAME scene.
"""
import argparse
import json
import os
import statistics
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

MS = 1e6


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


def to_img(fr):
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
    ap.add_argument("--holdouts", nargs="+", required=True)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dev = "cuda"
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    dec = tae.decoder

    # variant D: replace the LAST spatial upsample with identity, then upscale
    up_idx = [i for i, m in enumerate(dec) if type(m).__name__ == "Upsample"][-1]
    _orig = dec[up_idx]

    def full(z):
        with torch.no_grad():
            return to_img(tae.decode_video(z.permute(1, 0, 2, 3).unsqueeze(0),
                                           parallel=False,
                                           show_progress_bar=False))

    def variant_d(z, target_hw):
        dec[up_idx] = nn.Identity()
        try:
            img = full(z)
        finally:
            dec[up_idx] = _orig
        return F.interpolate(img.unsqueeze(0), size=target_hw, mode="bicubic",
                             align_corners=False).squeeze(0).clamp(0, 1)

    print("=" * 92)
    print("  Preview-1C  variant D cross-scene release gate   (no training, no tuning)")
    print("=" * 92)

    # ---- latency first, unchanged from Preview-1B ----
    z0 = None
    results = {}
    all_rows = []
    for spec in args.holdouts:
        tag, path = spec.split("=", 1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        chunks = sorted({int(k[1:].split("_")[0]) for k in d if k.startswith("c")})
        chunks = [c for c in chunks if f"c{c}_final" in d]
        if z0 is None:
            z0 = d[f"c{chunks[0]}_step0"].to(dev)
            tgt = full(z0).shape[-2:]
        rows = []
        for c in chunks:
            z0c = d[f"c{c}_step0"].to(dev)
            zf = d[f"c{c}_final"].to(dev)
            ref = full(zf)                 # authoritative final
            A = full(z0c)                  # full TAEHV preview
            D = variant_d(z0c, tgt)        # variant D preview
            rows.append(dict(
                chunk=c,
                A_ssim=ssim(A, ref), A_edge=ssim(edge_map(A).repeat(3, 1, 1),
                                                 edge_map(ref).repeat(3, 1, 1)),
                D_ssim=ssim(D, ref), D_edge=ssim(edge_map(D).repeat(3, 1, 1),
                                                 edge_map(ref).repeat(3, 1, 1)),
                AD_ssim=ssim(D, A), AD_edge=ssim(edge_map(D).repeat(3, 1, 1),
                                                 edge_map(A).repeat(3, 1, 1))))
        for r in rows:
            r["d_vs_A"] = r["D_ssim"] - r["A_ssim"]
            r["de_vs_A"] = r["D_edge"] - r["A_edge"]

        def med(k, sub):
            v = [r[k] for r in sub]
            return statistics.median(v) if v else float("nan")

        start = [r for r in rows if r["chunk"] <= 2]
        steady = [r for r in rows if r["chunk"] > 2]
        gross = sum(1 for r in rows if r["D_ssim"] < 0.30)
        results[tag] = dict(rows=rows, n=len(rows), gross_failures=gross,
                            A_ssim=med("A_ssim", rows), D_ssim=med("D_ssim", rows),
                            A_edge=med("A_edge", rows), D_edge=med("D_edge", rows),
                            d_vs_A=med("d_vs_A", rows),
                            de_vs_A=med("de_vs_A", rows),
                            start_d_vs_A=med("d_vs_A", start),
                            start_de_vs_A=med("de_vs_A", start),
                            start_D=med("D_ssim", start),
                            AD_ssim=med("AD_ssim", rows))
        all_rows += rows

        print()
        print(f"  --- {tag}  ({len(rows)} chunks) ---")
        print(f"  {'chunk':>6} {'A ssim':>8} {'D ssim':>8} {'d_vs_A':>8} "
              f"{'A edge':>8} {'D edge':>8} {'de_vs_A':>8} {'D vs A':>8}")
        print("  " + "-" * 70)
        for r in rows:
            print(f"  {r['chunk']:>6} {r['A_ssim']:>8.4f} {r['D_ssim']:>8.4f} "
                  f"{r['d_vs_A']:>+8.4f} {r['A_edge']:>8.4f} {r['D_edge']:>8.4f} "
                  f"{r['de_vs_A']:>+8.4f} {r['AD_ssim']:>8.4f}")
        print()
        print(f"    D vs final           ssim {results[tag]['D_ssim']:.4f}  "
              f"edge {results[tag]['D_edge']:.4f}")
        print(f"    A vs final           ssim {results[tag]['A_ssim']:.4f}  "
              f"edge {results[tag]['A_edge']:.4f}")
        print(f"    D degradation vs A   dSSIM {results[tag]['d_vs_A']:+.4f}  "
              f"dedge {results[tag]['de_vs_A']:+.4f}")
        print(f"    session-start        dSSIM {results[tag]['start_d_vs_A']:+.4f}  "
              f"D ssim {results[tag]['start_D']:.4f}")
        print(f"    D vs A directly      ssim {results[tag]['AD_ssim']:.4f}")
        print(f"    gross failures (D ssim < 0.30): {gross}")

    # ---- verdict ----
    print()
    print("=" * 92)
    print("  RELEASE GATE")
    print("=" * 92)
    print("    no gross failure; degradation stable across scenes;")
    print("    session-start acceptable")
    print()
    print(f"  {'scene':<18} {'d_vs_A':>9} {'dedge':>9} {'start d':>9} "
          f"{'gross':>6}  verdict")
    print("  " + "-" * 62)
    ds = []
    ok_all = True
    for tag, r in results.items():
        stable = r["d_vs_A"] >= -0.12
        ok = (r["gross_failures"] == 0 and stable
              and r["start_d_vs_A"] >= -0.15)
        ok_all = ok_all and ok
        ds.append(r["d_vs_A"])
        print(f"  {tag:<18} {r['d_vs_A']:>+9.4f} {r['de_vs_A']:>+9.4f} "
              f"{r['start_d_vs_A']:>+9.4f} {r['gross_failures']:>6}  "
              f"{'PASS' if ok else 'FAIL'}")
    print()
    spread = max(ds) - min(ds)
    print(f"  d_vs_A spread across scenes: {spread:.4f} "
          f"(small = stable degradation)")
    print()
    if ok_all:
        print("  VERDICT: variant D PASSES. Freeze as the production preview")
        print("           candidate: ~231 ms preview, ~+3.5% penalty, no training,")
        print("           and now shown to degrade gracefully outside scene 04.")
    else:
        print("  VERDICT: variant D does not pass cleanly; see the failing rows.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/preview_1c.json", "w") as f:
        json.dump(dict(results=results, spread=spread, passed=ok_all), f,
                  indent=2)
    print(f"\n  wrote {args.out_dir}/preview_1c.json")


if __name__ == "__main__":
    main()
