#!/usr/bin/env python
"""§Preview-2A: audit the three-frame trajectory the user actually sees.

    A = previous chunk's AUTHORITATIVE final
    P = current chunk's variant D(step0) preview
    F = current chunk's AUTHORITATIVE final

    A  --~231 ms-->  P  --~550 ms-->  F

Two handoffs, not one:

    A -> P   does the preview step in the right direction, or jump somewhere else?
    P -> F   how much correction is left when authority takes over?
    A -> F   the baseline: how much visual change would have happened anyway

Note on provenance: A is the previous chunk's authoritative final, not a D-decoded
version of it. The user was looking at an authoritative frame, so that is what the
comparison must start from.

The headline numbers are not raw SSIM but two ratios:

    progress_ratio  = d(A,P) / d(A,F)     <1 means the preview advanced toward F
    correction_ratio= d(P,F) / d(A,F)     <1 means less correction remains than the
                                          whole change; >1 means the preview moved
                                          the user FURTHER from F than the old frame

plus a direction check from the latent deltas, because a preview can look acceptable
against F while having pulled the user the wrong way first.
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.nn as nn
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


def l1(a, b):
    return (a.float() - b.float()).abs().mean().item()


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


def cos(a, b):
    a = a.flatten().float(); b = b.flatten().float()
    na, nb = a.norm(), b.norm()
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float((a @ b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdouts", nargs="+", required=True)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dev = "cuda"
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    dec = tae.decoder
    up_idx = [i for i, m in enumerate(dec) if type(m).__name__ == "Upsample"][-1]
    _orig = dec[up_idx]

    def full(z):
        with torch.no_grad():
            return to_img(tae.decode_video(z.permute(1, 0, 2, 3).unsqueeze(0),
                                           parallel=False,
                                           show_progress_bar=False))

    tgt_hw = None

    def variant_d(z):
        nonlocal tgt_hw
        dec[up_idx] = nn.Identity()
        try:
            img = full(z)
        finally:
            dec[up_idx] = _orig
        return F.interpolate(img.unsqueeze(0), size=tgt_hw, mode="bicubic",
                             align_corners=False).squeeze(0).clamp(0, 1)

    print("=" * 96)
    print("  Preview-2A  three-frame handoff audit   A -> P -> F")
    print("=" * 96)
    print("  A = previous authoritative final   P = variant D(step0)   "
          "F = current authoritative final")
    print()

    all_rows = []
    out = {}
    for spec in args.holdouts:
        tag, path = spec.split("=", 1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        chunks = sorted({int(k[1:].split("_")[0]) for k in d if k.startswith("c")})
        chunks = [c for c in chunks if f"c{c}_final" in d and
                  f"c{c}_step0" in d]
        if tgt_hw is None:
            tgt_hw = full(d[f"c{chunks[0]}_step0"].to(dev)).shape[-2:]

        rows = []
        for i in range(1, len(chunks)):          # need a previous chunk
            cprev, c = chunks[i - 1], chunks[i]
            A_lat = d[f"c{cprev}_final"].to(dev)
            S_lat = d[f"c{c}_step0"].to(dev)
            F_lat = d[f"c{c}_final"].to(dev)

            A, P, Fimg = full(A_lat), variant_d(S_lat), full(F_lat)

            d_AF = l1(A, Fimg)
            d_AP = l1(A, P)
            d_PF = l1(P, Fimg)
            rows.append(dict(
                chunk=c, prev=cprev,
                ssim_AF=ssim(A, Fimg), ssim_AP=ssim(A, P), ssim_PF=ssim(P, Fimg),
                e_AF=ssim(edge_map(A).repeat(3, 1, 1),
                          edge_map(Fimg).repeat(3, 1, 1)),
                e_AP=ssim(edge_map(A).repeat(3, 1, 1),
                          edge_map(P).repeat(3, 1, 1)),
                e_PF=ssim(edge_map(P).repeat(3, 1, 1),
                          edge_map(Fimg).repeat(3, 1, 1)),
                d_AF=d_AF, d_AP=d_AP, d_PF=d_PF,
                progress_ratio=(d_AP / d_AF) if d_AF > 1e-9 else float("nan"),
                correction_ratio=(d_PF / d_AF) if d_AF > 1e-9 else float("nan"),
                # latent-space direction agreement: is P in F's direction from A?
                cos_AP_AF=cos(S_lat - A_lat, F_lat - A_lat)))
        out[tag] = rows
        all_rows += rows

        print(f"  --- {tag}  ({len(rows)} transitions) ---")
        print(f"  {'chunk':>6} {'A->F ssim':>10} {'A->P':>7} {'P->F':>7} "
              f"{'progress':>9} {'correct':>8} {'cos':>7} {'dir':>13}")
        print("  " + "-" * 78)

        def classify(cv):
            if cv != cv:
                return "n/a"
            if cv < 0.0:
                return "WRONG DIR"
            if cv < 0.5:
                return "orthogonal"
            if cv < 0.9:
                return "partial"
            return "same dir"

        for r in rows:
            print(f"  {r['chunk']:>6} {r['ssim_AF']:>10.4f} "
                  f"{r['ssim_AP']:>7.4f} {r['ssim_PF']:>7.4f} "
                  f"{r['progress_ratio']:>9.3f} {r['correction_ratio']:>8.3f} "
                  f"{r['cos_AP_AF']:>7.3f} {classify(r['cos_AP_AF']):>13}")
        print()

    # ---- aggregate ----
    def agg(key, rows=None):
        v = [r[key] for r in (rows or all_rows)]
        v = [x for x in v if x == x]
        return statistics.median(v) if v else float("nan")

    start = [r for r in all_rows if r["chunk"] <= 2]
    steady = [r for r in all_rows if r["chunk"] > 2]
    wrong = sum(1 for r in all_rows if r["cos_AP_AF"] == r["cos_AP_AF"]
                and r["cos_AP_AF"] < 0)
    overshoot = sum(1 for r in all_rows
                    if r["correction_ratio"] == r["correction_ratio"]
                    and r["correction_ratio"] > 1.0)
    gross = sum(1 for r in all_rows if r["ssim_PF"] < 0.30)

    print("=" * 96)
    print("  AGGREGATE  (all scenes)")
    print("=" * 96)
    print(f"  baseline change   A->F   ssim {agg('ssim_AF'):.4f}  "
          f"L1 {agg('d_AF'):.4f}")
    print(f"  preview step      A->P   ssim {agg('ssim_AP'):.4f}  "
          f"L1 {agg('d_AP'):.4f}")
    print(f"  remaining         P->F   ssim {agg('ssim_PF'):.4f}  "
          f"L1 {agg('d_PF'):.4f}")
    print()
    print(f"  progress_ratio   (d(A,P)/d(A,F))  {agg('progress_ratio'):.3f}")
    print(f"  correction_ratio (d(P,F)/d(A,F))  {agg('correction_ratio'):.3f}")
    print(f"  latent direction cos(P-A, F-A)    {agg('cos_AP_AF'):.3f}")
    print()
    print(f"  wrong-direction handoffs : {wrong}/{len(all_rows)}")
    print(f"  overshoot (correction>1) : {overshoot}/{len(all_rows)}")
    print(f"  gross structural jumps   : {gross}/{len(all_rows)}")
    print()
    print(f"  session-start  progress {agg('progress_ratio', start):.3f}  "
          f"correction {agg('correction_ratio', start):.3f}  "
          f"cos {agg('cos_AP_AF', start):.3f}")
    print(f"  steady-state   progress {agg('progress_ratio', steady):.3f}  "
          f"correction {agg('correction_ratio', steady):.3f}  "
          f"cos {agg('cos_AP_AF', steady):.3f}")
    print()
    pr, cr = agg("progress_ratio"), agg("correction_ratio")
    if wrong == 0 and cr < 0.8 and pr < 1.0:
        print("  VERDICT: the preview advances toward F, the handoff correction is")
        print("           smaller than the full change, and no handoff went the")
        print("           wrong way. Direct replacement is likely sufficient;")
        print("           a blend would be polishing a working transition.")
    elif wrong == 0 and cr >= 0.8:
        print("  VERDICT: direction is right but the preview barely advances")
        print("           (correction_ratio ~= 1). The handoff still carries almost")
        print("           the whole change, so a blend is worth testing (Preview-2B).")
    else:
        print("  VERDICT: wrong-direction handoffs exist. A blend would only mask")
        print("           a preview that pulls the user the wrong way first; fix the")
        print("           preview before smoothing the transition.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/preview_2a.json", "w") as f:
        json.dump(dict(per_scene=out, n=len(all_rows), wrong_direction=wrong,
                       overshoot=overshoot, gross_jumps=gross,
                       progress_ratio=pr, correction_ratio=cr,
                       cos=agg("cos_AP_AF"),
                       start=dict(progress=agg("progress_ratio", start),
                                  correction=agg("correction_ratio", start),
                                  cos=agg("cos_AP_AF", start)),
                       steady=dict(progress=agg("progress_ratio", steady),
                                   correction=agg("correction_ratio", steady),
                                   cos=agg("cos_AP_AF", steady))), f, indent=2)
    print(f"\n  wrote {args.out_dir}/preview_2a.json")


if __name__ == "__main__":
    main()
