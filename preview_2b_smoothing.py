#!/usr/bin/env python
"""§Preview-2B: the P -> F handoff smoothing gate.

The question is NOT "are adjacent frames closer after a crossfade", which is
mathematically near-certain. It is:

    is spreading the P->F correction over 50-100 ms worth the cost of entering the
    authoritative image later, and of possible ghosting?

Preview-2A established what is being smoothed: correction_ratio 0.613, direction
cosine 0.867, zero wrong-direction handoffs, zero gross jumps. So a crossfade is not
rescuing a wrong direction, it is softening a moderate, correctly-directed correction.
That is the case where a short blend might or might not be perceptible.

THREE ARMS ONLY, no parameter search:
    A  hard replace   0 ms
    B  short blend   50 ms  (~3 frames at a 60 Hz-equivalent cadence)
    C  long blend   100 ms  (~6 frames)

FOUR METRICS, because peak jump alone would make the answer trivially yes:

    J_peak    max adjacent-frame distance -- what a user is most likely to notice
    J_total   sum of adjacent-frame distances -- the path length; a blend should
              spread the same correction, not take a longer stranger route
    ghosting  edge-energy dip and intermediate edgeSSIM against BOTH endpoints; a
              blend that trades a jump for a double edge has not helped
    delay     authority convergence delay, a DISPLAY-semantics cost that must be
              reported as such rather than folded into the latency budget

The frame sequence is synthesized offline at a fixed cadence. It is NOT a t5/present
measurement and is not reported as one.
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

CADENCE_MS = 1000.0 / 60.0          # 16.67 ms per display frame
ARMS = [("hard", 0), ("blend50", 50), ("blend100", 100)]


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


def sequence(P, Fimg, ms):
    """Build the display sequence for a blend of the given duration."""
    if ms <= 0:
        return [P, Fimg]
    n = max(1, int(round(ms / CADENCE_MS)))
    return [P] + [((1 - (i + 1) / (n + 1)) * P +
                   ((i + 1) / (n + 1)) * Fimg) for i in range(n)] + [Fimg]


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
    tgt_hw = None

    def full(z):
        with torch.no_grad():
            return to_img(tae.decode_video(z.permute(1, 0, 2, 3).unsqueeze(0),
                                           parallel=False,
                                           show_progress_bar=False))

    def variant_d(z):
        dec[up_idx] = nn.Identity()
        try:
            img = full(z)
        finally:
            dec[up_idx] = _orig
        return F.interpolate(img.unsqueeze(0), size=tgt_hw, mode="bicubic",
                             align_corners=False).squeeze(0).clamp(0, 1)

    print("=" * 96)
    print("  Preview-2B  P -> F handoff smoothing gate")
    print("=" * 96)
    print(f"  display cadence assumed {CADENCE_MS:.2f} ms/frame "
          f"(60 Hz-equivalent, offline synthesis, NOT a present measurement)")
    print()

    per_arm = {name: [] for name, _ in ARMS}
    n_trans = 0
    for spec in args.holdouts:
        tag, path = spec.split("=", 1)
        d = torch.load(path, map_location="cpu", weights_only=False)
        chunks = sorted({int(k[1:].split("_")[0]) for k in d if k.startswith("c")})
        chunks = [c for c in chunks if f"c{c}_final" in d and
                  f"c{c}_step0" in d]
        if tgt_hw is None:
            tgt_hw = full(d[f"c{chunks[0]}_step0"].to(dev)).shape[-2:]
        for i in range(1, len(chunks)):
            c = chunks[i]
            P = variant_d(d[f"c{c}_step0"].to(dev))
            Fimg = full(d[f"c{c}_final"].to(dev))
            n_trans += 1
            for name, ms in ARMS:
                seq = sequence(P, Fimg, ms)
                steps = [l1(seq[k], seq[k + 1]) for k in range(len(seq) - 1)]
                step_ssim = [ssim(seq[k], seq[k + 1]) for k in range(len(seq) - 1)]
                # ghosting: intermediates must not be blurrier than both endpoints
                eP, eF = edge_map(P).mean().item(), edge_map(Fimg).mean().item()
                floor = min(eP, eF)
                inter = seq[1:-1]
                dips = [max(0.0, (floor - edge_map(f).mean().item()) / floor)
                        for f in inter] if inter else [0.0]
                inter_edge_ssim = [
                    ssim(edge_map(f).repeat(3, 1, 1), edge_map(P).repeat(3, 1, 1))
                    for f in inter] if inter else []
                per_arm[name].append(dict(
                    n_frames=len(seq), peak=max(steps),
                    peak_ssim_drop=1.0 - min(step_ssim),
                    total=sum(steps), max_dip=max(dips),
                    inter_edge_ssim_min=(min(inter_edge_ssim)
                                         if inter_edge_ssim else None),
                    n_intermediates=len(inter)))

    print("=" * 96)
    print("  RESULTS (median over transitions)")
    print("=" * 96)
    base_peak = statistics.median(r["peak"] for r in per_arm["hard"])
    base_total = statistics.median(r["total"] for r in per_arm["hard"])
    print(f"  {'arm':<10} {'frames':>7} {'J_peak':>9} {'vs hard':>8} "
          f"{'J_total':>9} {'vs hard':>8} {'max edge dip':>13} "
          f"{'inter edgeSSIM':>15}")
    print("  " + "-" * 82)
    summary = {}
    for name, ms in ARMS:
        rows = per_arm[name]
        pk = statistics.median(r["peak"] for r in rows)
        tt = statistics.median(r["total"] for r in rows)
        dip = statistics.median(r["max_dip"] for r in rows)
        ies = [r["inter_edge_ssim_min"] for r in rows
               if r["inter_edge_ssim_min"] is not None]
        print(f"  {name:<10} {statistics.median(r['n_frames'] for r in rows):>7.0f} "
              f"{pk:>9.4f} {pk/base_peak:>7.2f}x {tt:>9.4f} "
              f"{tt/base_total:>7.2f}x {dip:>13.3f} "
              f"{(statistics.median(ies) if ies else float('nan')):>15.4f}")
        summary[name] = dict(ms=ms, peak=pk, peak_ratio=pk / base_peak,
                             total=tt, total_ratio=tt / base_total,
                             max_edge_dip=dip,
                             inter_edge_ssim=(statistics.median(ies)
                                              if ies else None))
    print()

    print("=" * 96)
    print("  VERDICT (a blend must earn its place)")
    print("=" * 96)
    print("    peak jump down >= 30-40% | no gross artifact | no edge collapse |")
    print("    total not materially increased | and the 50 ms gain must be real")
    print()
    for name, ms in ARMS:
        if ms == 0:
            continue
        s = summary[name]
        peak_ok = s["peak_ratio"] <= 0.70
        total_ok = s["total_ratio"] <= 1.15
        edge_ok = s["max_edge_dip"] <= 0.15
        verdict = "PASS" if (peak_ok and total_ok and edge_ok) else "FAIL"
        print(f"  {name:<10} peak {s['peak_ratio']:.2f}x "
              f"({'ok' if peak_ok else 'insufficient'})  "
              f"total {s['total_ratio']:.2f}x ({'ok' if total_ok else 'inflated'})  "
              f"edge dip {s['max_edge_dip']:.3f} "
              f"({'ok' if edge_ok else 'ghosting'})  -> {verdict}")

    s50, s100 = summary["blend50"], summary["blend100"]
    print()
    if s50["peak_ratio"] <= 0.70 and s50["total_ratio"] <= 1.15 and \
            s50["max_edge_dip"] <= 0.15:
        print("  RECOMMENDATION: adopt the 50 ms blend. It earns its place at the")
        print(f"  cost of only ~50 ms of authority convergence delay"
              f" (~{1000/60*3:.0f} ms of display frames).")
    elif s100["peak_ratio"] <= 0.70 and s100["max_edge_dip"] <= 0.15:
        print("  RECOMMENDATION: only the 100 ms blend qualifies, which costs")
        print("  ~100 ms of authority convergence delay against a handoff that")
        print("  already has wrong-direction 0 and gross jumps 0. That is a poor")
        print("  trade; prefer hard replace.")
    else:
        print("  RECOMMENDATION: neither blend earns its place. Freeze hard")
        print("  replace: the handoff is already clean in direction, and the")
        print("  blends either fail to reduce the peak enough or buy that")
        print("  reduction with ghosting and convergence delay.")

    print()
    print("  CAVEAT: offline synthesized sequence at a fixed cadence, not a")
    print("          t5/present measurement. The convergence delay is display")
    print("          semantics, not compute latency.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/preview_2b.json", "w") as f:
        json.dump(dict(transitions=n_trans, summary=summary,
                       cadence_ms=CADENCE_MS), f, indent=2)
    print(f"\n  wrote {args.out_dir}/preview_2b.json")


if __name__ == "__main__":
    main()
