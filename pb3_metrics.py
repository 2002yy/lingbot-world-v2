#!/usr/bin/env python
"""P2b-3 quality gate, analysis side.

Three distinct questions, deliberately kept apart:

A) PAIRED 21ch: does BF16 show degradation relative to FP8? The framing matters.
   BF16 is the ORIGINAL weights; FP8 is the lossy approximation. So "how close is
   BF16 to FP8" is the wrong target -- a large LPIPS simply means the two weights
   differ, which is expected and not evidence of harm. The gate is whether BF16 is
   internally healthy and coherent, with the paired numbers reported for context.

B) DETERMINISM: BF16 vs BF16, same seed, same inputs. Must be bit-identical. If
   it is, we have a new reference mode: original bf16 weights + FA2 repro path,
   against which every future FP8 / Sage / compile change should be judged.

C) LONG 65ch INTRINSIC HEALTH: not "is chunk 64 the same tree as FP8's chunk 64"
   -- S3-C already showed that a recurrent world model diverges into two worlds
   from any numerical difference, and that is not quality loss. Instead: does BF16
   keep producing a sane world on its own? NaN/explosion, latency and VRAM growth,
   colour and texture drift, self-similarity collapse, per-segment stability.

Usage:
    python pb3_metrics.py --paired FP8_DIR BF16_DIR
    python pb3_metrics.py --determinism BF16_DIR BF16B_DIR
    python pb3_metrics.py --health BF16_65_DIR
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F


def _pick(fr):
    """Return a [3,H,W] float image in 0..1 from whatever the decoder produced."""
    x = fr
    while x.dim() > 4:
        x = x[0]
    if x.dim() == 4:
        # [C,T,H,W] or [T,C,H,W]
        if x.shape[0] in (1, 3):
            x = x[:, x.shape[1] // 2]
        else:
            x = x[x.shape[0] // 2]
    if x.dim() == 3:
        if x.shape[0] not in (1, 3):
            x = x.permute(2, 0, 1)
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
    return x.float().clamp(0, 1)


def edge_map(x):
    g = x.float().mean(0, keepdim=True).unsqueeze(0)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                      device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    e = (gx * gx + gy * gy).sqrt()
    return (e / e.amax(dim=(-1, -2), keepdim=True).clamp_min(1e-6))[0]


def ssim(a, b, win_size=11):
    dev = a.device
    c = torch.arange(win_size, device=dev, dtype=torch.float32) - (win_size - 1) / 2
    g = torch.exp(-(c ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    a = a.float().unsqueeze(0); b = b.float().unsqueeze(0)
    pad = win_size // 2
    w3 = w.expand(3, 1, win_size, win_size)
    mu_a = F.conv2d(a, w3, padding=pad, groups=3)
    mu_b = F.conv2d(b, w3, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, w3, padding=pad, groups=3) - mu_a2
    sb_ = F.conv2d(b * b, w3, padding=pad, groups=3) - mu_b2
    sab = F.conv2d(a * b, w3, padding=pad, groups=3) - mu_ab
    s = ((2 * mu_ab + C1) * (2 * sab + C2)) / ((mu_a2 + mu_b2 + C1) * (sa + sb_ + C2))
    return s.mean().item()


def psnr(a, b, eps=1e-10):
    mse = (a.float() - b.float()).pow(2).mean().item()
    return 10 * math.log10(1.0 / max(mse, eps))


def segments_for(n):
    segs = [("0-3", 0, min(3, n - 1)), ("7-12", 7, min(12, n - 1)),
            ("13-20", 13, min(20, n - 1))]
    if n > 20:
        segs.append(("20-32", 20, min(32, n - 1)))
    if n > 32:
        segs.append(("32-48", 32, min(48, n - 1)))
    if n > 48:
        segs.append(("48-64", 48, n - 1))
    return [s for s in segs if s[1] <= s[2]]


def load(d):
    fr = torch.load(os.path.join(d, "frames.pt"), map_location="cpu",
                    weights_only=False)
    rj = json.load(open(os.path.join(d, "run.json")))
    return [_pick(f) for f in fr], rj


def try_lpips():
    try:
        import lpips as lpips_mod
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        return lpips_mod.LPIPS(net="alex").to(dev).eval(), dev
    except Exception as e:
        print(f"  LPIPS unavailable ({e}); reporting SSIM/PSNR only")
        return None, None


def paired(da, db, namea, nameb):
    A, ra = load(da)
    B, rb = load(db)
    n = min(len(A), len(B))
    print("=" * 78)
    print(f"  PAIRED: {namea} (A) vs {nameb} (B)   chunks={n}")
    print("=" * 78)
    print(f"  A: median {ra['median_ms']:.1f} ms  peak {ra['peak_mib']:.0f} MiB  "
          f"weight={ra['weight']}")
    print(f"  B: median {rb['median_ms']:.1f} ms  peak {rb['peak_mib']:.0f} MiB  "
          f"weight={rb['weight']}")
    print(f"  hash[0] A={ra['hashes'][0]}  B={rb['hashes'][0]}  "
          f"identical={ra['hashes']==rb['hashes']}")
    print()

    lp, ldev = try_lpips()
    rows = []
    for i in range(n):
        a, b = A[i], B[i]
        if lp is not None:
            with torch.no_grad():
                lv = lp((a.unsqueeze(0) * 2 - 1).to(ldev),
                        (b.unsqueeze(0) * 2 - 1).to(ldev)).item()
        else:
            lv = float("nan")
        rows.append(dict(chunk=i, lpips=lv, ssim=ssim(a, b), psnr=psnr(a, b),
                         edge_ssim=ssim(edge_map(a).repeat(3, 1, 1),
                                        edge_map(b).repeat(3, 1, 1))))

    def seg_line(lo, hi):
        sel = [r for r in rows if lo <= r["chunk"] <= hi]
        if not sel:
            return None
        return dict(lpips=statistics.mean(r["lpips"] for r in sel),
                    ssim=statistics.mean(r["ssim"] for r in sel),
                    psnr=statistics.mean(r["psnr"] for r in sel),
                    edge_ssim=statistics.mean(r["edge_ssim"] for r in sel),
                    n=len(sel))

    print(f"  {'segment':<10} {'LPIPS':>9} {'SSIM':>8} {'PSNR':>8} {'edgeSSIM':>9}")
    print("  " + "-" * 48)
    for nm, lo, hi in segments_for(n):
        s = seg_line(lo, hi)
        if s:
            print(f"  {nm:<10} {s['lpips']:>9.4f} {s['ssim']:>8.4f} "
                  f"{s['psnr']:>8.2f} {s['edge_ssim']:>9.4f}")
    alls = seg_line(0, n - 1)
    print("  " + "-" * 48)
    print(f"  {'ALL':<10} {alls['lpips']:>9.4f} {alls['ssim']:>8.4f} "
          f"{alls['psnr']:>8.2f} {alls['edge_ssim']:>9.4f}")

    # late-segment acceleration is the dangerous shape
    print()
    print("  late-vs-early (a RECURRENT warning shape is early~equal then a late")
    print("  blow-up; here a spread is EXPECTED since the weights differ):")
    e = seg_line(0, 3)
    l = seg_line(max(0, n - 8), n - 1)
    if e and l:
        print(f"    early LPIPS {e['lpips']:.4f} -> late LPIPS {l['lpips']:.4f}")
        print(f"    early SSIM  {e['ssim']:.4f} -> late SSIM  {l['ssim']:.4f}")
    return dict(namea=namea, nameb=nameb, n=n, rows=rows)


def determinism(da, db, namea, nameb):
    A, ra = load(da)
    B, rb = load(db)
    print("=" * 78)
    print(f"  DETERMINISM: {namea} vs {nameb}  (same seed, same inputs)")
    print("=" * 78)
    hs = ra["hashes"] == rb["hashes"]
    print(f"  latent hashes identical : {hs}  "
          f"({sum(1 for x, y in zip(ra['hashes'], rb['hashes']) if x != y)}"
          f"/{len(ra['hashes'])} differ)")
    n = min(len(A), len(B))
    worst = 0.0
    for i in range(n):
        d = (A[i] - B[i]).abs().max().item()
        worst = max(worst, d)
    print(f"  decoded frames          : max|diff| = {worst:.3e} over {n} chunks")
    print(f"  -> BF16 reference mode is "
          f"{'STABLE (usable as reference)' if hs and worst == 0 else 'NOT deterministic'}")
    return dict(hashes_identical=hs, frame_max_diff=worst)


def health(d, name):
    A, r = load(d)
    n = len(A)
    print("=" * 78)
    print(f"  INTRINSIC HEALTH: {name}  weight={r['weight']}  chunks={n}")
    print("=" * 78)

    ms = r["ms"]
    vr = r["vram_alloc_mib"]
    print(f"  latency: median {statistics.median(ms):.1f} ms  "
          f"min {min(ms):.1f}  max {max(ms):.1f}  "
          f"first10 {statistics.mean(ms[:10]):.1f}  "
          f"last10 {statistics.mean(ms[-10:]):.1f}")
    drift = (statistics.mean(ms[-10:]) - statistics.mean(ms[:10])) / \
        statistics.mean(ms[:10]) * 100
    print(f"           early->late drift {drift:+.1f}%  "
          f"(a large positive drift means growing cost with length)")
    print(f"  VRAM alloc: first {vr[0]:.0f} MiB  last {vr[-1]:.0f} MiB  "
          f"peak {r['peak_mib']:.0f} MiB")
    vdrift = (vr[-1] - vr[0]) / max(vr[0], 1) * 100
    print(f"           early->late drift {vdrift:+.1f}%")

    # NaN / explosion / dead frames
    bad = []
    for i, x in enumerate(A):
        if not torch.isfinite(x).all():
            bad.append((i, "non-finite"))
            continue
        sd = x.std().item()
        mn, mx = x.min().item(), x.max().item()
        if sd < 1e-4:
            bad.append((i, f"near-constant std={sd:.2e}"))
        if mn < -1e-6 or mx > 1 + 1e-6:
            pass  # already clamped
    print(f"  finite/health check: {'OK' if not bad else bad[:5]}")

    # colour and texture drift
    means = [x.mean(dim=(1, 2)).tolist() for x in A]
    stds = [x.std(dim=(1, 2)).tolist() for x in A]
    edges = [edge_map(x).mean().item() for x in A]

    def drift_of(series, label):
        if len(series) < 11:
            return
        f = statistics.mean(series[:10])
        l = statistics.mean(series[-10:])
        print(f"  {label:<22} first10 {f:.4f}  last10 {l:.4f}  "
              f"({(l-f)/max(abs(f),1e-9)*100:+.1f}%)")
    drift_of([sum(m) / 3 for m in means], "frame mean (brightness)")
    drift_of([sum(s) / 3 for s in stds], "frame std (contrast)")
    drift_of(edges, "edge energy (texture)")

    # self-similarity: how fast does the world drift from chunk 0
    print()
    print("  self-similarity vs chunk 0 (SSIM):")
    s0 = [ssim(A[0], A[i]) for i in range(n)]
    for lo, hi in ((0, 3), (7, 12), (13, 20), (20, 32), (32, 48), (48, n - 1)):
        if lo >= n:
            continue
        sel = s0[lo:min(hi + 1, n)]
        if sel:
            print(f"    chunks {lo:>2}-{hi:<2}: SSIM {statistics.mean(sel):.4f}")
    # monotone decay would signal collapse; a floor is healthy
    tail = s0[max(0, n - 8):]
    print(f"    tail mean {statistics.mean(tail):.4f}  "
          f"(a floor well above 0 = the world is still coherent)")
    print()
    print("  per-chunk consecutive SSIM (temporal stability):")
    cons = [ssim(A[i], A[i + 1]) for i in range(n - 1)]
    for nm, lo, hi in segments_for(n):
        sel = cons[lo:min(hi, n - 1)]
        if sel:
            print(f"    {nm:<10} {statistics.mean(sel):.4f}")

    return dict(median_ms=statistics.median(ms), latency_drift_pct=drift,
                vram_drift_pct=vdrift, unhealthy=bad[:5],
                self_ssim_tail=statistics.mean(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", nargs=2, metavar=("A_DIR", "B_DIR"))
    ap.add_argument("--determinism", nargs=2, metavar=("A_DIR", "B_DIR"))
    ap.add_argument("--health", metavar="DIR")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    res = {}
    if args.paired:
        a, b = args.paired
        res["paired"] = paired(a, b, os.path.basename(a), os.path.basename(b))
    if args.determinism:
        a, b = args.determinism
        res["determinism"] = determinism(a, b, os.path.basename(a),
                                         os.path.basename(b))
    if args.health:
        res["health"] = health(args.health, os.path.basename(args.health))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n[metrics] wrote {args.out}")


if __name__ == "__main__":
    main()
