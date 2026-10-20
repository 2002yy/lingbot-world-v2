#!/usr/bin/env python
"""S3-B.2: A vs D on cached frames -- the single most important diff of this line.

    A = FA2    + eager      (original production path)
    D = Hybrid + compile    (final candidate)

WHAT THIS ANSWERS
-----------------
"the -7.2% we actually got -- what did it cost visually?"

This is NOT the sum of the two increments. LPIPS/SSIM are not additive drift
quantities: LPIPS(A,D) != LPIPS(A,B) + LPIPS(B,D). The earlier statement that
the two drifts "stack" was wrong and is retracted. Only the direct A-vs-D
measurement answers the question, which is why we do it here.

NO RE-ROLLOUT NEEDED
--------------------
output/visual_gate/frames_A1.pt is the FA2+eager frames and
output/vg_db/frames_D.pt is the Hybrid+compile frames, both from an identical
21-chunk rollout (scene 04, seed 42, frames 81, sage_min_kv 2508). So the
comparison runs on cached tensors.

WHAT DECIDES IT
---------------
Not the absolute LPIPS number. The questions are:
    * same camera trajectory?
    * same tree / wall / path (topology)?
    * any object appearing or disappearing?
    * do silhouettes / geometry shift?
    * is the difference still concentrated in texture/detail?
    * is the late-stage LPIPS still decelerating rather than accelerating?

If at chunk 20 it is still "same tree / same wall / same path / same camera,
differences in texture and local detail", D is promoted to final candidate.
"""
import argparse
import json
import os
import statistics
import sys

import torch
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visual_gate import psnr, ssim, rgb_stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a_cache", default="output/visual_gate/frames_A1.pt")
    ap.add_argument("--d_cache", default="output/vg_db/frames_D.pt")
    ap.add_argument("--out_dir", default="output/vg_ad")
    ap.add_argument("--label_a", default="A_fa2_eager")
    ap.add_argument("--label_d", default="D_hybrid_compile")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    A = torch.load(args.a_cache, map_location="cpu", weights_only=False)
    D = torch.load(args.d_cache, map_location="cpu", weights_only=False)
    fa, fd = A["frames"], D["frames"]
    n = min(len(fa), len(fd))
    print(f"[ad] A={args.a_cache} ({len(fa)} frames, median "
          f"{statistics.median(A['ms'])*1000:.1f} ms)")
    print(f"[ad] D={args.d_cache} ({len(fd)} frames, median "
          f"{statistics.median(D['ms'])*1000:.1f} ms)")
    print(f"[ad] comparing {n} chunks\n")

    def sq(x):
        while x.dim() > 3:
            x = x[0]
        return x

    fa = [sq(x) for x in fa]
    fd = [sq(x) for x in fd]

    dev = "cuda"
    try:
        import lpips as lpips_mod
        lp = lpips_mod.LPIPS(net="alex").to(dev).eval()
    except Exception as e:
        print("[ad] LPIPS unavailable:", e)
        lp = None

    def lpx(a, b):
        if lp is None:
            return float("nan")
        return lp(a.to(dev) * 2 - 1, b.to(dev) * 2 - 1).mean().item()

    rows = []
    for i in range(n):
        a, b = fa[i], fd[i]
        m = dict(chunk=i, psnr=psnr(a, b), ssim=ssim(a[None], b[None]),
                 lpips=lpx(a[None], b[None]))
        m.update({f"rgb_{k}": v for k, v in rgb_stats(a[None], b[None]).items()})
        rows.append(m)

    SEGS = [("0-3", 0, 3), ("7-12", 7, 12), ("13-20", 13, n - 1)]
    print("[ad] ===== A (FA2+eager) vs D (Hybrid+compile), decoded frames =====")
    print("   {:>7} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10}".format(
        "segment", "PSNR", "SSIM", "LPIPS", "|d|mean", "|d|p95", "|d|max"))
    for name, lo, hi in SEGS:
        seg = rows[lo:hi + 1]
        if not seg:
            continue
        print("   {:>7} {:>8.2f} {:>8.4f} {:>8.4f} {:>10.5f} {:>10.5f} {:>10.5f}"
              .format(name,
                      statistics.mean(x["psnr"] for x in seg),
                      statistics.mean(x["ssim"] for x in seg),
                      statistics.mean(x["lpips"] for x in seg),
                      statistics.mean(x["rgb_mean"] for x in seg),
                      statistics.mean(x["rgb_p95"] for x in seg),
                      statistics.mean(x["rgb_max"] for x in seg)))
    print("\n   per-chunk PSNR : " + " ".join(f"{x['psnr']:.1f}" for x in rows))
    print("   per-chunk SSIM : " + " ".join(f"{x['ssim']:.3f}" for x in rows))
    print("   per-chunk LPIPS: " + " ".join(f"{x['lpips']:.4f}" for x in rows))

    def rate(lo, hi):
        if hi <= lo:
            return float("nan")
        return (rows[hi]["lpips"] - rows[lo]["lpips"]) / (hi - lo)
    print(f"\n   LPIPS growth rate: 0-7 {rate(0,7):+.4f}/chunk  "
          f"7-13 {rate(7,13):+.4f}/chunk  13-20 {rate(13,20):+.4f}/chunk")

    # reference points measured earlier, for context (NOT added together)
    print("\n   context (measured separately, NOT additive):")
    print("     A vs B (backend increment)  LPIPS@c20 0.2484  SSIM 0.5621")
    print("     B vs D (compile increment)  LPIPS@c20 0.3243  SSIM 0.4912")

    vis = os.path.join(args.out_dir, "vis")
    os.makedirs(vis, exist_ok=True)
    for i in [0, 8, 11, 16, n - 1]:
        a, b = fa[i], fd[i]
        dm = (b - a).abs()
        dm = (dm / max(dm.max().item(), 1e-6)).clamp(0, 1)
        row = torch.cat([a, b, dm.expand(3, -1, -1)], dim=2)
        torchvision.utils.save_image(row, f"{vis}/chunk{i:02d}_A_D_DIFF.png")
        # also an amplified side-by-side for the steepest region
    print(f"\n[ad] wrote {vis}/chunk*_A_D_DIFF.png  (A | D | |D-A|)")

    json.dump(dict(a_cache=args.a_cache, d_cache=args.d_cache, n_chunks=n,
                   a_median_ms=statistics.median(A["ms"]) * 1000,
                   d_median_ms=statistics.median(D["ms"]) * 1000,
                   segments={k: [lo, hi] for k, lo, hi in SEGS},
                   per_chunk=rows),
              open(f"{args.out_dir}/vg_ad.json", "w"), indent=1, default=str)
    print(f"[ad] wrote {args.out_dir}/vg_ad.json")


if __name__ == "__main__":
    main()
