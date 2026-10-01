#!/usr/bin/env python
"""Preview-1B: the cheap-decode frontier.

The decomposition found the natural early exit: temporal growth means later stages
process MORE frames (1 -> 2 -> 4), so stage 2 runs at the highest spatial resolution
AND at 4 frames, costing ~48% of the decoder on its own. The expensive tail is real.

Two ways to exploit that without training anything:

  B  latent spatially downsampled before decoding, then a cheap upscale
  C  a further downsampled variant

Both are weight-free. Neither is guaranteed to work: the decoder is nonlinear, so a
downsampled latent decodes to a different image rather than to a downsampled one.
That is exactly what the quality measurement is for.

Quality is reported separately for session-start (chunks 1-2) and steady state,
because the earlier measurement showed preview quality is weakest at the start of a
session -- precisely when a user forms their first impression. A cheap decoder that
looks fine in aggregate while collapsing chunk 1 would be worse in product terms.
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
    ap.add_argument("--latents", required=True)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dev = "cuda"
    d = torch.load(args.latents, map_location="cpu", weights_only=False)
    chunks = sorted({int(k[1:].split("_")[0]) for k in d if k.startswith("c")})
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()

    print("=" * 84)
    print("  Preview-1B cheap-decode frontier")
    print("=" * 84)

    def decode_full(z):
        with torch.no_grad():
            return to_img(tae.decode_video(z.permute(1, 0, 2, 3).unsqueeze(0),
                                           parallel=False,
                                           show_progress_bar=False))

    def decode_pooled(z, k, target_hw):
        """Downsample the latent spatially, decode, then upscale to display size.

        z is [C, T, H, W]; avg_pool2d reads it as [B=C, C=T, H, W] and pools the
        spatial dims, which is exactly the intended reduction.
        """
        zz = F.avg_pool2d(z, k) if k > 1 else z
        img = decode_full(zz)
        if k > 1:
            img = F.interpolate(img.unsqueeze(0), size=target_hw, mode="bicubic",
                                align_corners=False).squeeze(0).clamp(0, 1)
        return img

    # ---- D: skip the LAST spatial upsample, keeping every learned conv ----
    # Weight-free and much less invasive than downsampling the latent: the network
    # still sees an in-distribution input, it simply produces a frame at half the
    # spatial resolution which is then upscaled. The decomposition says the stages
    # after that upsample are the expensive tail.
    import torch.nn as nn
    dec = tae.decoder
    up_idx = None
    for i, m in enumerate(dec):
        if type(m).__name__ == "Upsample":
            up_idx = i
    assert up_idx is not None, "no Upsample child found in the decoder"
    _orig_up = dec[up_idx]

    def decode_skip_upsample(z, target_hw):
        dec[up_idx] = nn.Identity()
        try:
            img = decode_full(z)
        finally:
            dec[up_idx] = _orig_up
        return F.interpolate(img.unsqueeze(0), size=target_hw, mode="bicubic",
                             align_corners=False).squeeze(0).clamp(0, 1)

    # ---- timing ----
    z0 = d[f"c{chunks[0]}_step0"].to(dev)
    tgt = decode_full(z0).shape[-2:]
    variants = [("A full TAE", lambda z: decode_full(z)),
                ("B latent /2 + upscale", lambda z: decode_pooled(z, 2, tgt)),
                ("C latent /4 + upscale", lambda z: decode_pooled(z, 4, tgt)),
                ("D skip last upsample", lambda z: decode_skip_upsample(z, tgt))]
    times = {}
    print(f"  {'variant':<26} {'ms':>8} {'vs A':>9}")
    print("  " + "-" * 46)
    base = None
    for name, fn in variants:
        for _ in range(3):
            fn(z0)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(args.reps):
            fn(z0)
        e.record(); torch.cuda.synchronize()
        t = s.elapsed_time(e) / args.reps
        times[name] = t
        if base is None:
            base = t
        print(f"  {name:<26} {t:>8.2f} {(t-base)/base*100:>+8.1f}%")

    # ---- quality, split by session position ----
    # THE REFERENCE IS THE AUTHORITATIVE FINAL FRAME, not the full decode of the
    # same step0 latent. Comparing cheap(step0) against full(step0) measures decoder
    # fidelity, which is not what a preview is for. The product question is whether
    # a cheap preview still tells the user the world is moving in the right
    # direction, so the yardstick is the same one Preview-1A used:
    # preview(step0) against the authoritative final frame.
    print()
    print("  QUALITY vs the AUTHORITATIVE FINAL frame, split by session position")
    print(f"  {'chunk':<7} {'full A':>8} {'B /2':>8} {'C /4':>8} {'D skipup':>9} "
          f"{'A edge':>8} {'B edge':>8} {'C edge':>8} {'D edge':>8}")
    print("  " + "-" * 80)
    per = []
    for c in chunks:
        z0c = d[f"c{c}_step0"].to(dev)
        zf = d[f"c{c}_final"].to(dev)
        ref = decode_full(zf)                       # authoritative final
        row = dict(chunk=c)
        a = decode_full(z0c)
        row["A_ssim"] = ssim(a, ref)
        row["A_edge"] = ssim(edge_map(a).repeat(3, 1, 1),
                             edge_map(ref).repeat(3, 1, 1))
        for nm, fn in (("B", lambda z: decode_pooled(z, 2, tgt)),
                       ("C", lambda z: decode_pooled(z, 4, tgt)),
                       ("D", lambda z: decode_skip_upsample(z, tgt))):
            img = fn(z0c)
            row[f"{nm}_ssim"] = ssim(img, ref)
            row[f"{nm}_edge"] = ssim(edge_map(img).repeat(3, 1, 1),
                                     edge_map(ref).repeat(3, 1, 1))
        per.append(row)
        print(f"  {c:<7} {row['A_ssim']:>8.4f} {row['B_ssim']:>8.4f} "
              f"{row['C_ssim']:>8.4f} {row['D_ssim']:>9.4f} "
              f"{row['A_edge']:>8.4f} {row['B_edge']:>8.4f} "
              f"{row['C_edge']:>8.4f} {row['D_edge']:>8.4f}")

    def agg(key, subset):
        v = [r[key] for r in subset]
        return statistics.median(v) if v else float("nan")

    start = [r for r in per if r["chunk"] <= 2]
    steady = [r for r in per if r["chunk"] > 2]
    print()
    print(f"  {'window':<12} {'variant':<6} {'ssim p50':>10} {'edgeSSIM p50':>14}")
    print("  " + "-" * 46)
    for label, sub in (("session-start", start), ("steady-state", steady),
                       ("all", per)):
        for nm in ("B", "C", "D"):
            print(f"  {label:<12} {nm:<6} {agg(nm+'_ssim', sub):>10.4f} "
                  f"{agg(nm+'_edge', sub):>14.4f}")
    print()
    print("  NOTE: A's ssim column is 1.0000 by construction (it is the reference);")
    print("        the interesting columns are B and C against it.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/cheap_decode.json", "w") as f:
        json.dump(dict(times_ms=times, per_chunk=per), f, indent=2)
    print(f"\n  wrote {args.out_dir}/cheap_decode.json")


if __name__ == "__main__":
    main()
