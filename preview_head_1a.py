#!/usr/bin/env python
"""§Preview-Head-1A: can a 4-9 ms tail replace TAEHV's ~21 ms learned tail?

TWO OBJECTIVES, deliberately separated, because bundling them would conflate two
different questions:

    Arm T  stage1_feature(step0) -> full TAEHV(step0) RGB
           capacity gate: how small can the second half of the decoder be made?
    Arm F  stage1_feature(step0) -> authoritative FINAL RGB
           product value: is it better if the tail also predicts what the DiT has
           not produced yet?

Same architecture, same samples, same latency budget for both arms, so the
comparison is interpretable.

PREFIX IS FROZEN. TAEHV stage 0+1 weights are never fine-tuned. Letting the prefix
train with the tail would produce prettier loss curves while destroying the property
that matters -- a stable prefix plus a pluggable tail -- and would reintroduce model
version and weight compatibility surface.

Capacity is defined by MEASURED warm CUDA latency, not by parameter count: the budget
is prefix 10.84 ms, so a 15 ms total allows a ~4.2 ms tail and a 20 ms total allows
~9.2 ms. Three sizes are tried at those two boundaries.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

MS = 1e6
ATTACH_CHILD = 14          # stage 1's conv; output [B, 64, 264, 152]


# --------------------------------------------------------------------- helpers
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


# ----------------------------------------------------------------------- tails
def conv(i, o, k=3):
    return nn.Conv2d(i, o, k, padding=k // 2, bias=False)


class Tail(nn.Module):
    """64ch @ 264x152 -> 3ch @ 528x304. PixelShuffle for the 2x, then cheap convs."""

    def __init__(self, cin=64, width=32, blocks=1):
        super().__init__()
        layers = [conv(cin, width * 4), nn.ReLU(inplace=True),
                  nn.PixelShuffle(2)]
        for _ in range(blocks):
            layers += [conv(width, width), nn.ReLU(inplace=True)]
        layers += [conv(width, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build(width, blocks):
    return Tail(64, width, blocks)


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dev = "cuda"
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    for p in tae.parameters():
        p.requires_grad_(False)                      # PREFIX FROZEN
    dec = tae.decoder

    # ---- collect stage-1 features and both targets ----
    d = torch.load(args.latents, map_location="cpu", weights_only=False)
    chunks = sorted({int(k[1:].split("_")[0]) for k in d if k.startswith("c")})
    print("=" * 88)
    print(f"  Preview-Head-1A   samples={len(chunks)}  steps={args.steps}")
    print("=" * 88)

    hold = {}

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(t):
            hold["f"] = t.detach()
    h = dec[ATTACH_CHILD].register_forward_hook(hook)

    def prefix(z):
        """Run stages 0+1 only, returning the attach-point feature."""
        x = z.permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad():
            tae.decode_video(x, parallel=False, show_progress_bar=False)
        return hold["f"]

    feats, tgt_T, tgt_F, first_flag = [], [], [], []
    for c in chunks:
        z0 = d[f"c{c}_step0"].to(dev)
        zf = d[f"c{c}_final"].to(dev)
        feats.append(prefix(z0))
        with torch.no_grad():
            tgt_T.append(to_img(tae.decode_video(
                z0.permute(1, 0, 2, 3).unsqueeze(0), parallel=False,
                show_progress_bar=False)))
            tgt_F.append(to_img(tae.decode_video(
                zf.permute(1, 0, 2, 3).unsqueeze(0), parallel=False,
                show_progress_bar=False)))
        first_flag.append(c <= 2)
    h.remove()
    print(f"  feature {list(feats[0].shape)}   T target {list(tgt_T[0].shape)}")
    print(f"  session-start chunks: {[c for c, f in zip(chunks, first_flag) if f]}")
    print()

    X = torch.cat(feats, 0)          # features stack along batch
    YT = torch.stack(tgt_T, 0)       # targets are [3,H,W] each -> [N,3,H,W]
    YF = torch.stack(tgt_F, 0)
    print(f"  tensors: X {list(X.shape)}  YT {list(YT.shape)}  YF {list(YF.shape)}")
    print()

    # ---- the latency/quality frontier ----
    print("=" * 88)
    print("  TAIL LATENCY (warm, CUDA events) and QUALITY")
    print("=" * 88)
    sizes = [("S", 16, 1), ("M", 32, 1), ("L", 64, 2)]
    results = {}
    for obj, Y, tname in (("T", YT, "full TAEHV(step0)"), ("F", YF, "final")):
        print()
        print(f"  --- Arm {obj}: target = {tname} ---")
        print(f"  {'size':<6} {'width':>6} {'blocks':>7} {'params':>9} "
              f"{'tail ms':>9} {'total ms':>9} {'ssim':>8} {'edge':>8} "
              f"{'start ssim':>11}")
        print("  " + "-" * 78)
        for name, width, blocks in sizes:
            net = build(width, blocks).to(dev)
            opt = torch.optim.Adam(net.parameters(), lr=args.lr)
            for step in range(args.steps):
                pred = net(X)
                loss = F.mse_loss(pred, Y)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()

            with torch.no_grad():
                pred = net(X).clamp(0, 1)
            ss = [ssim(pred[i], Y[i]) for i in range(len(chunks))]
            es = [ssim(edge_map(pred[i]).repeat(3, 1, 1),
                       edge_map(Y[i]).repeat(3, 1, 1)) for i in range(len(chunks))]
            st = [ss for ss, f in zip(ss, first_flag) if f]

            for _ in range(5):
                net(X[:1])
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(args.reps):
                net(X[:1])
            e.record(); torch.cuda.synchronize()
            t_ms = s.elapsed_time(e) / args.reps

            npar = sum(p.numel() for p in net.parameters())
            results[f"{obj}_{name}"] = dict(
                arm=obj, size=name, width=width, blocks=blocks, params=npar,
                tail_ms=t_ms, total_ms=10.84 + t_ms,
                ssim=statistics.mean(ss), edge_ssim=statistics.mean(es),
                start_ssim=statistics.mean(st) if st else None,
                per_chunk_ssim=ss)
            print(f"  {name:<6} {width:>6} {blocks:>7} {npar:>9} {t_ms:>9.2f} "
                  f"{10.84+t_ms:>9.2f} {statistics.mean(ss):>8.4f} "
                  f"{statistics.mean(es):>8.4f} "
                  f"{(statistics.mean(st) if st else float('nan')):>11.4f}")

    # ---- reference points ----
    print()
    print("=" * 88)
    print("  REFERENCE")
    print("=" * 88)
    print(f"  full TAEHV preview(step0) vs final:  "
          f"ssim {statistics.mean(ssim(tgt_T[i], tgt_F[i]) for i in range(len(chunks))):.4f}"
          f"  edge {statistics.mean(ssim(edge_map(tgt_T[i]).repeat(3,1,1), edge_map(tgt_F[i]).repeat(3,1,1)) for i in range(len(chunks))):.4f}")
    print(f"  prefix alone                        10.84 ms")
    print(f"  TAEHV stage2 + final (what a tail replaces)  "
          f"{17.31 + 3.42:.2f} ms")
    print(f"  full TAEHV                          36.01 ms")

    print()
    print("  GATES")
    print(f"    Arm T capacity: head vs teacher ssim >= 0.95, edge >= 0.90")
    print(f"    runtime: prefix + tail <= 20 ms (prefer <= 15)")
    print(f"    product: head vs final within 0.03-0.05 of 0.8268 / 0.8003")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/head_1a.json", "w") as f:
        json.dump(dict(chunks=chunks, results=results,
                       prefix_ms=10.84, stage2_plus_final_ms=20.73,
                       full_taehv_ms=36.01), f, indent=2)
    print(f"\n  wrote {args.out_dir}/head_1a.json")


if __name__ == "__main__":
    main()
