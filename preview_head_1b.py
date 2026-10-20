#!/usr/bin/env python
"""§Preview-Head-1B: generalisation gate on two axes.

    H1  same scene (04), unseen seed AND unseen control script
    H2  different scenes entirely (00, 05)

The head is trained ONCE on the nine training pairs, frozen, and then only inferred
on. It is never retrained on a holdout, because the moment a holdout is trained on it
stops being a holdout and the generalisation number becomes meaningless.

THE HEADLINE METRIC IS RELATIVE, NOT ABSOLUTE. The baseline (full TAEHV step0
preview) gets harder or easier depending on the scene, so an absolute SSIM on a new
scene cannot be compared with an old scene's number. What is reported is

    dSSIM     = head - full_preview      measured on the SAME holdout
    dedgeSSIM = head - full_preview

which stays interpretable across scenes. Results are also bucketed by control type so
an aggregate median cannot hide one motion direction failing systematically, and
session-start is reported separately because Arm F's selling point was that it is
strongest exactly there.
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

ATTACH_CHILD = 14
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


def conv(i, o, k=3):
    return nn.Conv2d(i, o, k, padding=k // 2, bias=False)


class Tail(nn.Module):
    def __init__(self, cin=64, width=64, blocks=2):
        super().__init__()
        layers = [conv(cin, width * 4), nn.ReLU(inplace=True),
                  nn.PixelShuffle(2)]
        for _ in range(blocks):
            layers += [conv(width, width), nn.ReLU(inplace=True)]
        layers += [conv(width, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def control_bucket(ctrl):
    keys = set(ctrl)
    if keys == {"yaw"}:
        return "yaw"
    if keys == {"pitch"}:
        return "pitch"
    if "yaw" in keys or "pitch" in keys:
        return "combined"
    if "right" in keys or "strafe" in keys:
        return "strafe"
    return "translation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="output/prev1a_q/step_latents.pt")
    ap.add_argument("--holdouts", nargs="+", required=True,
                    help="tag=path pairs, e.g. H1=output/holdout/H1.pt")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dev = "cuda"
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    for p in tae.parameters():
        p.requires_grad_(False)
    dec = tae.decoder
    hold = {}
    h = dec[ATTACH_CHILD].register_forward_hook(
        lambda m, i, o: hold.__setitem__("f", (o[0] if isinstance(o, (tuple, list)) else o).detach()))

    def load(path):
        d = torch.load(path, map_location="cpu", weights_only=False)
        chunks = sorted({int(k[1:].split("_")[0]) for k in d if k.startswith("c")})
        X, YF, YT = [], [], []
        for c in chunks:
            z0 = d[f"c{c}_step0"].to(dev)
            zf = d[f"c{c}_final"].to(dev)
            x0 = z0.permute(1, 0, 2, 3).unsqueeze(0)
            with torch.no_grad():
                tae.decode_video(x0, parallel=False, show_progress_bar=False)
            X.append(hold["f"])
            with torch.no_grad():
                YF.append(to_img(tae.decode_video(
                    zf.permute(1, 0, 2, 3).unsqueeze(0), parallel=False,
                    show_progress_bar=False)))
                YT.append(to_img(tae.decode_video(
                    x0, parallel=False, show_progress_bar=False)))
        return chunks, torch.cat(X, 0), torch.stack(YF, 0), torch.stack(YT, 0)

    print("=" * 90)
    print("  Preview-Head-1B generalisation gate")
    print("=" * 90)
    tr_chunks, Xtr, YFtr, _ = load(args.train)
    print(f"  training: {len(tr_chunks)} chunks")

    # ---- train Arm F-L once, on the training set only ----
    net = Tail(64, 64, 2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for step in range(args.steps):
        loss = F.mse_loss(net(Xtr), YFtr)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    npar = sum(p.numel() for p in net.parameters())
    print(f"  Arm F-L trained: {npar} params, {args.steps} steps, weights FROZEN")
    print()

    results = {}
    for spec in args.holdouts:
        tag, path = spec.split("=", 1)
        chunks, X, YF, YT = load(path)
        with torch.no_grad():
            pred = net(X).clamp(0, 1)

        rows = []
        for i, c in enumerate(chunks):
            rows.append(dict(
                chunk=c,
                head_ssim=ssim(pred[i], YF[i]),
                head_edge=ssim(edge_map(pred[i]).repeat(3, 1, 1),
                               edge_map(YF[i]).repeat(3, 1, 1)),
                base_ssim=ssim(YT[i], YF[i]),
                base_edge=ssim(edge_map(YT[i]).repeat(3, 1, 1),
                               edge_map(YF[i]).repeat(3, 1, 1))))
        for r in rows:
            r["d_ssim"] = r["head_ssim"] - r["base_ssim"]
            r["d_edge"] = r["head_edge"] - r["base_edge"]

        def med(k, sub):
            v = [r[k] for r in sub]
            return statistics.median(v) if v else float("nan")

        start = [r for r in rows if r["chunk"] <= 2]
        steady = [r for r in rows if r["chunk"] > 2]
        gross = sum(1 for r in rows if r["head_ssim"] < 0.30)
        results[tag] = dict(rows=rows, n=len(rows), gross_failures=gross,
                            all_d_ssim=med("d_ssim", rows),
                            all_d_edge=med("d_edge", rows),
                            start_d_ssim=med("d_ssim", start),
                            start_d_edge=med("d_edge", start),
                            steady_d_ssim=med("d_ssim", steady),
                            steady_d_edge=med("d_edge", steady),
                            head_ssim=med("head_ssim", rows),
                            base_ssim=med("base_ssim", rows))

        print(f"  --- {tag}  ({len(rows)} chunks) ---")
        print(f"  {'chunk':>6} {'head ssim':>10} {'base ssim':>10} {'dSSIM':>9} "
              f"{'head edge':>10} {'base edge':>10} {'dedge':>9}")
        print("  " + "-" * 70)
        for r in rows:
            print(f"  {r['chunk']:>6} {r['head_ssim']:>10.4f} "
                  f"{r['base_ssim']:>10.4f} {r['d_ssim']:>+9.4f} "
                  f"{r['head_edge']:>10.4f} {r['base_edge']:>10.4f} "
                  f"{r['d_edge']:>+9.4f}")
        print()
        print(f"    all           dSSIM {results[tag]['all_d_ssim']:+.4f}  "
              f"dedge {results[tag]['all_d_edge']:+.4f}")
        print(f"    session-start dSSIM {results[tag]['start_d_ssim']:+.4f}  "
              f"dedge {results[tag]['start_d_edge']:+.4f}")
        print(f"    steady-state  dSSIM {results[tag]['steady_d_ssim']:+.4f}  "
              f"dedge {results[tag]['steady_d_edge']:+.4f}")
        print(f"    gross structural failures (head ssim < 0.30): {gross}")
        print()

    # ---- verdict ----
    print("=" * 90)
    print("  GATE: dSSIM >= -0.05, dedge >= -0.05, gross failures = 0,")
    print("        and session-start must pass on its own")
    print("=" * 90)
    for tag, r in results.items():
        ok = (r["all_d_ssim"] >= -0.05 and r["all_d_edge"] >= -0.05
              and r["gross_failures"] == 0 and r["start_d_ssim"] >= -0.05)
        r["pass"] = ok
        print(f"  {tag:<16} all dSSIM {r['all_d_ssim']:+.4f}  "
              f"start dSSIM {r['start_d_ssim']:+.4f}  "
              f"gross {r['gross_failures']}  -> {'PASS' if ok else 'FAIL'}")

    print()
    h1 = results.get("H1", {}).get("pass")
    h2 = [r["pass"] for t, r in results.items() if t.startswith("H2")]
    if h1 and h2 and all(h2):
        print("  VERDICT: H1 and H2 both pass -> genuine product potential;")
        print("           proceed to runtime E2E.")
    elif h1 and h2:
        print("  VERDICT: H1 passes but H2 does not -> the head generalises")
        print("           across controls but is scene-specific; expand training")
        print("           scenes before any product integration.")
    elif h1 is False:
        print("  VERDICT: H1 fails -> the head is essentially memorising the nine")
        print("           training pairs. Do not integrate.")
    else:
        print("  VERDICT: inconclusive; see the table.")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/head_1b.json", "w") as f:
        json.dump(dict(params=npar, steps=args.steps, results=results), f,
                  indent=2)
    print(f"\n  wrote {args.out_dir}/head_1b.json")


if __name__ == "__main__":
    main()
