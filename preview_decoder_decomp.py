#!/usr/bin/env python
"""Preview-1B step 1: decompose the 36.7 ms TAE preview decode.

Not a big profiler. The question is whether there is a natural early exit: if one
late high-resolution stage dominates, truncating there is the cheap win; if the cost
is spread evenly, truncation buys little and the frontier lies elsewhere.

The TAEHV decoder is an nn.Sequential with three stages, each being
MemBlock x3 -> spatial upsample -> temporal grow -> conv, then a final ReLU+conv.
Children are timed individually, and their sum is reconciled against the total so an
unattributed remainder cannot hide.
"""
import argparse
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

MS = 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--latent", required=True,
                    help="a saved step0 latent, [C, T, H, W]")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    dev = "cuda"
    obj = torch.load(args.latent, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        keys = [k for k in sorted(obj) if k.endswith("_step0")]
        if not keys:
            keys = sorted(obj)
        print(f"  latent file is a dict; using {keys[0]!r} of "
              f"{len(obj)} entries")
        obj = obj[keys[0]]
    z = obj.to(dev)
    print("=" * 84)
    print(f"  TAE decoder internal decomposition   latent {list(z.shape)} "
          f"{z.dtype}")
    print("=" * 84)

    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    dec = tae.decoder
    print(f"  decoder is {type(dec).__name__} with {len(dec)} children:")
    for i, m in enumerate(dec):
        print(f"    [{i:>2}] {type(m).__name__}")
    print()

    x = z.permute(1, 0, 2, 3).unsqueeze(0)

    def run_full():
        with torch.no_grad():
            return tae.decode_video(x, parallel=False, show_progress_bar=False)

    for _ in range(3):
        run_full()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(args.reps):
        run_full()
    e.record(); torch.cuda.synchronize()
    total = s.elapsed_time(e) / args.reps

    # Per-child exclusive time via hooks + CUDA events. Hooks avoid reimplementing
    # the MemBlock memory semantics, and CUDA events record asynchronously so no
    # per-child synchronisation is needed; all pairs are read once at the end.
    kids = list(dec)
    starts = [[] for _ in kids]
    ends = [[] for _ in kids]

    def mk_pre(i):
        def h(mod, inp):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            starts[i].append(ev)
        return h

    def mk_post(i):
        def h(mod, inp, out):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            ends[i].append(ev)
        return h

    hs = []
    for i, m in enumerate(kids):
        hs.append(m.register_forward_pre_hook(mk_pre(i)))
        hs.append(m.register_forward_hook(mk_post(i)))

    for _ in range(2):
        run_full()
    torch.cuda.synchronize()
    for i in range(len(kids)):
        starts[i].clear(); ends[i].clear()

    s.record()
    for _ in range(args.reps):
        run_full()
    e.record(); torch.cuda.synchronize()
    total_hooked = s.elapsed_time(e) / args.reps
    for h in hs:
        h.remove()

    rows = []
    for i, m in enumerate(kids):
        n = min(len(starts[i]), len(ends[i]))
        if n == 0:
            rows.append(dict(idx=i, name=type(m).__name__, calls=0,
                             exclusive_ms=0.0, pct=0.0))
            continue
        ms = sum(starts[i][k].elapsed_time(ends[i][k]) for k in range(n)) / \
            args.reps
        rows.append(dict(idx=i, name=type(m).__name__, calls=n // args.reps,
                         exclusive_ms=ms, pct=ms / total * 100.0))

    print(f"  {'idx':>4} {'module':<24} {'exclusive':>10} {'pct':>7} {'calls':>6}")
    print("  " + "-" * 58)
    for r in rows:
        print(f"  {r['idx']:>4} {r['name']:<24} {r['exclusive_ms']:>10.2f} "
              f"{r['pct']:>6.1f}% {r['calls']:>6}")
    print("  " + "-" * 58)
    acc = sum(r["exclusive_ms"] for r in rows)
    print(f"  {'':>4} {'TOTAL (wall, unhooked)':<24} {total:>10.2f}")
    print(f"  {'':>4} {'TOTAL (wall, hooked)':<24} {total_hooked:>10.2f}")
    print(f"  {'':>4} {'sum of exclusive':<24} {acc:>10.2f}")
    print(f"  {'':>4} {'unattributed':<24} {total - acc:>10.2f}")

    print()
    print("  INTERPRETATION")
    print("    a late child dominating means a natural early exit exists;")
    print("    an even spread means truncation buys little and the frontier is")
    print("    elsewhere (lower-resolution decode, or a different decoder).")
    late = sum(r["exclusive_ms"] for r in rows[len(rows) // 2:])
    print(f"    second half of the pipeline: {late:.2f} ms "
          f"({late/total*100:.1f}% of the total)")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/decoder_decomp.json", "w") as f:
        json.dump(dict(total_ms=total, rows=rows), f, indent=2)
    print(f"\n  wrote {args.out_dir}/decoder_decomp.json")


if __name__ == "__main__":
    main()
