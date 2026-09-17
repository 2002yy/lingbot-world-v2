#!/usr/bin/env python
"""§43P-0: does a multi-view anchor bank restore the identity margin?

Motivation (from §42):
  * local representation margin degrades with object count
    (identity_margin mean 0.499 -> 0.148; p10 -0.067; min -0.175 at N=10)
  * global one-to-one assignment currently absorbs that ambiguity
    (assignment_margin min +0.950), but that compensation should not be
    assumed to scale indefinitely
  * §41B-3b already showed latent-space rotation is NOT geometrically valid
    (un-rotating the rendered patch does not recover the 0-degree reference)

Section §42C-2/-3 measured the viewpoint tolerance with a SINGLE anchor:
    dYaw  0 deg -> self 0.776, margin +0.499
    dYaw  3 deg -> self 0.599  (falls through the 0.60 existence threshold)
    dYaw  5 deg -> margin +0.008 (margin collapses toward 0)
    dYaw  7 deg -> self 0.469

§43P asks whether a VIEW-CONDITIONED representation fixes that. The cheapest
decisive test needs no model run: tolerance_sweep.py already saved rendered
frames for dYaw = 0..7, so we can compare

    single-anchor : match against the dYaw=0 template only
    multi-anchor  : match against a bank captured at several dYaw

and see whether self-similarity / margin / existence recover under viewpoint
change. Anchors are taken from the BASELINE chunks of each run so the template
and the measured appearance share a render path (§42C-3's lesson).

  python multiview_anchor.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from object_permanence import patch  # noqa: E402

OBJECTS = {
    "A_move":   (0.28, 0.56, 0.52, 0.90),
    "B_health": (0.68, 0.42, 0.88, 0.72),
    "C_fov":    (0.80, 0.10, 0.94, 0.40),
    "D_occl":   (0.06, 0.66, 0.22, 0.94),
    "E_ctrl":   (0.42, 0.70, 0.56, 0.94),
}
TARGET = "C_fov"
ID_EXIST = 0.60
BANK_VIEWS = [0.0, 2.0, 4.0, 6.0]      # dYaw at which bank anchors are captured


def main():
    from eval_two_layer import load_models, dino_feat
    load_models()

    def feat(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    files = {}
    for f in glob.glob("output/tolerance/frames_*.npy"):
        ang = float(os.path.basename(f).replace("frames_", "").replace(".npy", ""))
        files[ang] = f
    if not files:
        print("[mv] no tolerance frames; run tolerance_sweep.py first")
        return
    angles = sorted(files)
    print(f"[mv] available dYaw: {angles}")

    # Anchors must be captured AT EACH VIEW. Taking them from the baseline chunk
    # is wrong: every run shares the same baseline camera pose (only the RETURN
    # pose differs by dYaw), so all "anchors" would be identical and the bank
    # would be a no-op -- which is exactly what the first attempt measured.
    templates = {}
    for ang in angles:
        F = np.load(files[ang])
        templates[ang] = {nm: patch(F[-1], bb) for nm, bb in OBJECTS.items()}
    print(f"[mv] anchor bank views: {[a for a in BANK_VIEWS if a in templates]}")
    # sanity: are the bank anchors actually different from each other?
    base = templates[angles[0]][TARGET]
    for b in BANK_VIEWS:
        if b in templates and b != angles[0]:
            d = float(np.abs(templates[b][TARGET].astype(float)
                             - base.astype(float)).mean())
            print(f"[mv]   anchor {angles[0]:.1f} vs {b:.1f}: mean|diff| {d:.2f}")

    def self_other(crop_feat, anchors):
        s = max(float((crop_feat * feat(a[TARGET])).sum()) for a in anchors)
        o = max(float((crop_feat * feat(a[nm])).sum())
                for a in anchors for nm in OBJECTS if nm != TARGET)
        return s, o

    print(f"\n[mv] ===== single anchor vs multi-view bank =====")
    print(f"  {'dYaw':>5s} | {'single self':>11s} {'margin':>8s} {'exist':>6s} | "
          f"{'multi self':>10s} {'margin':>8s} {'exist':>6s}")
    rows = []
    for ang in angles:
        F = np.load(files[ang])
        n = len(F)
        # test on the post-return window
        s1, s2, m1, m2 = [], [], [], []
        for cid in range(max(0, n - 8), n):
            crop = patch(F[cid], OBJECTS[TARGET])
            cf = feat(crop)
            # single: only the dYaw=0 anchor
            a0 = [templates[0.0]]
            s_s, o_s = self_other(cf, a0)
            # multi: the bank (leaving-one-out is NOT done -- this is the
            # deployable setting where the bank is pre-registered)
            bank = [templates[b] for b in BANK_VIEWS if b in templates]
            s_m, o_m = self_other(cf, bank)
            s1.append(s_s); m1.append(s_s - o_s)
            s2.append(s_m); m2.append(s_m - o_m)
        r = dict(angle=ang, single_self=float(np.mean(s1)),
                 single_margin=float(np.mean(m1)),
                 single_exist=float(np.mean(np.array(s1) >= ID_EXIST)),
                 multi_self=float(np.mean(s2)),
                 multi_margin=float(np.mean(m2)),
                 multi_exist=float(np.mean(np.array(s2) >= ID_EXIST)))
        rows.append(r)
        print(f"  {ang:5.1f} | {r['single_self']:11.3f} "
              f"{r['single_margin']:+8.3f} {r['single_exist']*100:5.0f}% | "
              f"{r['multi_self']:10.3f} {r['multi_margin']:+8.3f} "
              f"{r['multi_exist']*100:5.0f}%")

    s1 = np.array([r["single_self"] for r in rows])
    s2 = np.array([r["multi_self"] for r in rows])
    m1 = np.array([r["single_margin"] for r in rows])
    m2 = np.array([r["multi_margin"] for r in rows])
    print(f"\n[mv] ===== summary =====")
    print(f"  single anchor : self {s1.mean():.3f} (min {s1.min():.3f})  "
          f"margin {m1.mean():+.3f} (min {m1.min():+.3f})  "
          f"exist-pass {np.mean(s1>=ID_EXIST)*100:.0f}%")
    print(f"  multi bank    : self {s2.mean():.3f} (min {s2.min():.3f})  "
          f"margin {m2.mean():+.3f} (min {m2.min():+.3f})  "
          f"exist-pass {np.mean(s2>=ID_EXIST)*100:.0f}%")
    print(f"\n  self   delta  : {s2.mean()-s1.mean():+.3f}")
    print(f"  margin delta  : {m2.mean()-m1.mean():+.3f}")
    print(f"  worst-case self: single {s1.min():.3f} -> multi {s2.min():.3f}")

    # viewpoint range where each stays above the existence threshold
    def max_ok(vals, thr=ID_EXIST):
        ok = [r["angle"] for r, v in zip(rows, vals) if v >= thr]
        return max(ok) if ok else None
    print(f"\n  tolerance (self >= {ID_EXIST}) : single "
          f"{max_ok(s1)} deg  |  multi {max_ok(s2)} deg")

    verdict = ("multi-view anchor bank RESTORES margin"
               if m2.mean() > m1.mean() + 0.05 else
               "multi-view anchor bank does NOT clearly help")
    print(f"\n  §43P-0 verdict: {verdict}")
    json.dump(dict(rows=rows, bank_views=[b for b in BANK_VIEWS if b in templates],
                   single_self_mean=float(s1.mean()),
                   multi_self_mean=float(s2.mean()),
                   single_margin_mean=float(m1.mean()),
                   multi_margin_mean=float(m2.mean()),
                   verdict=verdict),
              open("output/tolerance/multiview.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
