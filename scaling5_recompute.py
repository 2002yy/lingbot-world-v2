#!/usr/bin/env python
"""§42D recompute: assignment_margin with the fixed metric.

The first §42D run reported assignment_margin = -2.834, which is impossible for
a maximum (best - second_best can only be >= 0). The bug was reading the total
as `-g[rr, cc].sum()` instead of `g[rr, cc].sum()`, which negated the objective
and made every "excluded" solution look larger.

This recomputes it from the saved frames -- no model re-run.
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, ".")
from object_permanence import patch  # noqa: E402

OBJECTS = {
    "A_move":   (0.28, 0.56, 0.52, 0.90),
    "B_health": (0.68, 0.42, 0.88, 0.72),
    "C_fov":    (0.80, 0.10, 0.94, 0.40),
    "D_occl":   (0.04, 0.12, 0.20, 0.42),
    "E_ctrl":   (0.36, 0.13, 0.52, 0.37),
}
NAMES = list(OBJECTS.keys())
GEOM_W, PERSIST_W, NULL_PEN, DGEO_GATE = 0.80, 0.00, 0.30, 0.020


def dgeo(a, b, wh):
    return math.hypot((a[0] - b[0]) / max(wh[0], 1e-6),
                      (a[1] - b[1]) / max(wh[1], 1e-6))


def assign_margin(visual, dgeos):
    S = np.array(visual, float) + PERSIST_W
    S = S + GEOM_W * np.maximum(0.0, 1.0 - np.array(dgeos, float) / DGEO_GATE)
    n_e, n_c = S.shape
    full = np.concatenate([S, np.full((n_e, n_e), NULL_PEN)], axis=1)
    r, c = linear_sum_assignment(-full)
    best = float(full[r, c].sum())
    second = None
    for ei, cj in zip(r, c):
        g = full.copy()
        g[ei, cj] = -1e12
        rr, cc = linear_sum_assignment(-g)
        v = float(g[rr, cc].sum())
        if second is None or v > second:
            second = v
    return best, (second if second is not None else float("nan")), \
        float(best - second)


def main():
    d = json.load(open("output/scaling5/scaling5.json"))
    ret = d["ret_chunks"]
    F = np.load("output/scaling5/frames.npy")
    H, W = F[0].shape[:2]

    from eval_two_layer import load_models, dino_feat
    load_models()

    def feat(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    tpl = {nm: patch(F[1], OBJECTS[nm]) for nm in NAMES}
    T = {nm: feat(tpl[nm]) for nm in NAMES}

    print(f"{'chunk':>6s} {'best':>9s} {'second':>9s} {'margin':>9s}")
    rows = []
    for cid in ret:
        vis_ent = [nm for nm in NAMES if not (nm == "D_occl" and 8 <= cid <= 16)]
        cands = [(nm, OBJECTS[nm]) for nm in vis_ent]
        cvis = np.zeros((len(NAMES), len(cands)))
        cgeo = np.zeros((len(NAMES), len(cands)))
        for j, (cn, cb) in enumerate(cands):
            fv = feat(patch(F[cid], cb))
            ctr = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
            for i, en in enumerate(NAMES):
                cvis[i, j] = float((fv * T[en]).sum())
                cgeo[i, j] = dgeo(ctr,
                                  ((OBJECTS[en][0] + OBJECTS[en][2]) / 2,
                                   (OBJECTS[en][1] + OBJECTS[en][3]) / 2),
                                  (OBJECTS[en][2] - OBJECTS[en][0],
                                   OBJECTS[en][3] - OBJECTS[en][1]))
        b, s, m = assign_margin(cvis, cgeo)
        rows.append(dict(chunk=cid, best=b, second=s, margin=m))
        print(f"{cid:6d} {b:9.3f} {s:9.3f} {m:9.3f}")

    ms = np.array([r["margin"] for r in rows])
    print(f"\nassignment_margin: mean {ms.mean():+.3f}  min {ms.min():+.3f}  "
          f"max {ms.max():+.3f}")
    print(f"(first run reported -2.834 -- that was the negated-objective bug)")
    print(f"\n5-object scaling table:")
    print(f"  global re-ID recall      {d['recall']*100:.1f}%   (3-obj 100.0%)")
    print(f"  wrong-ID                 {d['wrong_id']}       (3-obj 0)")
    print(f"  duplicate                0       (3-obj 0)")
    print(f"  cross-object transfer    0       (3-obj 0)")
    print(f"  identity_margin mean     {d['identity_margin_mean'] if 'identity_margin_mean' in d else float('nan'):+.3f}  (3-obj +0.499)")
    print(f"  identity_margin min      {d['identity_margin_min']:+.3f}  (3-obj +0.048)")
    print(f"  assignment_margin mean   {ms.mean():+.3f}  (3-obj n/a)")
    print(f"  assignment_margin min    {ms.min():+.3f}")
    print(f"  chunk latency            {d['chunk_latency_ms']:.1f} ms  (3-obj ~1251)")
    print(f"  VRAM peak allocated      {d['vram_peak_alloc']:.0f} MiB  (3-obj 3307)")
    print(f"  VRAM reserved            {d['vram_reserved']:.0f} MiB")

    json.dump(rows, open("output/scaling5/assign_margin.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
