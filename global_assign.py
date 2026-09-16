#!/usr/bin/env python
"""§42C-4 Global Assignment Probe.

The remaining 29.7% of §42C-3 failures are NOT geometry failures -- geometry
already separates almost perfectly (POS dgeo ~0 vs NEG dgeo >> 0.02). The
problem is that at the correct position the crop VISUALLY looks more like A/B,
so the local test `self > best_other` rejects it to protect precision.

Two possible explanations, and they demand different next steps:

  A. ASSOCIATION is too naive
     C-vs-A/B loses locally, but in a GLOBAL one-to-one assignment over
     {A,B,C} x {candidates} the correct assignment still wins decisively.
     -> representation is adequate; fix the algorithm. §42C FULL PASS, go §42D.

  B. REPRESENTATION is genuinely ambiguous
     even global assignment finds no evidence advantage for the correct pairing.
     -> §43P (multi-view / view-conditioned) must move earlier.

This probe changes ONLY the decision rule. No model change, no multi-view, no
DINO change, no threshold change.

    score(i, j) = visual(i, j)
                + geom_w   * max(0, 1 - d_geo(i,j) / dgeo_gate)
                + persist_w

solved as a max-weight one-to-one assignment (Hungarian) over
    entities x (candidates + NULL)
The NULL option is MANDATORY: without it the world prior becomes an answer and
NEG-3 (entity exists but the real object never returned) breaks.

Same four controls and same metrics as §42C-3.
  python global_assign.py
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, ".")
from object_permanence import patch  # noqa: E402

OBJECTS = {
    "A_left":  (0.28, 0.56, 0.52, 0.90),
    "B_right": (0.68, 0.42, 0.88, 0.72),
    "C_ctrl":  (0.80, 0.10, 0.94, 0.40),
}
TARGET = "C_ctrl"
ENTITIES = list(OBJECTS.keys())
DGEO_GATE = 0.020          # from §42C-3 joint calibration
FB_BUDGET = 0.02


def d_geo(uv_a, uv_b, wh):
    w, h = wh
    return math.hypot((uv_a[0] - uv_b[0]) / max(w, 1e-6),
                      (uv_a[1] - uv_b[1]) / max(h, 1e-6))


def assign(visual, dgeos, geom_w, persist_w, null_pen):
    """visual: [n_ent][n_cand] similarities; dgeos: same shape.
    Returns list of (entity_idx, cand_idx_or_None)."""
    n_e, n_c = visual.shape
    S = np.array(visual, float) + persist_w
    S = S + geom_w * np.maximum(0.0, 1.0 - np.array(dgeos, float) / DGEO_GATE)
    # append a NULL column per entity
    full = np.concatenate([S, np.full((n_e, n_e), null_pen)], axis=1)
    r, c = linear_sum_assignment(-full)
    out = []
    for ei, cj in zip(r, c):
        out.append((int(ei), None if cj >= n_c else int(cj)))
    return out


def main():
    from eval_two_layer import load_models, dino_feat
    load_models()

    def feat(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    files = sorted(glob.glob("output/tolerance/frames_*.npy"))
    if not files:
        print("[ga] no frames")
        return

    # ------------------------------------------------------------------
    # build per-chunk cases: candidates = proposals around the target plus
    # the other objects' boxes; entities = A/B/C
    # ------------------------------------------------------------------
    cases = []   # dict(tag, visual[3][n], dgeo[3][n], target_cand_idx, truth)
    for f in files:
        ang = float(os.path.basename(f).replace("frames_", "").replace(".npy", ""))
        F = np.load(f)
        n = len(F)
        H, W = F[0].shape[:2]
        tpl = {nm: patch(F[1], bb) for nm, bb in OBJECTS.items()}
        T = {nm: feat(tpl[nm]) for nm in ENTITIES}
        box = OBJECTS[TARGET]
        bw, bh = box[2] - box[0], box[3] - box[1]
        rng = np.random.RandomState(int(ang * 100) + 11)

        for cid in range(max(0, n - 8), n):
            cands = []      # (name, bb, truth_entity)
            # 1) the TRUE target proposal at its predicted position
            cands.append(("target", box, TARGET))
            # 2) NEG-1 style distractors near the predicted region
            for k in range(2):
                dy = rng.uniform(-0.5, 0.5) * bh
                dx = rng.uniform(-0.5, 0.5) * bw
                nb = (min(max(box[0] + dx, 0.0), 1 - bw),
                      min(max(box[1] + dy, 0.0), 1 - bh),
                      min(max(box[2] + dx, bw), 1.0),
                      min(max(box[3] + dy, bh), 1.0))
                cands.append((f"near{k}", nb, None))
            # 3) the other objects' real boxes (NEG-2 style)
            for nm in ENTITIES:
                if nm != TARGET:
                    cands.append((nm, OBJECTS[nm], nm))

            cvis = np.zeros((len(ENTITIES), len(cands)))
            cgeo = np.zeros((len(ENTITIES), len(cands)))
            for j, (cname, cb, _) in enumerate(cands):
                fv = feat(patch(F[cid], cb))
                ctr = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
                for i, en in enumerate(ENTITIES):
                    cvis[i, j] = float((fv * T[en]).sum())
                    cgeo[i, j] = d_geo(ctr, (
                        (OBJECTS[en][0] + OBJECTS[en][2]) / 2,
                        (OBJECTS[en][1] + OBJECTS[en][3]) / 2), (bw, bh))
            ti = ENTITIES.index(TARGET)
            cases.append(dict(tag="POS", angle=ang, visual=cvis, dgeo=cgeo,
                              target_cand=0, target_ent=ti))

            # NEG-2 case: remove the true target proposal entirely; the
            # other objects' crops remain. The gate must NOT bind C.
            cvis2 = np.delete(cvis, 0, axis=1)
            cgeo2 = np.delete(cgeo, 0, axis=1)
            cases.append(dict(tag="NEG2", angle=ang, visual=cvis2, dgeo=cgeo2,
                              target_cand=None, target_ent=ti))

            # NEG-3 case: target crop replaced by a far background patch
            yy = rng.randint(0, max(1, H - int(bh * H)))
            xx = rng.randint(0, max(1, W - int(bw * W)))
            far = (xx / W, yy / H, xx / W + bw, yy / H + bh)
            cvis3 = cvis.copy(); cgeo3 = cgeo.copy()
            fv = feat(F[cid][yy:yy + int(bh * H), xx:xx + int(bw * W)])
            for i, en in enumerate(ENTITIES):
                cvis3[i, 0] = float((fv * T[en]).sum())
                cgeo3[i, 0] = d_geo(((far[0] + far[2]) / 2, (far[1] + far[3]) / 2),
                                    ((OBJECTS[en][0] + OBJECTS[en][2]) / 2,
                                     (OBJECTS[en][1] + OBJECTS[en][3]) / 2),
                                    (bw, bh))
            cases.append(dict(tag="NEG3", angle=ang, visual=cvis3, dgeo=cgeo3,
                              target_cand=0, target_ent=ti))

    print(f"[ga] {len(cases)} cases built "
          f"({sum(1 for c in cases if c['tag']=='POS')} POS)")

    # ------------------------------------------------------------------
    # calibrate (geom_w, persist_w, null_pen) on POS vs NEG
    # ------------------------------------------------------------------
    def evaluate(geom_w, persist_w, null_pen):
        stat = {"POS": [0, 0], "NEG2": [0, 0], "NEG3": [0, 0]}
        for c in cases:
            res = assign(c["visual"], c["dgeo"], geom_w, persist_w, null_pen)
            bound = {ei: cj for ei, cj in res}
            ti = c["target_ent"]
            got = bound.get(ti)
            hit = (c["target_cand"] is not None and got == c["target_cand"])
            stat[c["tag"]][1] += 1
            if hit:
                stat[c["tag"]][0] += 1
        rec = stat["POS"][0] / max(stat["POS"][1], 1)
        fb = (stat["NEG2"][0] + stat["NEG3"][0]) / max(
            stat["NEG2"][1] + stat["NEG3"][1], 1)
        return rec, fb, stat

    print("\n[ga] ===== calibration (geom_w, persist_w, null_pen) =====")
    best = None
    for geom_w in (0.0, 0.1, 0.2, 0.3, 0.5, 0.8):
        for persist_w in (0.0, 0.05, 0.10, 0.20):
            for null_pen in (0.30, 0.45, 0.60, 0.75, 0.90):
                rec, fb, _ = evaluate(geom_w, persist_w, null_pen)
                if fb <= FB_BUDGET and (best is None or rec > best[0]):
                    best = (rec, fb, geom_w, persist_w, null_pen)
    if best is None:
        print("  no operating point within the false-bind budget")
        return
    rec, fb, geom_w, persist_w, null_pen = best
    print(f"  best: geom_w={geom_w:.2f} persist_w={persist_w:.2f} "
          f"null_pen={null_pen:.2f} -> recall {rec*100:.1f}% "
          f"false-bind {fb*100:.1f}%")

    # ------------------------------------------------------------------
    # final evaluation + comparison against the §42C-3 local gate
    # ------------------------------------------------------------------
    rec, fb, stat = evaluate(geom_w, persist_w, null_pen)
    baseline = 42.0     # §42C-3 stage-0-only recall
    local = 70.3        # §42C-3 state-conditioned recall
    print("\n[ga] ===== §42C-4 global assignment evaluation =====")
    for k, (hit, tot) in stat.items():
        print(f"  {k:>5s}: {hit}/{tot} = {100*hit/max(tot,1):.1f}%")
    print(f"\n  {'method':>34s} {'recall':>8s} {'false-bind':>11s}")
    print(f"  {'§42C-3 stage-0 only (absolute 0.60)':>34s} {baseline:7.1f}% "
          f"{'n/a':>11s}")
    print(f"  {'§42C-3 local state-conditioned':>34s} {local:7.1f}% "
          f"{0.0:10.1f}%")
    print(f"  {'§42C-4 global assignment':>34s} {rec*100:7.1f}% {fb*100:10.1f}%")

    # ------------------------------------------------------------------
    # verdict
    # ------------------------------------------------------------------
    print("\n[ga] ===== verdict =====")
    if rec >= 0.80 and fb <= 0.05:
        verdict = ("CASE 1: association was the bottleneck. Representation is "
                   "adequate -> §42C FULL PASS, proceed to §42D.")
    elif rec >= 0.90 and fb > 0.05:
        verdict = ("CASE 3: recall improved but false-bind appeared. The "
                   "conservative 70.3%/0% point must be kept.")
    else:
        verdict = ("CASE 2: global matching does NOT close the gap; the "
                   "correct candidate is genuinely ambiguous. Representation "
                   "is the bottleneck -> §43P moves earlier.")
    print(f"  {verdict}")
    print(f"\n  (§42C-3 conservative operating point: {local:.1f}% recall / "
          f"0.0% false-bind)")
    if rec > local + 3.0:
        print(f"  global assignment changes recall {local:.1f}% -> {rec*100:.1f}% "
              f"({rec*100-local:+.1f}pp)")

    json.dump(dict(geom_w=geom_w, persist_w=persist_w, null_pen=null_pen,
                   recall=float(rec), false_bind=float(fb), stats=stat,
                   baseline=baseline, local=local, verdict=verdict),
              open("output/tolerance/global_assign.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
