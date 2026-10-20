#!/usr/bin/env python
"""§42C-3 calibration: where should the persistent-prior margin gate sit?

Do NOT guess the margin gate. Calibrate it from real positive and negative
samples that already exist on disk:

  POSITIVE   the target C's crop after a long out-of-FOV + return
             (tolerance_sweep.py saved frames_<dYaw>.npy per angle)
  NEG_OBJECT a crop of a KNOWN DIFFERENT object evaluated with C's template
             (this is exactly the wrong-ID negative from §39)
  NEG_BG     a random background crop with no object in it

For each sample we compute
    self  = sim(crop, C_template)
    other = max over the other known templates
    margin = self - other
and ask whether the POSITIVE margin distribution is cleanly separated from the
NEGATIVE ones. If it is, the gate is just a threshold on margin. If they
overlap, geometry/world-state priors must carry the rest.

  python prior_gate_calib.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from object_permanence import patch  # noqa: E402

OBJECTS = {
    "A_left":  (0.28, 0.56, 0.52, 0.90),
    "B_right": (0.68, 0.42, 0.88, 0.72),
    "C_ctrl":  (0.80, 0.10, 0.94, 0.40),   # the §42C-2 target
}
TARGET = "C_ctrl"
ID_EXIST = 0.60


def main():
    from eval_two_layer import load_models, dino_feat
    load_models()

    def feat(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    files = sorted(glob.glob("output/tolerance/frames_*.npy"))
    print(f"[pc] found {len(files)} tolerance_sweep frame sets")
    if not files:
        print("[pc] nothing to calibrate on; run tolerance_sweep.py first")
        return

    pos_self, pos_other, pos_margin = [], [], []
    negobj_self, negobj_other, negobj_margin = [], [], []
    negbg_self, negbg_other, negbg_margin = [], [], []
    per_angle = []

    for f in files:
        ang = os.path.basename(f).replace("frames_", "").replace(".npy", "")
        F = np.load(f)
        n = len(F)
        H, W = F[0].shape[:2]
        # templates from the baseline chunks, where every object is injected at
        # its home roi -> a fair "known appearance" reference
        tpl = {nm: patch(F[1], bb) for nm, bb in OBJECTS.items()}
        T = {nm: feat(tpl[nm]) for nm in OBJECTS}

        # ---- POSITIVE: the target's crop near the end (post-return) ----
        bb = OBJECTS[TARGET]
        a_self, a_other, a_marg = [], [], []
        for cid in range(max(0, n - 8), n):
            crop = patch(F[cid], bb)
            fv = feat(crop)
            s = float((fv * T[TARGET]).sum())
            o = max(float((fv * T[nm]).sum()) for nm in OBJECTS if nm != TARGET)
            a_self.append(s); a_other.append(o); a_marg.append(s - o)
        pos_self += a_self; pos_other += a_other; pos_margin += a_marg

        # ---- NEG_OBJECT: a different object's crop scored with C's template ----
        for nm in OBJECTS:
            if nm == TARGET:
                continue
            for cid in range(max(0, n - 8), n):
                fv = feat(patch(F[cid], OBJECTS[nm]))
                s = float((fv * T[TARGET]).sum())
                o = max(float((fv * T[o]).sum()) for o in OBJECTS if o != nm)
                negobj_self.append(s); negobj_other.append(o)
                negobj_margin.append(s - o)

        # ---- NEG_BG: random background crops (no object) ----
        rng = np.random.RandomState(0)
        for _ in range(40):
            hh = int(0.18 * H); ww = int(0.18 * W)
            y = rng.randint(0, H - hh); x = rng.randint(0, W - ww)
            crop = F[n - 1][y:y + hh, x:x + ww]
            fv = feat(crop)
            sims = {nm: float((fv * T[nm]).sum()) for nm in OBJECTS}
            order = sorted(sims.values(), reverse=True)
            negbg_self.append(order[0]); negbg_other.append(order[1])
            negbg_margin.append(order[0] - order[1])

        s = float(np.mean(a_self)); m = float(np.mean(a_marg))
        per_angle.append(dict(angle=ang, self=s, margin=m))
        print(f"[pc] dYaw {ang:>5s}: positive self={s:.3f} margin={m:+.3f}")

    print("\n[pc] ===== sample distributions =====")
    for nm, ms, ss in (("POSITIVE (true C return)", pos_margin, pos_self),
                       ("NEG_OBJECT (wrong object crop)", negobj_margin, negobj_self),
                       ("NEG_BG (background crop)", negbg_margin, negbg_self)):
        a = np.array(ms)
        print(f"  {nm:34s} n={len(a):4d}  self_mean={np.mean(ss):.3f}  "
              f"margin mean {a.mean():+.3f}  sd {a.std():.3f}  "
              f"min {a.min():+.3f}  p95 {np.percentile(a,95):+.3f}  "
              f"max {a.max():+.3f}")

    P = np.array(pos_margin)
    N1 = np.array(negobj_margin)
    N2 = np.array(negbg_margin)
    N = np.concatenate([N1, N2])
    print(f"\n[pc] NEG combined: mean {N.mean():+.3f} p95 "
          f"{np.percentile(N,95):+.3f} p99 {np.percentile(N,99):+.3f} "
          f"max {N.max():+.3f}")
    print(f"[pc] POS        : mean {P.mean():+.3f} p5 "
          f"{np.percentile(P,5):+.3f} min {P.min():+.3f}")

    # clean separation?
    sep = np.percentile(P, 5) - np.percentile(N, 99)
    print(f"\n[pc] separation = POS_p5 - NEG_p99 = {sep:+.3f}")
    if sep > 0:
        gate = float((np.percentile(P, 5) + np.percentile(N, 99)) / 2)
        print(f"  -> CLEAN GAP. margin_gate = {gate:+.3f} separates them")
        print(f"     (POS recall at that gate = "
              f"{np.mean(P > gate)*100:.0f}%, NEG false-bind = "
              f"{np.mean(N > gate)*100:.0f}%)")
        rec = dict(gate=gate, clean=True)
    else:
        print("  -> OVERLAP. A margin gate alone cannot separate them;")
        print("     the geometry / world-state prior must carry the rest.")
        # best achievable by margin alone
        best = None
        for g in np.linspace(-0.2, 0.3, 501):
            rec_ = float(np.mean(P > g)); fb = float(np.mean(N > g))
            score = rec_ - fb
            if best is None or score > best[2]:
                best = (g, rec_, fb, score)
        print(f"     best margin-only gate {best[0]:+.3f}: recall "
              f"{best[1]*100:.0f}%, false-bind {best[2]*100:.0f}%")
        rec = dict(gate=float(best[0]), clean=False,
                   recall=float(best[1]), false_bind=float(best[2]))

    # also: does the ABSOLUTE threshold alone separate?
    print("\n[pc] ===== absolute-threshold view =====")
    print(f"  POS self  mean {np.mean(pos_self):.3f}  max {np.max(pos_self):.3f}")
    print(f"  NEG_OBJ self mean {np.mean(negobj_self):.3f} "
          f"max {np.max(negobj_self):.3f}")
    print(f"  NEG_BG self  mean {np.mean(negbg_self):.3f} "
          f"max {np.max(negbg_self):.3f}")
    ok60 = np.mean(np.array(pos_self) >= ID_EXIST)
    print(f"  fraction of POS passing self>={ID_EXIST}: {ok60*100:.0f}%")
    print("  -> this is why §42C failed: the true positive does not clear the")
    print("     absolute bar, while the margin still identifies it.")

    json.dump(dict(per_angle=per_angle,
                   pos=dict(self=pos_self, margin=pos_margin),
                   neg_obj=dict(self=negobj_self, margin=negobj_margin),
                   neg_bg=dict(self=negbg_self, margin=negbg_margin),
                   gate=rec), open("output/tolerance/gate_calib.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
