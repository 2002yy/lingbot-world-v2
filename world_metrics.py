#!/usr/bin/env python
"""§41F: calibrate the two metric rulers (corrected design).

FIRST ATTEMPT WAS WRONG -- recorded here because the mistake is instructive:
  * thresholds were fitted on samples where a large change IS EXPECTED
    (the object left the ROI), so no PASS gate could come out of it.
    A gate needs NEGATIVE samples: regions where nothing changed.
  * `z` blew up (52-157) because the control bank's std is tiny; the useful
    quantity is the ABSOLUTE excess `raw_mean - ctrl_mean`, since the vacated
    ROI changed by 46-55 grey levels while the background drift is ~1-3.
  * the residue template for relocation was identical for old and new, so
    `state_leak` was identically 0 by construction.

CORRECTED RULERS

A. bg_excess  (drift-controlled, absolute)
       raw   = mean |F - D0| over the TARGET roi
       ctrl  = same over a BANK of same-size tiles scattered over the frame
       excess      = raw - ctrl_mean          [grey levels]
       excess_p90  = p90(target) - ctrl_mean
   Calibrated on NEGATIVE samples (targets far from any object) -> excess ~ 0.
   A positive sample (an object ROI that really changed) is reported too, so
   the metric is shown to be sensitive, not just quiet.

B. state_leak  (always applicable; no displacement requirement)
   Evaluated AT THE OLD ROI only:
       S_old = sim(old_template, F[old_roi])
       S_new = sim(new_template, F[old_roi])
       state_leak = S_old - S_new
   state_leak > 0  => the OLD visual state is more findable than the new one
                      at the location it was replaced at = residue.
   No D0 baseline needed: old and new are compared at the same place.
   This closes the §41B coverage gap (it does not need >= 1 object width of
   displacement).

  python world_metrics.py --calibrate
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from object_permanence import patch  # noqa: E402

EPS = 1e-6
DOOR_BB = (0.70, 0.50, 0.84, 0.66)
WALL_BB = (0.28, 0.56, 0.52, 0.90)
CTRL_FLOOR = 1.0          # grey levels; keeps ratios finite


# --------------------------------------------------------------------------
def tile_bank(H, W, tile_hw, n=48, exclude=(), seed=0):
    rng = np.random.RandomState(seed)
    th, tw = tile_hw
    out, tries = [], 0
    while len(out) < n and tries < n * 200:
        tries += 1
        y = rng.randint(0, max(1, H - th))
        x = rng.randint(0, max(1, W - tw))
        if all((y + th <= ey0 or y >= ey1 or x + tw <= ex0 or x >= ex1)
               for (ey0, ey1, ex0, ex1) in exclude):
            out.append((y, x))
    return out


def bg_excess(F, D0, roi, bank, eps=EPS):
    y0, y1, x0, x1 = roi
    th, tw = y1 - y0, x1 - x0
    d = np.abs(F.astype(np.float32) - D0.astype(np.float32)).mean(-1)
    tgt = d[y0:y1, x0:x1].ravel()
    ctrl = np.concatenate([d[y:y + th, x:x + tw].ravel() for (y, x) in bank])
    mu = float(ctrl.mean())
    p90 = float(np.percentile(tgt, 90))
    return dict(raw=float(tgt.mean()), ctrl=mu, ctrl_p90=float(np.percentile(ctrl, 90)),
                excess=float(tgt.mean()) - mu, excess_p90=p90 - mu,
                ratio=float(tgt.mean()) / max(mu, CTRL_FLOOR))


def state_leak(feat, F, old_tpl, new_tpl, roi):
    """At the OLD roi only: is the old appearance more findable than the new?"""
    y0, y1, x0, x1 = roi
    seg = F[y0:y1, x0:x1]
    s_old = float((feat(old_tpl) * feat(seg)).sum())
    s_new = float((feat(new_tpl) * feat(seg)).sum())
    return dict(s_old=s_old, s_new=s_new, state_leak=s_old - s_new)


def bb_to_roi(bb, H, W):
    return (int(bb[1] * H), int(bb[3] * H), int(bb[0] * W), int(bb[2] * W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out_dir", default="output/metrics")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from eval_two_layer import load_models, dino_feat
    load_models()

    def feat(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    def load(p):
        return np.load(p)

    report = {}

    # ================= A. bg_excess =================
    print("=" * 74)
    print("A. bg_excess (drift-controlled, ABSOLUTE excess in grey levels)")
    print("=" * 74)
    print("   NEGATIVE samples = targets far from any object -> excess should ~ 0")
    print("   POSITIVE sample  = the object ROI that really changed -> large")
    print()
    print(f"   {'case':>24s} {'kind':>9s} {'raw':>7s} {'ctrl':>7s} "
          f"{'excess':>8s} {'p90exc':>8s} {'ratio':>7s}")

    neg, pos = [], []
    for name, fp, d0p in (
            ("dyn/slow_P1", "output/dyn/slow_P1_frames.npy", "output/dyn/D0_frames.npy"),
            ("dyn/fast_P1", "output/dyn/fast_P1_frames.npy", "output/dyn/D0_frames.npy"),
            ("d5/R1", "output/d5/R1_frames.npy", "output/d5/D0_frames.npy"),
            ("d5/R3", "output/d5/R3_frames.npy", "output/d5/D0_frames.npy"),
            ("reloc/T1_raw_s5", "output/reloc/T1_raw_s5_frames.npy",
             "output/reloc/D0_frames.npy")):
        if not (os.path.exists(fp) and os.path.exists(d0p)):
            continue
        F, D0 = load(fp), load(d0p)
        n = min(len(F), len(D0))
        H, W = F[0].shape[:2]
        obj = bb_to_roi(DOOR_BB, H, W)
        th, tw = obj[1] - obj[0], obj[3] - obj[2]
        ex_cl = (obj[0] - 2 * th, obj[1] + 2 * th, obj[2] - 2 * tw, obj[3] + 2 * tw)
        bank = tile_bank(H, W, (th, tw), n=64, exclude=[ex_cl])

        # positive: the object roi
        rs = [bg_excess(F[c], D0[c], obj, bank) for c in range(n)]
        p = dict(case=name, raw=float(np.mean([r["raw"] for r in rs])),
                 ctrl=float(np.mean([r["ctrl"] for r in rs])),
                 excess=float(np.mean([r["excess"] for r in rs])),
                 excess_p90=float(np.mean([r["excess_p90"] for r in rs])),
                 ratio=float(np.mean([r["ratio"] for r in rs])))
        pos.append(p)
        print(f"   {name:>24s} {'object':>9s} {p['raw']:7.2f} {p['ctrl']:7.2f} "
              f"{p['excess']:8.2f} {p['excess_p90']:8.2f} {p['ratio']:7.2f}")

        # negative: several far-away target rois
        for j, (ty, tx) in enumerate([(int(0.06 * H), int(0.06 * W)),
                                      (int(0.06 * H), int(0.62 * W)),
                                      (int(0.72 * H), int(0.06 * W))]):
            troi = (ty, ty + th, tx, tx + tw)
            if troi[1] > H or troi[3] > W:
                continue
            b2 = tile_bank(H, W, (th, tw), n=64,
                           exclude=[ex_cl, (troi[0] - th, troi[1] + th,
                                            troi[2] - tw, troi[3] + tw)])
            rs2 = [bg_excess(F[c], D0[c], troi, b2) for c in range(n)]
            q = dict(case=f"{name}#neg{j}",
                     raw=float(np.mean([r["raw"] for r in rs2])),
                     ctrl=float(np.mean([r["ctrl"] for r in rs2])),
                     excess=float(np.mean([r["excess"] for r in rs2])),
                     excess_p90=float(np.mean([r["excess_p90"] for r in rs2])),
                     ratio=float(np.mean([r["ratio"] for r in rs2])))
            neg.append(q)
            print(f"   {q['case']:>24s} {'far':>9s} {q['raw']:7.2f} "
                  f"{q['ctrl']:7.2f} {q['excess']:8.2f} {q['excess_p90']:8.2f} "
                  f"{q['ratio']:7.2f}")

    if neg:
        e = np.array([q["excess"] for q in neg])
        ep = np.array([q["excess_p90"] for q in neg])
        r = np.array([q["ratio"] for q in neg])
        print(f"\n   NEGATIVE (far) excess : mean {e.mean():+.2f} sd {e.std():.2f} "
              f"max {e.max():+.2f}")
        print(f"   NEGATIVE (far) p90exc : mean {ep.mean():+.2f} sd {ep.std():.2f} "
              f"max {ep.max():+.2f}")
        print(f"   NEGATIVE (far) ratio  : mean {r.mean():.2f} max {r.max():.2f}")
        thr_e = float(np.ceil(e.max() + 2 * e.std() + 0.5))
        thr_ep = float(np.ceil(ep.max() + 2 * ep.std() + 0.5))
        thr_r = float(np.ceil(r.max() + 2 * r.std() + 0.05))
        print(f"\n   => PASS thresholds (neg max + 2sd, floored & rounded up):")
        print(f"        bg_excess      <= {thr_e:.1f} grey levels")
        print(f"        bg_excess_p90  <= {thr_ep:.1f} grey levels")
        print(f"        bg_excess_ratio<= {thr_r:.2f}")
        report["bg_negative"] = neg
        report["bg_positive"] = pos
        report["bg_thresholds"] = dict(excess=thr_e, excess_p90=thr_ep, ratio=thr_r)

    # ================= B. state_leak =================
    print()
    print("=" * 74)
    print("B. state_leak at the OLD roi (always applicable)")
    print("=" * 74)
    print("   state_leak = sim(old_tpl, F[old_roi]) - sim(new_tpl, F[old_roi])")
    print("   <0 correct switch | >0 OLD state still more findable = residue")
    print()
    print(f"   {'case':>30s} {'S_old':>7s} {'S_new':>7s} {'leak':>8s} {'n':>4s}")

    leaks = []

    def _leak_case(label, fp_new, fp_old, kind, ref_frac=None):
        """Templates must be taken AFTER the injection/switch has happened.
        For these runs the revisit chunks are late (e.g. 42,43 of 52) and the
        observe window follows, so `n-3` is safely past it. Taking the
        registration chunk (index 2) makes both arms identical and the leak
        identically zero -- the mistake made in the first two passes."""
        if not (os.path.exists(fp_new) and os.path.exists(fp_old)):
            return None
        Fn, Fo = load(fp_new), load(fp_old)
        n = min(len(Fn), len(Fo))
        ri = n - 3
        new_tpl = patch(Fn[ri], DOOR_BB)
        old_tpl = patch(Fo[ri], DOOR_BB)
        H, W = Fn[0].shape[:2]
        roi = bb_to_roi(DOOR_BB, H, W)
        rs = [state_leak(feat, Fn[c], old_tpl, new_tpl, roi) for c in range(n)]
        d = dict(case=label, kind=kind, ref_chunk=ri, n=n,
                 s_old=float(np.mean([x["s_old"] for x in rs])),
                 s_new=float(np.mean([x["s_new"] for x in rs])),
                 leak=float(np.mean([x["state_leak"] for x in rs])))
        leaks.append(d)
        print(f"   {label:>34s} {d['s_old']:7.3f} {d['s_new']:7.3f} "
              f"{d['leak']:8.3f} {n:4d}  (tpl@{ri})")
        return d

    # REAL: state switch d1(CLOSED) -> d4(canonicalised OPEN)
    a = _leak_case("d4 D4(new open) vs D1(old closed)",
                   "output/d4/D4_frames.npy", "output/d4/D1_frames.npy",
                   "state_switch")
    # MIRROR CONTROL: swap old/new -> the sign must flip. This validates that
    # the ruler is sensitive and signed, without needing a no-change sample.
    b = _leak_case("MIRROR d4 D1 vs D4 (must flip)",
                   "output/d4/D1_frames.npy", "output/d4/D4_frames.npy",
                   "mirror")
    # anchor switch: R0(once) -> R3(every chunk), same OPEN anchor
    _leak_case("d5 R3 vs old R0 (anchor switch)",
               "output/d5/R3_frames.npy", "output/d5/R0_frames.npy",
               "anchor_switch")
    # relocation: T1_raw_s10 moved away vs D0 (object was at the old roi)
    _leak_case("reloc T1_s10 vs D0 (object left)",
               "output/reloc/T1_raw_s10_frames.npy",
               "output/reloc/D0_frames.npy", "relocation")

    if a and b:
        print(f"\n   MIRROR CHECK: {a['leak']:+.3f} vs {b['leak']:+.3f} "
              f"-> signs {'FLIP (sensitive)' if a['leak'] * b['leak'] < 0 else 'DO NOT FLIP (broken)'}")
        report["mirror_ok"] = bool(a["leak"] * b["leak"] < 0)

    if leaks:
        ref = max(abs(x["leak"]) for x in leaks)
        thr = float(np.ceil(ref * 100 + 2) / 100)      # round UP to 2 dp
        print(f"\n   max |leak| observed on real cases: {ref:.3f}")
        print(f"   => report threshold: |state_leak| <= {thr:.2f}")
        report["state_leak_threshold"] = thr
        report["state_leak"] = leaks

    print("\n   NOTE: state_leak is now ALWAYS computable (evaluated at the old")
    print("   roi only; no displacement >= 1 object width needed) -> the §41B")
    print("   coverage gap is closed.")
    print("   bg_excess is in grey levels with an explicit negative baseline, so")
    print("   a PASS gate exists instead of a bare number.")

    json.dump(report, open(f"{args.out_dir}/world_metrics.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
