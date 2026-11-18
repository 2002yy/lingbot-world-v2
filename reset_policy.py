#!/usr/bin/env python
"""Runtime reset / correction policy derived from the warp error-growth model.

Ground truth (LPIPS vs the real future frame) is not available at runtime, so
the policy keys off cheap online proxies. Measured correlation with LPIPS:

    epe    r=+0.765   (needs ground truth -> offline only)
    disp   r=+0.747   <-- best ONLINE proxy (mean pixel displacement of H)
    trans  r=+0.643
    yaw    r=+0.569
    cond   r=+0.516
    valid  r=-0.475

Error model (least squares on output/motions/error_growth.csv):
    LPIPS ~ 0.1262 + 0.00263*disp + 0.01924*yaw_deg     R2=0.651, RMSE=0.112
    (disp alone: R2=0.558; +trans: R2=0.729 but collinear)

Policy states keyed on the predicted LPIPS:
    ok     pred < 0.20   warp only
    soft   pred < 0.30   blend toward the reference
    reset  pred < 0.45   raise correction weight
    fail   pred >= 0.45 or valid < 0.90   hold / soften motion
"""
import argparse
import csv

import numpy as np

# --- error model coefficients ---
LPIPS_INTERCEPT = 0.1262
LPIPS_PER_DISP = 0.00263
LPIPS_PER_YAW = 0.01924

# --- predicted-LPIPS thresholds ---
PRED_SOFT = 0.20
PRED_RESET = 0.30
PRED_FAIL = 0.45
VALID_FAIL = 0.90

# --- displacement thresholds used only for the timing estimate ---
DISP_SOFT = 15.0
DISP_RESET = 30.0
DISP_FAIL = 70.0


def predict_lpips(disp, yaw_deg=0.0):
    return (LPIPS_INTERCEPT + LPIPS_PER_DISP * float(disp)
            + LPIPS_PER_YAW * float(yaw_deg))


def policy(disp, valid=1.0, yaw_deg=0.0):
    """Return (state, action, correction_weight)."""
    p = predict_lpips(disp, yaw_deg)
    if valid < VALID_FAIL or p >= PRED_FAIL:
        return "fail", "hold_and_soften", 1.0
    if p >= PRED_RESET:
        return "reset", "raise_correction_weight", 0.7
    if p >= PRED_SOFT:
        return "soft", "blend_toward_reference", 0.35
    return "ok", "warp_only", 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="output/motions/error_growth.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    for r in rows:
        for k in r:
            if k not in ("clip", "scene", "seg"):
                r[k] = float(r[k])

    # one chunk == the pose step at d in {2,3,4} (all share the next latent)
    print("Per-motion error at the END of one chunk (d=4, 1250 ms):")
    print(f"{'seg':11s} {'disp':>7s} {'LPIPS':>7s} {'pred':>6s} {'valid':>6s} "
          f"{'yaw':>6s} {'T_soft':>7s} {'T_reset':>8s} {'T_fail':>7s} {'state':>6s}")
    segs = sorted({r["seg"] for r in rows})
    for seg in segs:
        sub = [r for r in rows if r["seg"] == seg and abs(r["ms"] - 1250) < 1]
        if not sub:
            continue
        f = lambda k: float(np.mean([s[k] for s in sub]))
        disp, lp, valid, yaw = f("disp"), f("lpips"), f("valid"), f("yaw_deg")
        pred = predict_lpips(disp, yaw)
        # disp grows ~linearly with time inside a chunk -> scale 1.25 s
        t_soft = 1.25 * min(1.0, DISP_SOFT / max(disp, 1e-6))
        t_reset = 1.25 * min(1.0, DISP_RESET / max(disp, 1e-6))
        t_fail = 1.25 * min(1.0, DISP_FAIL / max(disp, 1e-6))
        if valid < VALID_FAIL:
            t_fail = min(t_fail, 1.25 * (valid / 1.0))
        st, act, w = policy(disp, valid, yaw)
        print(f"{seg:11s} {disp:7.1f} {lp:7.3f} {pred:6.3f} {valid:6.3f} "
              f"{yaw:6.2f} {t_soft:6.2f}s {t_reset:7.2f}s {t_fail:6.2f}s {st:>6s}")

    print("\nProxy correlation with LPIPS (Pearson):")
    lp = np.array([r["lpips"] for r in rows])
    for k in ["disp", "valid", "yaw_deg", "trans", "cond"]:
        x = np.array([r[k] for r in rows])
        print(f"  {k:8s} r={float(np.corrcoef(x, lp)[0,1]):+.3f}")

    print("\nPolicy states over all samples:")
    from collections import Counter
    c = Counter(policy(r["disp"], r["valid"], r["yaw_deg"])[0] for r in rows)
    print(" ", dict(c))


if __name__ == "__main__":
    main()
