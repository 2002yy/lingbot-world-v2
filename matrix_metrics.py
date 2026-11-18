#!/usr/bin/env python
"""Memory-matrix metrics: motion response, signed yaw/pitch response, closure, memory.

  python matrix_metrics.py --dir output/matrix --out output/matrix/metrics.csv
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from eval_two_layer import load_models, dino_feat, lpips_d, orb_closure


def read_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


def flow_field(a, b):
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)


def yaw_pitch_slope(fl):
    """Signed rotation response.

    For a small yaw the optical flow is an almost uniform horizontal shift
    (mean dx ~ -f*theta); for a pitch it is a uniform vertical shift. The
    radial (x-cx) / (y-cy) slopes capture zoom/translation instead, so we
    report both: mean shift (rotation) and slope (zoom).
    """
    h, w = fl.shape[:2]
    xs = np.arange(w, dtype=np.float32)[None, :] - w / 2.0
    ys = np.arange(h, dtype=np.float32)[:, None] - h / 2.0
    xs = np.broadcast_to(xs, (h, w))
    ys = np.broadcast_to(ys, (h, w))
    dx, dy = fl[..., 0], fl[..., 1]
    a = float((dx * xs).sum() / max((xs * xs).sum(), 1e-6))
    b = float((dy * ys).sum() / max((ys * ys).sum(), 1e-6))
    return float(dx.mean()), float(dy.mean()), a, b


def phase(frames, lo, hi):
    mags, mx, my, zx, zy = [], [], [], [], []
    for i in range(max(1, lo), min(hi, len(frames))):
        fl = flow_field(frames[i - 1], frames[i])
        mags.append(float(np.mean(np.linalg.norm(fl, axis=2))))
        m_x, m_y, a, b = yaw_pitch_slope(fl)
        mx.append(m_x); my.append(m_y); zx.append(a); zy.append(b)
    if not mags:
        nan = float("nan")
        return dict(mag=nan, yaw=nan, pitch=nan, zoom_x=nan, zoom_y=nan)
    return dict(mag=float(np.mean(mags)), yaw=float(np.mean(mx)),
                pitch=float(np.mean(my)), zoom_x=float(np.mean(zx)),
                zoom_y=float(np.mean(zy)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output/matrix")
    ap.add_argument("--out", default="output/matrix/metrics.csv")
    args = ap.parse_args()

    load_models()
    rows = []
    for jf in sorted(glob.glob(f"{args.dir}/*.json")):
        with open(jf) as fh:
            m = json.load(fh)
        mp4 = m["mp4"]
        if not os.path.exists(mp4):
            continue
        fr = read_frames(mp4)
        h, w = fr[0].shape[:2]
        p_fwd = phase(fr, *m["fwd"])
        p_ho = phase(fr, *m["hold_out"])
        p_ret = phase(fr, *m["ret"])
        p_yp = phase(fr, *m["yaw_plus"])
        # matched-length window: the yaw_minus segment is 2x longer, so use the
        # first len(yaw_plus) frames for a fair comparison
        yp_lo, yp_hi = m["yaw_plus"]
        ym_lo, ym_hi = m["yaw_minus"]
        p_ym = phase(fr, ym_lo, ym_lo + (yp_hi - yp_lo))

        ri = m["return_idx"]
        cl = orb_closure(fr[0], fr[ri]) if ri < len(fr) else None
        if cl:
            Hinv = np.linalg.inv(cl["H"])
            warped = cv2.warpPerspective(fr[ri], Hinv, (w, h),
                                         flags=cv2.INTER_LINEAR)
            d_all = 1.0 - float((dino_feat(cv2.cvtColor(fr[0], cv2.COLOR_BGR2RGB))
                                 * dino_feat(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))).sum())
            lp_all = lpips_d(cv2.cvtColor(fr[0], cv2.COLOR_BGR2RGB),
                             cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
            nGood, inlier, medReproj = cl["nGood"], cl["inlier"], cl["medReproj"]
        else:
            d_all = lp_all = float("nan")
            nGood = inlier = medReproj = float("nan")

        rows.append(dict(
            run=os.path.basename(jf)[:-5], scene=m["scene"],
            local=m["local"], sink=m["sink"], K=m["K"], seed=m["seed"],
            fwd_mag=p_fwd["mag"], ho_mag=p_ho["mag"], ret_mag=p_ret["mag"],
            yawp_mag=p_yp["mag"], yawm_mag=p_ym["mag"],
            yaw_plus=p_yp["yaw"], yaw_minus=p_ym["yaw"],
            yaw_asym=p_yp["yaw"] - p_ym["yaw"],
            pitch_plus=p_yp["pitch"], pitch_minus=p_ym["pitch"],
            reversal=p_ret["mag"] / max(p_fwd["mag"], 1e-6),
            nGood=nGood, inlier=inlier, medReproj=medReproj,
            dino_dist=d_all, lpips=lp_all))

    keys = list(rows[0].keys()) if rows else []
    with open(args.out, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[mm] {len(rows)} runs -> {args.out}")

    # compact summary
    print(f"\n{'run':24s} {'fwd':>5s} {'ho':>5s} {'ret':>5s} "
          f"{'yaw+':>7s} {'yaw-':>7s} {'asym':>7s} {'nGood':>6s} "
          f"{'inl':>5s} {'repr':>5s} {'DINO':>6s} {'LPIPS':>6s}")
    for r in rows:
        print(f"{r['run']:24s} {r['fwd_mag']:5.2f} {r['ho_mag']:5.2f} "
              f"{r['ret_mag']:5.2f} {r['yaw_plus']:7.3f} {r['yaw_minus']:7.3f} "
              f"{r['yaw_asym']:7.3f} {r['nGood']:6.0f} {r['inlier']:5.2f} "
              f"{r['medReproj']:5.1f} {r['dino_dist']:6.3f} {r['lpips']:6.3f}")


if __name__ == "__main__":
    main()
