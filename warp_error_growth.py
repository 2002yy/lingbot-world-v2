#!/usr/bin/env python
"""Warp error-growth model: pure reprojection vs the real future frame.

For every motion segment and every reference frame, warp the reference frame to
the pose Delta frames ahead (Level 0 homography) and compare against the frame
the world model actually produced at that time (ground truth).

Also records cheap, ground-truth-free online proxies so a runtime reset policy
can be built without LPIPS.

  python warp_error_growth.py --clips output/motions
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch

from eval_two_layer import load_models, dino_feat, lpips_d
from pose_warp import PoseWarper
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses

# 4 output frames per chunk, one chunk every ~1.25 s -> 312.5 ms per frame
MS_PER_FRAME = 312.5
DELTAS = [0, 1, 2, 3, 4, 6, 8]


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


def latent_poses(traj, L):
    P = np.asarray(traj, float)
    N = len(P)
    return interpolate_camera_poses(
        np.linspace(0, N - 1, N), P[:, :3, :3], P[:, :3, 3],
        np.linspace(0, N - 1, L)).numpy()


def pose_for_output_frame(lat, k):
    i = 0 if k == 0 else (k + 3) // 4
    return lat[min(i, len(lat) - 1)]


def band_err(a, b, band):
    h, w = a.shape[:2]
    y0, y1 = int(band * h), int((1 - band) * h)
    x0, x1 = int(band * w), int((1 - band) * w)
    return float(np.abs(a[y0:y1, x0:x1].astype(float) -
                        b[y0:y1, x0:x1].astype(float)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="output/motions")
    ap.add_argument("--out", default="output/motions/error_growth.csv")
    ap.add_argument("--n_refs", type=int, default=3)
    args = ap.parse_args()

    load_models()
    rows = []
    for mp4 in sorted(glob.glob(f"{args.clips}/*.mp4")):
        tag = os.path.basename(mp4)[:-4]
        scene = tag.split("_")[0]
        ex = f"examples/mot_{scene}"
        marks = json.load(open(f"{ex}/marks.json"))
        traj = np.load(f"{ex}/poses.npy")
        frames = read_frames(mp4)
        h, w = frames[0].shape[:2]
        L = (len(traj) - 1) // 4 + 1
        lat = latent_poses(traj, L)
        Ks = torch.from_numpy(np.load(f"{ex}/intrinsics.npy")).float()
        Ks = get_Ks_transformed(Ks, 480, 832, h, w, h, w)[0].numpy()

        for seg, (lo, hi) in marks.items():
            if hi - lo < 8:
                continue
            refs = np.linspace(lo, hi - 1, args.n_refs).astype(int)
            for ri in refs:
                p_ref = pose_for_output_frame(lat, ri)
                warper = PoseWarper(Ks, h, w, nominal_depth=4.0)
                warper.set_reference(frames[ri], p_ref)
                for d in DELTAS:
                    j = ri + d
                    if j >= len(frames):
                        continue
                    p_j = pose_for_output_frame(lat, j)
                    t = np.linalg.inv(p_ref) @ p_j
                    yaw = np.degrees(np.arccos(np.clip((np.trace(t[:3, :3]) - 1) / 2, -1, 1)))
                    tnorm = float(np.linalg.norm(t[:3, 3]))
                    H = warper.homography(p_j)
                    cond = float(np.linalg.cond(H))
                    # mean pixel displacement of the homography
                    pts = np.stack([np.arange(0, w, 8).repeat(len(range(0, h, 8))),
                                    np.tile(np.arange(0, h, 8), len(range(0, w, 8))),
                                    np.ones(len(range(0, w, 8)) * len(range(0, h, 8)))], 0).astype(np.float64)
                    q = H @ pts
                    q = q[:2] / q[2]
                    disp = float(np.mean(np.linalg.norm(q - pts[:2], axis=0)))

                    out = warper.warp(p_j)
                    gt = frames[j]
                    # valid mask: 1 where the warp sampled from inside the source
                    mask = cv2.warpPerspective(np.ones((h, w), np.uint8) * 255, H, (w, h),
                                               flags=cv2.INTER_NEAREST,
                                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    valid = float((mask > 127).mean())
                    mse = float(np.mean((out.astype(float) - gt.astype(float)) ** 2))
                    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-9))
                    g1 = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                    g2 = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
                    fl = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    epe = float(np.mean(np.linalg.norm(fl, axis=2)))
                    rgb_o = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                    rgb_g = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
                    dino = 1.0 - float((dino_feat(rgb_o) * dino_feat(rgb_g)).sum())
                    lp = lpips_d(rgb_o, rgb_g)
                    rows.append(dict(clip=tag, scene=scene, seg=seg, ref=int(ri),
                                     d=d, ms=d * MS_PER_FRAME, psnr=psnr, lpips=lp,
                                     dino=dino, epe=epe, valid=valid, cond=cond,
                                     disp=disp, yaw_deg=float(yaw),
                                     trans=tnorm,
                                     edge=band_err(out, gt, 0.12),
                                     center=band_err(out, gt, 0.20)))
        print(f"[eg] {tag}: {sum(1 for r in rows if r['clip']==tag)} samples", flush=True)

    keys = list(rows[0].keys())
    with open(args.out, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[eg] {len(rows)} samples -> {args.out}")

    # ---- summary: mean per (seg, ms) ----
    print(f"\n{'seg':11s} {'ms':>6s} {'psnr':>6s} {'lpips':>6s} {'dino':>6s} "
          f"{'epe':>5s} {'valid':>6s} {'edge':>6s} {'ctr':>6s} {'yaw':>6s} "
          f"{'trans':>6s} {'disp':>6s}")
    segs = sorted({r["seg"] for r in rows})
    for seg in segs:
        for ms in [d * MS_PER_FRAME for d in DELTAS]:
            sub = [r for r in rows if r["seg"] == seg and abs(r["ms"] - ms) < 1]
            if not sub:
                continue
            f = lambda k: np.mean([s[k] for s in sub])
            print(f"{seg:11s} {ms:6.0f} {f('psnr'):6.1f} {f('lpips'):6.3f} "
                  f"{f('dino'):6.3f} {f('epe'):5.2f} {f('valid'):6.3f} "
                  f"{f('edge'):6.1f} {f('center'):6.1f} {f('yaw_deg'):6.2f} "
                  f"{f('trans'):6.3f} {f('disp'):6.2f}")
        print()

    # ---- proxy correlation with LPIPS ----
    print("[eg] proxy correlation with LPIPS (Pearson):")
    lp = np.array([r["lpips"] for r in rows])
    for k in ["valid", "disp", "yaw_deg", "trans", "cond", "epe"]:
        x = np.array([r[k] for r in rows])
        c = float(np.corrcoef(x, lp)[0, 1])
        print(f"    {k:8s} r={c:+.3f}")


if __name__ == "__main__":
    main()
