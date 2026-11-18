#!/usr/bin/env python
"""Full display path: Level 0 immediate pose warp + Level 1 real-frame correction.

Simulates the real-time pipeline at display rate using the frames actually
produced by interactive_loop.py:

  - real frames arrive on the world-model cadence (1 chunk / ~1.25 s)
  - between arrivals, the last real frame is reprojected to the current pose
    (Level 0, sub-ms)
  - when a real frame arrives it is blended in over a short window instead of
    popping (Level 1)
  - the newly arrived real frame becomes the new reprojection reference

  python realtime_display.py --video output/interactive/interactive.mp4 \
      --base examples/00 --fps 60
"""
import argparse
import os
import time

import cv2
import numpy as np

from pose_warp import PoseWarper
from cam_controller import CameraController

# must match interactive_loop.SESSION
SESSION = [
    (3, 0.0, 0.0, 0.0, 0.0),
    (6, 0.6, 0.0, 0.0, 0.0),
    (6, 0.4, 0.0, -0.6, 0.0),
    (5, 0.5, 0.6, 0.0, 0.0),
    (5, 0.0, 0.0, -0.8, 0.0),
    (4, 0.0, 0.0, 0.0, 0.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="output/interactive/interactive.mp4")
    ap.add_argument("--base", default="examples/00")
    ap.add_argument("--area", default="528x288")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--real_cadence", type=float, default=1.25)
    ap.add_argument("--depth", type=float, default=4.0)
    ap.add_argument("--blend", type=int, default=6, help="display frames to blend a new real frame over")
    ap.add_argument("--out_dir", default="output/realtime")
    args = ap.parse_args()

    from wan.utils.cam_utils import get_Ks_transformed
    import torch
    W, H = (int(x) for x in args.area.lower().split("x"))

    cap = cv2.VideoCapture(args.video)
    real = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        real.append(cv2.cvtColor(cv2.resize(f, (W, H)), cv2.COLOR_BGR2RGB))
    cap.release()
    n_chunks = len(real) // 4
    print(f"[rd] {len(real)} real frames = {n_chunks} chunks")

    Ks = torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float()
    Ks = get_Ks_transformed(Ks, 480, 832, H, W, H, W)[0].numpy()
    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])

    disp = int(round(args.fps * args.real_cadence))   # display frames per chunk
    dt = 1.0 / disp
    warper = PoseWarper(Ks, H, W, nominal_depth=args.depth)

    # controller pose history at display rate, and chunk boundary indices
    poses, chunk_start = [], []
    for n, fwd, strafe, yaw, pitch in SESSION:
        ctl.set_input(fwd, strafe, yaw, pitch)
        for _ in range(n * disp):
            ctl.step(dt=dt)
            poses.append(ctl.pose.copy())
        chunk_start.append(len(poses))

    frames, warp_ms, blend_cnt, ref_updates = [], [], 0, 0
    cur_ref = None
    pending_real = None
    blend_left = 0
    for f, pose in enumerate(poses):
        # which real chunk is available now (1-chunk latency model)
        avail = min(f // disp, n_chunks - 1)
        real_frame = real[min(avail * 4, len(real) - 1)]
        real_pose = poses[min(avail * disp, len(poses) - 1)]

        if cur_ref is None or avail != getattr(main, "_last_avail", -1):
            # new real frame arrived -> it becomes the reference, start blending
            main._last_avail = avail
            if cur_ref is not None:
                pending_real = (real_frame, real_pose)
                blend_left = args.blend
                blend_cnt += 1
            cur_ref = (real_frame, real_pose)
            ref_updates += 1

        warper.set_reference(cur_ref[0], cur_ref[1])
        t0 = time.perf_counter()
        out = warper.warp(pose)
        warp_ms.append((time.perf_counter() - t0) * 1000.0)

        if blend_left > 0 and pending_real is not None:
            a = 1.0 - blend_left / (args.blend + 1.0)
            out = cv2.addWeighted(pending_real[0], a, out, 1.0 - a, 0.0)
            blend_left -= 1
        frames.append(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

    os.makedirs(args.out_dir, exist_ok=True)
    mp4 = f"{args.out_dir}/realtime_display.mp4"
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()

    wm = np.array(warp_ms)
    print(f"[rd] {len(frames)} display frames @ {args.fps:.0f}fps "
          f"({len(frames)/args.fps:.1f}s)")
    print(f"[rd] warp cost mean={wm.mean():.2f}ms p90={np.percentile(wm,90):.2f}ms")
    print(f"[rd] real-frame reference updates={ref_updates}  blends={blend_cnt}")
    print(f"[rd] metrics: control-to-warp={wm.mean():.2f}ms | "
          f"control-to-real-frame={args.real_cadence*1000:.0f}ms | "
          f"generation={1000/(args.real_cadence*1000):.2f} chunk/s | "
          f"presentation={args.fps:.0f}fps")
    print(f"[rd] -> {mp4}")

    sheet = frames[::max(1, len(frames)//6)][:6]
    tile = np.concatenate([np.concatenate(sheet[:3], axis=1),
                           np.concatenate(sheet[3:6], axis=1)], axis=0)
    cv2.imwrite(f"{args.out_dir}/realtime_display_sheet.png", tile)


if __name__ == "__main__":
    main()
