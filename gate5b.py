#!/usr/bin/env python
"""Gate 5 (rebuilt): dominant-scalar changepoint transients.

Stage 1 -- A_current SANITY GATE (no CV comparison allowed until this passes)
    Each motion gets one physically meaningful scalar:
        fwd/back    -> radial divergence   div
        yaw         -> mean(dx)            mdx
        strafe      -> mean(dx)            mdx
    Steady segments must reproduce, on A_current:
        yaw+ / yaw-   -> opposite-signed mdx, well above static noise
        fwd+ / fwd-   -> opposite-signed div
        strafe+/-     -> opposite-signed mdx
    If this fails, the metric is invalid and we STOP.

Stage 2 -- transient metrics, type-specific
    reversals (W->S, yaw swap, strafe swap):
        sign_flip_latency, wrong_direction_duration,
        peak_wrong_way_excursion, settling_time, integrated_abs_tracking_error
    release (W->release):
        residual_motion_duration, time_to_50, time_to_20,
        overshoot_after_release, integrated_residual_motion

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python gate5b.py --scene 04 --stage all
"""
import argparse
import gc
import hashlib
import json
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses
from cam_controller import CameraController
from gen_predictive import predict, cv_gain, write_traj

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

STEADY_SEGS = [
    ("static0", 2, dict()),
    ("fwd+", 3, dict(fwd=0.7)),
    ("static1", 1, dict()),
    ("fwd-", 3, dict(fwd=-0.7)),
    ("static2", 1, dict()),
    ("yaw+", 3, dict(yaw=0.7)),
    ("static3", 1, dict()),
    ("yaw-", 3, dict(yaw=-0.7)),
    ("static4", 1, dict()),
    ("str+", 3, dict(strafe=0.7)),
    ("static5", 1, dict()),
    ("str-", 3, dict(strafe=-0.7)),
    ("static6", 1, dict()),
]

CHANGES = [
    ("W_to_S", dict(fwd=0.7), dict(fwd=-0.7), "reversal", "div"),
    ("yaw_swap", dict(yaw=0.7, fwd=0.4), dict(yaw=-0.7, fwd=0.4), "reversal", "mdx"),
    ("strafe_rev", dict(strafe=0.8), dict(strafe=-0.8), "reversal", "mdx"),
    ("W_release", dict(fwd=0.7), dict(), "release", "div"),
]
PRE, POST, SETTLE = 3, 3, 2


def make_ctl(scene):
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max = 6.0
    ctl.cfg.pitch_rate_max = 2.0
    ctl.cfg.v_max = 1.0
    return ctl


def run_segments(ctl, segs, chunk_states, frame_poses):
    marks = {}
    for name, n, inp in segs:
        ctl.set_input(**inp)
        lo = len(frame_poses)
        for _ in range(n):
            chunk_states.append(ctl.clone())
            for _ in range(4):
                ctl.step(dt=0.25)
                frame_poses.append(ctl.pose.copy())
        marks[name] = (lo, len(frame_poses))
    return marks


def build_steady(scene):
    ctl = make_ctl(scene)
    cs, fp = [], []
    marks = run_segments(ctl, STEADY_SEGS, cs, fp)
    return cs, np.stack(fp), marks


def build_change(scene, phase):
    ctl = make_ctl(scene)
    cs, fp, events = [], [], []
    ctl.set_input()
    for _ in range(SETTLE):
        cs.append(ctl.clone())
        for _ in range(4):
            ctl.step(dt=0.25); fp.append(ctl.pose.copy())
    for name, pre, post, kind, scal in CHANGES:
        ctl.set_input(**pre)
        for _ in range(PRE):
            cs.append(ctl.clone())
            for _ in range(4):
                ctl.step(dt=0.25); fp.append(ctl.pose.copy())
        ctl.set_input(**post)
        for j in range(4):
            ctl.step(dt=0.25); fp.append(ctl.pose.copy())
            if j == phase:
                cs.append(ctl.clone())
        for _ in range(POST - 1):
            cs.append(ctl.clone())
            for _ in range(4):
                ctl.step(dt=0.25); fp.append(ctl.pose.copy())
        events.append(dict(name=name, kind=kind, scal=scal,
                           frame=len(fp) - 4 * POST + phase))
    return cs, np.stack(fp), events


def flow_fields(frames):
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    out = [None]
    for i in range(1, len(g)):
        out.append(cv2.calcOpticalFlowFarneback(g[i - 1], g[i], None,
                                                0.5, 3, 15, 3, 5, 1.2, 0))
    return out


def scalars(fl, cx, cy):
    """(radial divergence, mean dx) for one flow field."""
    if fl is None:
        return 0.0, 0.0
    h, w = fl.shape[:2]
    xs = np.arange(w, dtype=np.float32)[None, :] - cx
    ys = np.arange(h, dtype=np.float32)[:, None] - cy
    r2 = (xs ** 2 + ys ** 2)
    num = float((fl[..., 0] * xs + fl[..., 1] * ys).sum())
    div = num / max(float(r2.sum()), 1e-6)
    return div, float(fl[..., 0].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--ts", default="0,250,750")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--phase", type=int, default=1)
    ap.add_argument("--out_dir", default="output/gate5b")
    ap.add_argument("--stage", default="all", choices=["sanity", "transient", "all"])
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    timesteps = [int(x) for x in args.ts.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[g5b] pipe built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    if key not in pipe._t5_cache:
        ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
        pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    def generate(trajdir, frames, mp4):
        if os.path.exists(mp4):
            return
        img_pil = Image.open(f"{trajdir}/image.jpg").convert("RGB")
        t0 = time.perf_counter()
        video = pipe.generate(PROMPT, img_pil, action_path=trajdir, chunk_size=1,
                              max_area=W * H, frame_num=frames,
                              timesteps_index=timesteps, shift=args.shift,
                              seed=args.seed, offload_model=False)
        save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                   nrow=1, normalize=True, value_range=(-1, 1))
        del video
        gc.collect(); torch.cuda.empty_cache()
        print(f"[g5b] {os.path.basename(mp4)} {time.perf_counter()-t0:.0f}s", flush=True)

    def kparams(scene, h, w):
        Ks = get_Ks_transformed(
            torch.from_numpy(np.load(f"examples/{scene}/intrinsics.npy")).float(),
            480, 832, h, w, h, w)[0].numpy()
        return float(Ks[2]), float(Ks[3])

    def frames_of(mp4):
        cap = cv2.VideoCapture(mp4)
        out = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            out.append(f)
        cap.release()
        return out

    # ---------- Stage 1 ----------
    sanity = None
    if args.stage in ("sanity", "all"):
        cs, fp, marks = build_steady(args.scene)
        np.save(f"examples/g5b_{args.scene}_steady_intended.npy", fp)
        d = f"examples/g5b_{args.scene}_steady_A_current"
        N = write_traj(None, [c.pose for c in cs], d, f"examples/{args.scene}")
        mp4 = f"{args.out_dir}/{args.scene}_steady_A_current_s{args.seed}.mp4"
        generate(d, N, mp4)
        frames = frames_of(mp4)
        h, w = frames[0].shape[:2]
        cx, cy = kparams(args.scene, h, w)
        fl = flow_fields(frames)
        seg_val = {}
        for name, (lo, hi) in marks.items():
            vals = [scalars(fl[k], cx, cy) for k in range(lo + 1, min(hi, len(fl)))]
            if not vals:
                continue
            seg_val[name] = (float(np.mean([v[0] for v in vals])),
                             float(np.mean([v[1] for v in vals])))
        print(f"\n[g5b] ===== STAGE 1: A_current sanity gate (scene {args.scene}) =====")
        print(f"{'segment':10s} {'div':>10s} {'mean_dx':>10s}")
        for k, v in seg_val.items():
            print(f"{k:10s} {v[0]:10.4f} {v[1]:10.4f}")
        noise = max(abs(seg_val["static0"][0]), abs(seg_val["static0"][1]), 1e-6)
        checks = [
            ("yaw  mdx sign flip", np.sign(seg_val["yaw+"][1]) != np.sign(seg_val["yaw-"][1])),
            ("yaw  mdx > 3x noise", min(abs(seg_val["yaw+"][1]), abs(seg_val["yaw-"][1])) > 3 * noise),
            ("fwd  div sign flip", np.sign(seg_val["fwd+"][0]) != np.sign(seg_val["fwd-"][0])),
            ("fwd  div > 3x noise", min(abs(seg_val["fwd+"][0]), abs(seg_val["fwd-"][0])) > 3 * noise),
            ("str  mdx sign flip", np.sign(seg_val["str+"][1]) != np.sign(seg_val["str-"][1])),
            ("str  mdx > 3x noise", min(abs(seg_val["str+"][1]), abs(seg_val["str-"][1])) > 3 * noise),
        ]
        sanity = bool(all(bool(c[1]) for c in checks))
        checks = [(n, bool(o)) for n, o in checks]
        for n, ok in checks:
            print(f"  [{'OK ' if ok else 'FAIL'}] {n}")
        print(f"  -> SANITY {'PASS' if sanity else 'FAIL'}")
        if not sanity:
            print("[g5b] sanity gate FAILED -- refusing to compare CV. STOP.")
            json.dump(dict(sanity=False, seg=seg_val, checks=checks),
                      open(f"{args.out_dir}/sanity.json", "w"), indent=1)
            return
        json.dump(dict(sanity=True, seg=seg_val, checks=checks),
                  open(f"{args.out_dir}/sanity.json", "w"), indent=1)

    if args.stage == "sanity":
        return

    # ---------- Stage 2 ----------
    print(f"\n[g5b] ===== STAGE 2: transients (phase {args.phase}) =====")
    cs, fp, events = build_change(args.scene, args.phase)
    np.save(f"examples/g5b_{args.scene}_cp_intended.npy", fp)
    gain = cv_gain(cs, 1.0)
    methods = ["A_current", "B_cv", "E_scaled"]
    for mode in methods:
        lats = [predict(cs, c, mode, 1.0, gain=gain) for c in range(len(cs))]
        d = f"examples/g5b_{args.scene}_p{args.phase}_{mode}"
        N = write_traj(None, lats, d, f"examples/{args.scene}")
        generate(d, N, f"{args.out_dir}/{args.scene}_p{args.phase}_{mode}_s{args.seed}.mp4")

    # reference intended scalars
    fi = frames_of(f"{args.out_dir}/{args.scene}_p{args.phase}_A_current_s{args.seed}.mp4")
    h, w = fi[0].shape[:2]
    cx, cy = kparams(args.scene, h, w)
    Pint = interpolate_camera_poses(
        np.linspace(0, len(fp) - 1, len(fp)),
        torch.from_numpy(fp[:, :3, :3]).float(),
        torch.from_numpy(fp[:, :3, 3]).float(),
        np.linspace(0, len(fp) - 1, len(fi))).numpy()
    # analytic scalars from the intended poses
    fx = float(np.load(f"examples/{args.scene}/intrinsics.npy")[0][0]) * (w / 832.0)
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])
    Ki = np.linalg.inv(K)
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 0).astype(np.float64)
    an = [None]
    for k in range(1, len(Pint)):
        R = Pint[k][:3, :3] @ Pint[k - 1][:3, :3].T
        H = K @ R @ Ki
        q = H @ pts
        q = q[:2] / q[2]
        an.append((q - pts[:2]).T.reshape(h, w, 2).astype(np.float32))

    rows = []
    print(f"{'change':11s} {'kind':9s} {'method':10s} "
          f"{'flip':>5s} {'wrong':>6s} {'peakW':>7s} {'settle':>7s} "
          f"{'iATE':>7s} {'resid':>6s} {'t50':>5s} {'t20':>5s} {'osRel':>6s}")
    for mode in methods:
        fg = flow_fields(frames_of(
            f"{args.out_dir}/{args.scene}_p{args.phase}_{mode}_s{args.seed}.mp4"))
        n = min(len(fg), len(an))
        for ev in events:
            F, scal = ev["frame"], ev["scal"]
            pre = [scalars(fg[k], cx, cy)[0 if scal == "div" else 1]
                   for k in range(max(1, F - 8), F)]
            pre_sign = np.sign(np.mean(pre)) if pre else 0
            post = [scalars(fg[k], cx, cy)[0 if scal == "div" else 1]
                    for k in range(F, min(F + 14, n))]
            ref = [scalars(an[k], cx, cy)[0 if scal == "div" else 1]
                   for k in range(F, min(F + 14, n))]
            post = np.array(post); ref = np.array(ref)
            if ev["kind"] == "reversal":
                tgt = -pre_sign
                wrong = int(np.sum(np.sign(post) == pre_sign)) if pre_sign else 0
                flip = next((i for i, v in enumerate(post) if np.sign(v) == tgt), len(post))
                pw = float(np.max(np.abs(post[:flip + 1]))) if flip > 0 else 0.0
                settle = next((i for i, v in enumerate(post)
                               if i > flip and abs(v) > 0.5 * abs(np.mean(ref))), len(post))
                iate = float(np.mean(np.abs(post - ref)))
                row = dict(flip=flip, wrong=wrong, peakW=pw, settle=int(settle),
                           iate=iate, resid=float("nan"), t50=float("nan"),
                           t20=float("nan"), osRel=float("nan"))
            else:
                base = abs(np.mean(ref)) if abs(np.mean(ref)) > 1e-6 else 1.0
                mag = np.abs(post)
                resid = int(np.sum(mag > 0.2 * base))
                t50 = next((i for i, v in enumerate(mag) if v < 0.5 * base), len(mag))
                t20 = next((i for i, v in enumerate(mag) if v < 0.2 * base), len(mag))
                osr = float(np.max(mag) / base) if base else float("nan")
                row = dict(flip=float("nan"), wrong=float("nan"), peakW=float("nan"),
                           settle=float("nan"), iate=float("nan"), resid=resid,
                           t50=int(t50), t20=int(t20), osRel=osr)
            rows.append(dict(change=ev["name"], kind=ev["kind"], method=mode, **row))
            print(f"{ev['name']:11s} {ev['kind']:9s} {mode:10s} "
                  f"{row['flip']!s:>5s} {row['wrong']!s:>6s} {row['peakW']:7.2f} "
                  f"{row['settle']!s:>7s} {row['iate']:7.3f} {row['resid']!s:>6s} "
                  f"{row['t50']!s:>5s} {row['t20']!s:>5s} {row['osRel']:6.3f}")

    print("\n[g5b] ===== paired summary (CV / E_scaled vs A_current) =====")
    for mode in ["B_cv", "E_scaled"]:
        print(f"  {mode}:")
        for ev in events:
            a = [r for r in rows if r["change"] == ev["name"] and r["method"] == "A_current"][0]
            b = [r for r in rows if r["change"] == ev["name"] and r["method"] == mode][0]
            if ev["kind"] == "reversal":
                print(f"    {ev['name']:11s} dWrong={b['wrong']-a['wrong']:+d}f "
                      f"dFlip={b['flip']-a['flip']:+d}f "
                      f"dPeak={100*(b['peakW']/max(a['peakW'],1e-6)-1):+.1f}% "
                      f"dIATE={100*(b['iate']/max(a['iate'],1e-9)-1):+.1f}%")
            else:
                print(f"    {ev['name']:11s} dResid={b['resid']-a['resid']:+d}f "
                      f"dT50={b['t50']-a['t50']:+d}f dT20={b['t20']-a['t20']:+d}f "
                      f"dOsRel={100*(b['osRel']/max(a['osRel'],1e-6)-1):+.1f}%")
    json.dump(rows, open(f"{args.out_dir}/transient_p{args.phase}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
