#!/usr/bin/env python
"""§42C-2: viewpoint tolerance sweep for render re-identification.

§42C showed that after 47 chunks out-of-FOV with a ~7.5 deg viewpoint shift, the
world state survives (existence / state / no duplicate / no wrong-ID) but render
re-identification FAILS: the target's self-similarity drops below the 0.60
existence threshold and it is declared a NEW object.

That run mixed two variables (invisibility duration AND viewpoint shift). This
sweep isolates the viewpoint shift: everything else is held fixed and only the
return yaw changes.

Poses are synthesized DIRECTLY (fov_pose_synth.py) because CameraController's
rate-limited yaw is ~4x asymmetric between directions, which made dYaw
uncontrollable (a 7.5 deg floor, or 23 deg+ whenever the target returned).

Offline gate (fov_pose_synth.py):
    theta_ret  0.0 .. 7.0  ->  out>=50, base=5, ret=8-9, dYaw exact, du 7..54 px
    theta_ret  7.5+        ->  target no longer returns

Reported per angle:
    self_similarity(C), best_other_similarity, identity_margin,
    existence flag, correct_reid, reacquisition_latency, du/dv

The key diagnostic is the SHAPE of the two curves:
    self_similarity vs dYaw
    identity_margin  vs dYaw
If self drops below threshold while the margin stays clearly positive, the
failure is an over-rigid ABSOLUTE threshold (fixable with a persistent prior).
If the margin collapses toward zero too, it is genuine representation confusion
(needs multi-view / view-conditioned anchors, i.e. §43P).

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python tolerance_sweep.py --scene 04 --seed 42
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange
from object_permanence import patch
from object_state import PersistentObjectStore

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
SKY_BB = (0.05, 0.02, 0.35, 0.16)
OBJECTS = {
    "A_left":  dict(bb=(0.28, 0.56, 0.52, 0.90), kind="crack", target=False),
    "B_right": dict(bb=(0.68, 0.42, 0.88, 0.72), kind="intact", target=False),
    "C_ctrl":  dict(bb=(0.80, 0.10, 0.94, 0.40), kind="hole", target=True),
}
TARGET = "C_ctrl"
THETA_OUT = -55.0
BASELINE_N, DEPART_N, HOLD_N, RETURN_N, OBS_N = 5, 6, 30, 6, 8
ID_EXIST = 0.60          # SAME threshold as §42C -- do not relax it
ANGLES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
DEPTH = 5.0
VIS_LO, VIS_HI = 0.02, 0.98
OUT_LO, OUT_HI = 0.00, 1.00


def rot_y(deg):
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], float)


def slerp_R(Ra, Rb, t):
    dR = Ra.T @ Rb
    tr = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    ang = math.acos(tr)
    if ang < 1e-8:
        return Ra.copy()
    ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                   dR[1, 0] - dR[0, 1]]) / (2.0 * math.sin(ang))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return Ra @ (np.eye(3) + math.sin(ang * t) * K +
                 (1 - math.cos(ang * t)) * (K @ K))


def synth(scene, theta_ret):
    p = np.load(f"examples/{scene}/poses.npy")
    R0 = p[0, :3, :3].copy()
    t0 = p[0, :3, 3].copy()
    Rout = R0 @ rot_y(THETA_OUT)
    Rret = R0 @ rot_y(-theta_ret)

    def pose(R):
        P = np.eye(4)
        P[:3, :3] = R
        P[:3, 3] = t0
        return P

    fr = [pose(R0) for _ in range(BASELINE_N * 4)]
    for i in range(DEPART_N * 4):
        fr.append(pose(slerp_R(R0, Rout, (i + 1) / (DEPART_N * 4))))
    fr += [pose(Rout) for _ in range(HOLD_N * 4)]
    for i in range(RETURN_N * 4):
        fr.append(pose(slerp_R(Rout, Rret, (i + 1) / (RETURN_N * 4))))
    fr += [pose(Rret) for _ in range(OBS_N * 4)]
    traj = np.stack(fr)
    # conditioning/patchify needs frames_n == 1 (mod 4)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n]


def _cracks(frame, bb, n, th, seed):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    out = frame.copy()
    rng = np.random.RandomState(seed)
    seg = out[y0:y1, x0:x1]
    hh, ww = seg.shape[:2]
    for _ in range(n):
        pts = [(rng.randint(0, max(1, ww)), rng.randint(0, max(1, hh)))]
        for _ in range(4):
            pts.append((int(np.clip(pts[-1][0] + rng.randint(-ww // 5, ww // 5),
                                    0, ww - 1)),
                        int(np.clip(pts[-1][1] + rng.randint(0, hh // 3),
                                    0, hh - 1))))
        cv2.polylines(out[y0:y1, x0:x1], [np.array(pts, np.int32)], False,
                      (12, 10, 10), th)
    return out


def make_correction(frame, bb, kind):
    if kind == "intact":
        return frame
    if kind == "crack":
        return _cracks(frame, bb, 10, 3, 3)
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    bx0, by0, bx1, by1 = [int(SKY_BB[0] * W), int(SKY_BB[1] * H),
                          int(SKY_BB[2] * W), int(SKY_BB[3] * H)]
    out = frame.copy()
    hole = cv2.resize(frame[by0:by1, bx0:bx1], (x1 - x0, y1 - y0))
    rim = max(2, (x1 - x0) // 12)
    out[y0 + rim:y1 - rim, x0 + rim:x1 - rim] = \
        hole[rim:hole.shape[0] - rim, rim:hole.shape[1] - rim]
    return out


def vram_free_mb():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--angles", default=",".join(str(a) for a in ANGLES))
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/tolerance")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    names = list(OBJECTS.keys())
    angles = [float(x) for x in args.angles.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[ts] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lh_ = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads
    from eval_two_layer import load_models, dino_feat
    load_models()

    def dino_np(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    # shared trajectory geometry (only the return pose differs per angle)
    traj0 = synth(scene, angles[0])
    frames_n = len(traj0)
    n_lat = (frames_n - 1) // 4 + 1
    print(f"[ts] trajectory {frames_n} frames -> {n_lat} chunks; "
          f"{len(angles)} angles", flush=True)
    d = f"examples/ts_{scene}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch_sz[1] * patch_sz[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch_sz[2] * patch_sz[2])
    h = lat_h * vae_stride[1]; w = lat_w * vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = int(math.ceil(fsl / pipe.sp_size)) * pipe.sp_size
    kv_size = fsl * 6
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    K = Ks.cpu().numpy().astype(np.float64)
    fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
    LR = {}
    for nm in names:
        bb = OBJECTS[nm]["bb"]
        LR[nm] = (int(bb[1] * lat_h), int(bb[3] * lat_h),
                  int(bb[0] * lat_w), int(bb[2] * lat_w))

    def build_y(first):
        with torch.no_grad():
            z = pipe.vae.encode([torch.concat([
                first, torch.zeros(3, frames_n - 1, h, w)], dim=1).to(dev)])[0]
        m = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
        m[:, 1:] = 0
        m = torch.concat([torch.repeat_interleave(m[:, 0:1], repeats=4, dim=1),
                          m[:, 1:]], dim=1)
        m = m.view(1, m.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        return torch.concat([m, z]).detach()

    y = build_y(torch.nn.functional.interpolate(
        img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1))
    ref_img = np.array(img_pil.resize((w, h), Image.BICUBIC))
    ys = {}
    for nm in names:
        if OBJECTS[nm]["kind"] == "intact":
            ys[nm] = None
        else:
            corr = make_correction(ref_img, OBJECTS[nm]["bb"], OBJECTS[nm]["kind"])
            ys[nm] = build_y(TF.to_tensor(Image.fromarray(corr)).sub_(0.5)
                             .div_(0.5).unsqueeze(0).transpose(0, 1))
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[ts] setup done, free {vram_free_mb():.0f}MiB", flush=True)

    def run_gen(y_cond, rel_all, c2w, max_chunks=None):
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, lats = [], []
        N = n_lat if max_chunks is None else min(n_lat, max_chunks)
        for cid in range(N):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            p = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                          c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y_cond.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * fsl,
                  "max_attention_size": kv_size, "frame_seqlen": fsl}
            for ti in range(len(pipe.scheduler.timesteps[[0, 250, 750]])):
                ts = pipe.scheduler.timesteps[[0, 250, 750]]
                with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                    npred = pipe.model(
                        x=[cur.to(dev)], t=torch.stack([ts[ti]]).to(dev),
                        cross_attn_first_call=not pipe._cross_attn_initialized,
                        **kw)[0]
                    pipe._cross_attn_initialized = True
                    x0 = pipe._convert_flow_pred_to_x0(
                        flow_pred=npred, xt=cur, timestep=ts[ti],
                        scheduler=pipe.scheduler)
                    if ti < len(ts) - 1:
                        cur = pipe.scheduler.add_noise(
                            x0, torch.randn(x0.shape, generator=g,
                                            device=x0.device, dtype=x0.dtype),
                            ts[ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([ts[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            with torch.no_grad():
                fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                      parallel=False, show_progress_bar=False)
            outs.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                         * 255.0).clip(0, 255).astype(np.uint8))
            lats.append(x0.detach().float().cpu())
        del self_kv, cross_kv
        gc.collect(); torch.cuda.empty_cache()
        return outs, lats

    # anchors from D0
    print("[ts] === D0 ===", flush=True)
    c2w0 = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj0[:, :3, :3]).float(),
        torch.from_numpy(traj0[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel0 = compute_relative_poses(c2w0, framewise=True)
    D0f, D0_lat = run_gen(y, rel0, c2w0)
    A_anc, TPL = {}, {}
    for nm in names:
        y0, y1, x0_, x1_ = LR[nm]
        if OBJECTS[nm]["kind"] == "intact":
            A_anc[nm] = D0_lat[2][:, :, y0:y1, x0_:x1_].clone().to(dev)
            TPL[nm] = patch(D0f[2], OBJECTS[nm]["bb"])
            continue
        print(f"[ts] === canonicalise {nm} ===", flush=True)
        f, l = run_gen(ys[nm], rel0, c2w0, max_chunks=args.canon_chunks)
        ci = min(2, len(l) - 1)
        A_anc[nm] = l[ci][:, :, y0:y1, x0_:x1_].clone().to(dev)
        TPL[nm] = patch(f[ci], OBJECTS[nm]["bb"])

    # sweep
    results = []
    for theta_ret in angles:
        traj = synth(scene, theta_ret)
        c2w = interpolate_camera_poses(
            np.linspace(0, frames_n - 1, frames_n),
            torch.from_numpy(traj[:, :3, :3]).float(),
            torch.from_numpy(traj[:, :3, 3]).float(),
            np.linspace(0, frames_n - 1, n_lat)).to(dev)
        rel = compute_relative_poses(c2w, framewise=True)
        c2w_np = c2w.cpu().numpy()
        ref = c2w_np[2]
        bb = OBJECTS[TARGET]["bb"]
        uc = (bb[0] + bb[2]) / 2 * w
        vc = (bb[1] + bb[3]) / 2 * h
        Xc = np.array([(uc - cx) / fx, (vc - cy) / fy, 1.0]) * DEPTH
        Xw = ref[:3, :3] @ Xc + ref[:3, 3]

        def proj(cid):
            P = c2w_np[cid]
            X = P[:3, :3].T @ (Xw - P[:3, 3])
            if X[2] <= 1e-3:
                return None
            return ((fx * X[0] / X[2] + cx) / w, (fy * X[1] / X[2] + cy) / h)

        def vis(cid):
            uv = proj(cid)
            if uv is None:
                return False, None
            u, v = uv
            if u < OUT_LO or u > OUT_HI or v < OUT_LO or v > OUT_HI:
                return False, uv
            return (VIS_LO <= u <= VIS_HI and VIS_LO <= v <= VIS_HI), uv

        vis_seq = [vis(c)[0] for c in range(n_lat)]
        vis_chunks = [c for c in range(n_lat) if vis_seq[c]]
        out_chunks = [c for c in range(n_lat) if not vis_seq[c]]
        ret_chunks = [c for c in vis_chunks if out_chunks and
                      c > max(out_chunks)] if out_chunks else []
        if not ret_chunks:
            print(f"[ts] theta_ret={theta_ret:4.1f}: target never returns "
                  f"({len(out_chunks)} out) -- skipped", flush=True)
            results.append(dict(theta_ret=theta_ret, out=len(out_chunks),
                                ret=0, reid=False, note="no return"))
            continue

        print(f"[ts] === theta_ret={theta_ret:.1f} deg, generating "
              f"(out={len(out_chunks)}, ret={len(ret_chunks)}) ===", flush=True)
        frames, _ = run_gen(y, rel, c2w)

        store = PersistentObjectStore()
        ids = {}
        for nm in names:
            st = store.register("obj", np.eye(4), [1, 1, 1],
                                anchor=dino_np(TPL[nm]), t=0.0,
                                gameplay_state=dict(uv=((OBJECTS[nm]["bb"][0] +
                                                         OBJECTS[nm]["bb"][2]) / 2,
                                                        (OBJECTS[nm]["bb"][1] +
                                                         OBJECTS[nm]["bb"][3]) / 2)))
            ids[nm] = st.persistent_id
            store.set_anchor_for_state(st.persistent_id, "home",
                                       feature=dino_np(TPL[nm]))
            store.set_state(st.persistent_id, state_key="home", open=False)

        Hf, Wf = frames[0].shape[:2]
        selfs, others, margins, exs, reids = [], [], [], [], []
        for cid in ret_chunks:
            okv, uv = vis(cid)
            bh = (bb[3] - bb[1]) / 2
            bw = (bb[2] - bb[0]) / 2
            box = (uv[0] - bw, uv[1] - bh, uv[0] + bw, uv[1] + bh)
            op = patch(frames[cid], box)
            o = dino_np(op)
            s_self = float((o * dino_np(TPL[TARGET])).sum())
            best = max((float((o * dino_np(TPL[nm])).sum())
                        for nm in names if nm != TARGET), default=-1.0)
            selfs.append(s_self); others.append(best)
            margins.append(s_self - best)
            exs.append(s_self > ID_EXIST)
            for nm in names:
                uvo = proj(cid) if nm == TARGET else (
                    (OBJECTS[nm]["bb"][0] + OBJECTS[nm]["bb"][2]) / 2,
                    (OBJECTS[nm]["bb"][1] + OBJECTS[nm]["bb"][3]) / 2)
                store.set_prediction(ids[nm], uv=uvo)
            dec = store.reacquire([dict(anchor=o, uv=uv,
                                        world_transform=np.eye(4),
                                        bounds=[1, 1, 1])],
                                  t=cid * 0.25, motion_aware=True)[0]
            reids.append(dec["id"] == ids[TARGET])
        results.append(dict(
            theta_ret=theta_ret, out=len(out_chunks), ret=len(ret_chunks),
            self_sim=float(np.mean(selfs)), best_other=float(np.mean(others)),
            margin=float(np.mean(margins)), margin_min=float(np.min(margins)),
            existence=float(np.mean(exs)), reid=float(np.mean(reids)),
            reid_any=bool(any(reids)), n_eval=len(ret_chunks)))
        r = results[-1]
        print(f"[ts]   dYaw={theta_ret:4.1f}  self={r['self_sim']:.3f} "
              f"other={r['best_other']:.3f} margin={r['margin']:+.3f} "
              f"ex={r['existence']*100:.0f}% reid={r['reid']*100:.0f}%",
              flush=True)
        np.save(f"{args.out_dir}/frames_{theta_ret:.1f}.npy",
                np.stack(frames[:max(ret_chunks) + 1]))
        del frames
        gc.collect(); torch.cuda.empty_cache()

    print("\n[ts] ===== §42C-2 viewpoint tolerance sweep =====")
    print(f"  {'dYaw':>6s} {'out':>4s} {'ret':>4s} {'self':>7s} {'other':>7s} "
          f"{'margin':>8s} {'existence':>10s} {'re-ID':>7s}")
    for r in results:
        if r.get("ret", 0) == 0:
            print(f"  {r['theta_ret']:6.1f} {r['out']:4d} {0:4d} "
                  f"{'n/a':>7s} {'n/a':>7s} {'n/a':>8s} {'n/a':>10s} "
                  f"{'no-ret':>7s}")
            continue
        print(f"  {r['theta_ret']:6.1f} {r['out']:4d} {r['ret']:4d} "
              f"{r['self_sim']:7.3f} {r['best_other']:7.3f} {r['margin']:+8.3f} "
              f"{r['existence']*100:9.0f}% {r['reid']*100:6.0f}%")

    ok = [r for r in results if r.get("ret", 0) > 0]
    passing = [r["theta_ret"] for r in ok if r["reid"] >= 0.9]
    failing = [r["theta_ret"] for r in ok if r["reid"] < 0.9]
    print(f"\n  re-ID PASS at dYaw: {passing}")
    print(f"  re-ID FAIL at dYaw: {failing}")
    if passing:
        print(f"  -> render viewpoint tolerance theta_reid >= {max(passing):.1f} deg")
    failing_sorted = sorted(failing)
    passing_sorted = sorted(passing)
    if passing_sorted and failing_sorted and max(passing_sorted) < min(failing_sorted):
        print(f"  -> boundary between {max(passing_sorted):.1f} and "
              f"{min(failing_sorted):.1f} deg")

    # failure mechanism
    print("\n[ts] ===== failure mechanism =====")
    print("  if self < threshold while margin stays clearly positive")
    print("    -> over-rigid ABSOLUTE threshold (persistent prior can fix it)")
    print("  if margin also collapses toward 0/negative")
    print("    -> genuine representation confusion (needs multi-view anchors)")
    for r in ok:
        mx = r["margin_min"]
        verdict = ("absolute-threshold limited" if mx > 0.05 and
                   r["existence"] < 0.9 else
                   ("representation confusion" if mx <= 0.05 else "ok"))
        print(f"    dYaw={r['theta_ret']:4.1f}: self={r['self_sim']:.3f} "
              f"margin_min={mx:+.3f} -> {verdict}")

    json.dump(dict(results=results, passing=passing, failing=failing,
                   id_exist_threshold=ID_EXIST, theta_out=THETA_OUT,
                   geometry=dict(baseline=BASELINE_N, depart=DEPART_N,
                                 hold=HOLD_N, ret_phase=RETURN_N, obs=OBS_N)),
              open(f"{args.out_dir}/tolerance.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
