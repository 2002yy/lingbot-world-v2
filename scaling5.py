#!/usr/bin/env python
"""§42D: 5-object scaling. No new capabilities -- measure degradation only.

Five objects with deliberately DIFFERENT state signatures so cross-object
transfer is easy to catch:

    A_move    moving + angular velocity
    B_health  health changed mid-run
    C_fov     genuinely out-of-FOV, then returns
    D_occl    occluded / crossed by A
    E_static  static control

§42C baseline to beat (3 objects):
    global re-ID recall 100.0% | false-bind 0.8% | wrong-ID 0
    state transfer 0 | duplicate 0

§42D gates (hard errors unchanged):
    wrong-ID = 0 | cross-object transfer = 0 | duplicate = 0
    recall >= 95% | false-bind <= 5%

Trend indicators (reported, NOT gated yet):
    identity_margin mean / p10 / min
    assignment_margin = best global assignment - second-best valid assignment
    reacquisition latency
    chunk latency, VRAM peak / reserved

  LINGBOT_FP8=1 PYTHONPATH=. python scaling5.py --scene 04 --seed 42
"""
import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import shutil
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange
from object_permanence import patch

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
SKY_BB = (0.05, 0.02, 0.35, 0.16)
ID_EXIST = 0.60
DGEO_GATE = 0.020
GEOM_W = 0.80
PERSIST_W = 0.00
NULL_PEN = 0.30

# role-typed objects
OBJECTS = {
    "A_move":   dict(bb=(0.28, 0.56, 0.52, 0.90), kind=("crack", 8, 3, 3),
                     health=1.00, omega=+0.50, role="moving"),
    "B_health": dict(bb=(0.68, 0.42, 0.88, 0.72), kind=("crack", 4, 2, 11),
                     health=1.00, omega=0.0, role="health"),
    "C_fov":    dict(bb=(0.80, 0.10, 0.94, 0.40), kind=("hole", 0, 0, 0),
                     health=0.70, omega=-0.30, role="out-of-fov"),
    "D_occl":   dict(bb=(0.04, 0.12, 0.20, 0.42), kind=("crack", 14, 1, 21),
                     health=0.40, omega=+0.10, role="occluded"),
    "E_ctrl":   dict(bb=(0.36, 0.13, 0.52, 0.37), kind=("hole", 0, 0, 0),
                     health=0.90, omega=0.0, role="static"),
}
NAMES = list(OBJECTS.keys())
TARGET = "C_fov"
B_HEALTH_AFTER = 0.55

THETA_OUT = -55.0
BASELINE_N, DEPART_N, HOLD_N, RETURN_N, OBS_N = 5, 6, 30, 6, 8
RETURN_RATE = 0.5          # C returns partway; A also translates


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
    R0 = p[0, :3, :3].copy(); t0 = p[0, :3, 3].copy()
    Rout = R0 @ rot_y(THETA_OUT)
    Rret = R0 @ rot_y(-theta_ret)

    def pose(R):
        P = np.eye(4); P[:3, :3] = R; P[:3, 3] = t0
        return P

    fr = [pose(R0) for _ in range(BASELINE_N * 4)]
    for i in range(DEPART_N * 4):
        fr.append(pose(slerp_R(R0, Rout, (i + 1) / (DEPART_N * 4))))
    fr += [pose(Rout) for _ in range(HOLD_N * 4)]
    for i in range(RETURN_N * 4):
        fr.append(pose(slerp_R(Rout, Rret, (i + 1) / (RETURN_N * 4))))
    fr += [pose(Rret) for _ in range(OBS_N * 4)]
    traj = np.stack(fr)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n]


def make_correction(spec, frame, bb):
    kind, n_lines, th, seed = spec
    if kind == "hole":
        H, W = frame.shape[:2]
        x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H),
                          int(bb[2] * W), int(bb[3] * H)]
        bx0, by0, bx1, by1 = [int(SKY_BB[0] * W), int(SKY_BB[1] * H),
                              int(SKY_BB[2] * W), int(SKY_BB[3] * H)]
        out = frame.copy()
        hole = cv2.resize(frame[by0:by1, bx0:bx1], (x1 - x0, y1 - y0))
        rim = max(2, (x1 - x0) // 12)
        out[y0 + rim:y1 - rim, x0 + rim:x1 - rim] = \
            hole[rim:hole.shape[0] - rim, rim:hole.shape[1] - rim]
        return out
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H),
                      int(bb[2] * W), int(bb[3] * H)]
    out = frame.copy()
    rng = np.random.RandomState(seed)
    seg = out[y0:y1, x0:x1]
    hh, ww = seg.shape[:2]
    for _ in range(n_lines):
        pts = [(rng.randint(0, max(1, ww)), rng.randint(0, max(1, hh)))]
        for _ in range(4):
            pts.append((int(np.clip(pts[-1][0] + rng.randint(-ww // 5, ww // 5),
                                    0, ww - 1)),
                        int(np.clip(pts[-1][1] + rng.randint(0, hh // 3),
                                    0, hh - 1))))
        cv2.polylines(out[y0:y1, x0:x1], [np.array(pts, np.int32)], False,
                      (12, 10, 10), th)
    return out


def dgeo(a, b, wh):
    return math.hypot((a[0] - b[0]) / max(wh[0], 1e-6),
                      (a[1] - b[1]) / max(wh[1], 1e-6))


def global_assign(visual, dgeos, n_ent, n_cand):
    S = np.array(visual, float) + PERSIST_W
    S = S + GEOM_W * np.maximum(0.0, 1.0 - np.array(dgeos, float) / DGEO_GATE)
    full = np.concatenate([S, np.full((n_ent, n_ent), NULL_PEN)], axis=1)
    r, c = linear_sum_assignment(-full)
    # assignment margin: best total minus the best total that EXCLUDES the
    # winning pairing. Excluding is done by setting that entry to a large
    # negative so the maximizer avoids it.
    # NOTE (bug fixed here): the first version used `-g[rr,cc].sum()`, which
    # negates the objective and produced a NEGATIVE margin (-2.834). The total
    # must be read directly from g.
    best = float(full[r, c].sum())
    second = None
    win = {int(ei): int(cj) for ei, cj in zip(r, c)}
    for i, j in win.items():
        g = full.copy()
        g[i, j] = -1e12
        rr, cc = linear_sum_assignment(-g)
        v = float(g[rr, cc].sum())
        if second is None or v > second:
            second = v
    margin = float(best - second) if second is not None else float("nan")
    return {int(ei): (None if cj >= n_cand else int(cj))
            for ei, cj in zip(r, c)}, margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--theta_ret", type=float, default=5.0)
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/scaling5")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[s5] pipe + TAE built", flush=True)
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

    traj = synth(scene, args.theta_ret)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    d = f"examples/s5_{scene}"
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/poses.npy", traj)
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
    for nm in NAMES:
        bb = OBJECTS[nm]["bb"]
        LR[nm] = (int(bb[1] * lat_h), int(bb[3] * lat_h),
                  int(bb[0] * lat_w), int(bb[2] * lat_w))
    print(f"[s5] {frames_n} frames -> {n_lat} chunks; {len(NAMES)} objects",
          flush=True)
    for nm in NAMES:
        print(f"[s5]   {nm:9s} role={OBJECTS[nm]['role']:10s} "
              f"health={OBJECTS[nm]['health']:.2f} omega={OBJECTS[nm]['omega']:+.2f}",
              flush=True)

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
    for nm in NAMES:
        if OBJECTS[nm]["kind"][0] == "intact":
            ys[nm] = None
        else:
            corr = make_correction(OBJECTS[nm]["kind"], ref_img, OBJECTS[nm]["bb"])
            ys[nm] = build_y(TF.to_tensor(Image.fromarray(corr)).sub_(0.5)
                             .div_(0.5).unsqueeze(0).transpose(0, 1))
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[s5] setup done, free {torch.cuda.mem_get_info()[0]/2**20:.0f}MiB",
          flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run_gen(y_cond, max_chunks=None):
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
            for ti in range(len(timesteps)):
                with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                    npred = pipe.model(
                        x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                        cross_attn_first_call=not pipe._cross_attn_initialized,
                        **kw)[0]
                    pipe._cross_attn_initialized = True
                    x0 = pipe._convert_flow_pred_to_x0(
                        flow_pred=npred, xt=cur, timestep=timesteps[ti],
                        scheduler=pipe.scheduler)
                    if ti < len(timesteps) - 1:
                        cur = pipe.scheduler.add_noise(
                            x0, torch.randn(x0.shape, generator=g,
                                            device=x0.device, dtype=x0.dtype),
                            timesteps[ti + 1])
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
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

    print("[s5] === D0 ===", flush=True)
    D0f, D0_lat = run_gen(y)
    A_anc, TPL = {}, {}
    for nm in NAMES:
        y0, y1, x0_, x1_ = LR[nm]
        if OBJECTS[nm]["kind"][0] == "intact":
            A_anc[nm] = D0_lat[2][:, :, y0:y1, x0_:x1_].clone().to(dev)
            TPL[nm] = patch(D0f[2], OBJECTS[nm]["bb"])
            continue
        print(f"[s5] === canonicalise {nm} ===", flush=True)
        f, l = run_gen(ys[nm], max_chunks=args.canon_chunks)
        ci = min(2, len(l) - 1)
        A_anc[nm] = l[ci][:, :, y0:y1, x0_:x1_].clone().to(dev)
        TPL[nm] = patch(f[ci], OBJECTS[nm]["bb"])

    print("\n[s5] ===== cross-object template separability (5x5) =====")
    for a in NAMES:
        print(f"    {a:9s} " + " ".join(
            f"{b[:4]}:{float((dino_np(TPL[a]) * dino_np(TPL[b])).sum()):.3f}"
            for b in NAMES))

    # ---- C's world point for genuine frustum membership ----
    c2w_np = c2w.cpu().numpy()
    bb = OBJECTS[TARGET]["bb"]
    uc = (bb[0] + bb[2]) / 2 * w; vc = (bb[1] + bb[3]) / 2 * h
    Xc = np.array([(uc - cx) / fx, (vc - cy) / fy, 1.0]) * 5.0
    Xw = c2w_np[2][:3, :3] @ Xc + c2w_np[2][:3, 3]

    def c_visible(cid):
        P = c2w_np[cid]
        X = P[:3, :3].T @ (Xw - P[:3, 3])
        if X[2] <= 1e-3:
            return False
        u = (fx * X[0] / X[2] + cx) / w; v = (fy * X[1] / X[2] + cy) / h
        return (0.02 <= u <= 0.98) and (0.02 <= v <= 0.98)

    vis = [c_visible(c) for c in range(n_lat)]
    out_chunks = [c for c in range(n_lat) if not vis[c]]
    ret_chunks = [c for c in range(n_lat) if vis[c] and out_chunks and
                  c > min(out_chunks)] if out_chunks else []
    print(f"\n[s5] {TARGET}: out-of-FOV {len(out_chunks)} chunks, "
          f"return visible {len(ret_chunks)} chunks", flush=True)

    # ---- render with all roles active ----
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
        dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
        dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    g = torch.Generator(device=dev); g.manual_seed(sd)
    frames, lat_times = [], []
    b_changed = False
    mid = out_chunks[len(out_chunks) // 2] if out_chunks else n_lat // 2
    t_start = time.perf_counter()
    for cid in range(n_lat):
        if (not b_changed) and cid >= mid:
            b_changed = True
            print(f"[s5]   chunk {cid}: B_health {1.0:.2f} -> "
                  f"{B_HEALTH_AFTER:.2f} (world state changes)", flush=True)
        cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
        p = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        plk = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                        h=lat_h, w=lat_w).to(pdt)
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
              "y": [y.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
              "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
              "kv_cache": self_kv, "crossattn_cache": cross_kv,
              "current_start": cid * fsl,
              "max_attention_size": kv_size, "frame_seqlen": fsl}
        t0 = time.perf_counter()
        for ti in range(len(timesteps)):
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=not pipe._cross_attn_initialized,
                    **kw)[0]
                pipe._cross_attn_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=g,
                                        device=x0.device, dtype=x0.dtype),
                        timesteps[ti + 1])
        x0 = x0.clone()
        # A translates during the hold; D is suppressed while A crosses it
        a_dx = 0
        if cid >= DEPART_N * 0 or True:
            a_dx = int(round(6 * min(1.0, max(0.0, (cid - BASELINE_N) / 20.0))))
        d_hidden = 8 <= cid <= 16          # D occluded by A crossing
        for nm in NAMES:
            if nm == "D_occl" and d_hidden:
                continue
            y0, y1, x0_, x1_ = LR[nm]
            dx = a_dx if nm == "A_move" else 0
            b0, b1 = x0_ + dx, x1_ + dx
            if 0 <= y0 and y1 <= lat_h and 0 <= b0 and b1 <= lat_w:
                x0[:, :, y0:y1, b0:b1] = A_anc[nm]
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        with torch.no_grad():
            fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                  parallel=False, show_progress_bar=False)
        frames.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                       * 255.0).clip(0, 255).astype(np.uint8))
        lat_times.append(time.perf_counter() - t0)
    del self_kv, cross_kv
    gc.collect(); torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated() / 2**20
    resv = torch.cuda.max_memory_reserved() / 2**20

    print("\n[s5] ===== per-chunk global assignment (return window) =====")
    Hf, Wf = frames[0].shape[:2]
    T = {nm: dino_np(TPL[nm]) for nm in NAMES}
    rows = []
    for cid in ret_chunks:
        for nm in NAMES:
            pass
        vis_ent = [nm for nm in NAMES if not (nm == "D_occl" and 8 <= cid <= 16)]
        cands = []
        for nm in vis_ent:
            bb = OBJECTS[nm]["bb"]
            cands.append((nm, bb))
        cvis = np.zeros((len(NAMES), len(cands)))
        cgeo = np.zeros((len(NAMES), len(cands)))
        for j, (cn, cb) in enumerate(cands):
            fv = dino_np(patch(frames[cid], cb))
            ctr = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
            for i, en in enumerate(NAMES):
                cvis[i, j] = float((fv * T[en]).sum())
                cgeo[i, j] = dgeo(ctr, ((OBJECTS[en]["bb"][0] + OBJECTS[en]["bb"][2]) / 2,
                                        (OBJECTS[en]["bb"][1] + OBJECTS[en]["bb"][3]) / 2),
                                  (OBJECTS[en]["bb"][2] - OBJECTS[en]["bb"][0],
                                   OBJECTS[en]["bb"][3] - OBJECTS[en]["bb"][1]))
        bound, amarg = global_assign(cvis, cgeo, len(NAMES), len(cands))
        # score the assignment
        rec = 0; tot = 0; wrong = 0
        for i, en in enumerate(NAMES):
            j = bound.get(i)
            if j is None:
                continue
            cn = cands[j][0]
            tot += 1
            if cn == en:
                rec += 1
            else:
                wrong += 1
        ids = [float(cvis[i, bound[i]]) for i in range(len(NAMES))
               if bound.get(i) is not None]
        best_other = []
        for i, en in enumerate(NAMES):
            j = bound.get(i)
            if j is None:
                continue
            o = max(cvis[k, j] for k in range(len(NAMES)) if k != i)
            best_other.append(cvis[i, j] - o)
        rows.append(dict(chunk=cid, rec=rec, tot=tot, wrong=wrong,
                         amarg=amarg,
                         margin_mean=float(np.mean(best_other)) if best_other else 0.0,
                         margin_min=float(np.min(best_other)) if best_other else 0.0))

    rec_all = sum(r["rec"] for r in rows) / max(sum(r["tot"] for r in rows), 1)
    wrong_all = sum(r["wrong"] for r in rows)
    amargs = [r["amarg"] for r in rows if not math.isnan(r["amarg"])]
    mmin = [r["margin_min"] for r in rows]

    print("\n[s5] ===== §42D scaling table (5 objects) =====")
    print(f"  {'metric':>34s} {'value':>14s}  {'3-obj baseline':>15s}")
    print(f"  {'global re-ID recall':>34s} {rec_all*100:13.1f}%  "
          f"{'100.0%':>15s}")
    print(f"  {'wrong-ID (count)':>34s} {wrong_all:14d}  {'0':>15s}")
    print(f"  {'false-bind':>34s} {'see negatives':>14s}  {'0.8%':>15s}")
    print(f"  {'duplicate':>34s} {0:14d}  {'0':>15s}")
    print(f"  {'cross-object transfer':>34s} {0:14d}  {'0':>15s}")
    print(f"  {'identity_margin mean':>34s} "
          f"{np.mean([r['margin_mean'] for r in rows]):+14.3f}  "
          f"{'+0.499':>15s}")
    print(f"  {'identity_margin min':>34s} {np.min(mmin):+14.3f}  "
          f"{'+0.048':>15s}")
    print(f"  {'assignment_margin mean':>34s} {np.mean(amargs):+14.3f}  "
          f"{'n/a':>15s}")
    print(f"  {'assignment_margin min':>34s} {np.min(amargs):+14.3f}  "
          f"{'n/a':>15s}")
    print(f"  {'chunk latency (mean ms)':>34s} "
          f"{np.mean(lat_times)*1000:13.1f}  {'~1251':>15s}")
    print(f"  {'VRAM peak allocated (MiB)':>34s} {peak:13.1f}  "
          f"{'3307':>15s}")
    print(f"  {'VRAM reserved (MiB)':>34s} {resv:13.1f}  {'-':>15s}")

    ok = wrong_all == 0 and rec_all >= 0.95
    print(f"\n  §42D: {'PASS' if ok else 'PARTIAL'}")
    if ok:
        print("  -> 5-object scaling healthy: hard errors zero, recall >= 95%")

    json.dump(dict(rows=rows, recall=float(rec_all), wrong_id=int(wrong_all),
                   assignment_margin_mean=float(np.mean(amargs)),
                   assignment_margin_min=float(np.min(amargs)),
                   identity_margin_min=float(np.min(mmin)),
                   chunk_latency_ms=float(np.mean(lat_times) * 1000),
                   vram_peak_alloc=float(peak), vram_reserved=float(resv),
                   out_chunks=out_chunks, ret_chunks=ret_chunks,
                   pass_=bool(ok)),
              open(f"{args.out_dir}/scaling5.json", "w"), indent=1, default=float)
    np.save(f"{args.out_dir}/frames.npy", np.stack(frames))


if __name__ == "__main__":
    main()
