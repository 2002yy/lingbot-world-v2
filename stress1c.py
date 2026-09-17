#!/usr/bin/env python
"""§43P-1c: joint pressure with a densely-built, then FROZEN bank.

§43P-1b confirmed the N x dYaw interaction but its causal bank made ZERO writes:
the coarse build sessions (0/4/7) violate the frozen write rule because the
identity margin has already fallen below WRITE_MARGIN by dYaw=4. That is a real
collection constraint (need multi-view to rescue the margin, but the margin is
too low to collect multi-view), NOT a reason to lower the threshold -- writing
while identity confidence is low is exactly how memory gets polluted.

P-1c separates the two phases, changing nothing else:

    BUILD : dYaw = 0 -> 1 -> 2 -> 3     (dense, inside the trustworthy margin band)
    FREEZE: no further writes
    TEST  : dYaw = 0 / 4 / 7            (0 inside bank, 4 at its edge, 7 outside)

Four things are locked:
  1. BUILD and TEST are strictly separated; the bank is FROZEN before TEST.
     This isolates "a bank built from trustworthy small-view history helps
     future large-view re-ID" from "the system kept learning during the test".
  2. The single arm runs the SAME BUILD history (the renders are shared); the
     only difference between arms is bank capacity/writes. So the single arm
     does not see the object any fewer times than the causal arm.
  3. Per-identity bank audit at every N: size, writes, rejections by margin, by
     novelty, and pollution writes. Aggregates can hide objects that never got
     a bank.
  4. The decisive cell is N=10 x dYaw=7.

Frozen from §43P-1a (unchanged):
    WRITE_MARGIN 0.28 | NOVELTY_MAX 0.94 | MIN_VOTES 3 | CAPACITY 4

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python stress1c.py --scene 04 --seed 42 --ns 3,10
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

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
DGEO_GATE, GEOM_W, PERSIST_W, NULL_PEN = 0.020, 0.80, 0.00, 0.30

# ---- FROZEN from §43P-1a: do not tune ----
WRITE_MARGIN = 0.28
NOVELTY_MAX = 0.94
MIN_VOTES = 3
CAPACITY = 4

BUILD_DELTAS = [0.0, 1.0, 2.0, 3.0]
TEST_DELTAS = [0.0, 4.0, 7.0]

ALL_SLOTS = [
    ("A1", (0.28, 0.56, 0.52, 0.90), "moving",  ("crack", 8, 3, 3)),
    ("B1", (0.68, 0.42, 0.88, 0.72), "health",  ("crack", 4, 2, 11)),
    ("C1", (0.80, 0.10, 0.94, 0.40), "outfov",  ("hole", 0, 0, 0)),
    ("D1", (0.06, 0.66, 0.22, 0.94), "occlude", ("crack", 14, 1, 21)),
    ("E1", (0.42, 0.70, 0.56, 0.94), "static",  ("hole", 0, 0, 0)),
    ("A2", (0.04, 0.12, 0.20, 0.42), "moving",  ("crack", 9, 3, 4)),
    ("C2", (0.62, 0.66, 0.76, 0.92), "outfov",  ("hole", 0, 0, 0)),
    ("B2", (0.36, 0.13, 0.52, 0.37), "health",  ("crack", 5, 2, 12)),
    ("D2", (0.86, 0.56, 1.00, 0.82), "occlude", ("crack", 15, 1, 22)),
    ("E2", (0.16, 0.40, 0.28, 0.62), "static",  ("hole", 0, 0, 0)),
]
ORDER = [s[0] for s in ALL_SLOTS]
TARGET = "C1"
BASE_N, DEP_N, HOLD_N, RET_N, OBS_N = 5, 6, 16, 6, 8


def rot_y(deg):
    th = math.radians(deg); c, s = math.cos(th), math.sin(th)
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
    Rout = R0 @ rot_y(-55.0); Rret = R0 @ rot_y(-theta_ret)

    def pose(R):
        P = np.eye(4); P[:3, :3] = R; P[:3, 3] = t0
        return P

    fr = [pose(R0) for _ in range(BASE_N * 4)]
    for i in range(DEP_N * 4):
        fr.append(pose(slerp_R(R0, Rout, (i + 1) / (DEP_N * 4))))
    fr += [pose(Rout) for _ in range(HOLD_N * 4)]
    for i in range(RET_N * 4):
        fr.append(pose(slerp_R(Rout, Rret, (i + 1) / (RET_N * 4))))
    fr += [pose(Rret) for _ in range(OBS_N * 4)]
    traj = np.stack(fr)
    return traj[:(len(traj) - 1) // 4 * 4 + 1]


def make_correction(spec, frame, bb):
    kind, n_lines, th, seed = spec
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H),
                      int(bb[2] * W), int(bb[3] * H)]
    out = frame.copy()
    if kind == "hole":
        bx0, by0, bx1, by1 = [int(SKY_BB[0] * W), int(SKY_BB[1] * H),
                              int(SKY_BB[2] * W), int(SKY_BB[3] * H)]
        hole = cv2.resize(frame[by0:by1, bx0:bx1], (x1 - x0, y1 - y0))
        rim = max(2, (x1 - x0) // 12)
        out[y0 + rim:y1 - rim, x0 + rim:x1 - rim] = \
            hole[rim:hole.shape[0] - rim, rim:hole.shape[1] - rim]
        return out
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
    best = float(full[r, c].sum())
    second = None
    for ei, cj in zip(r, c):
        g = full.copy(); g[ei, cj] = -1e12
        rr, cc = linear_sum_assignment(-g)
        v = float(g[rr, cc].sum())
        if second is None or v > second:
            second = v
    return ({int(ei): (None if cj >= n_cand else int(cj))
             for ei, cj in zip(r, c)},
            best - second if second is not None else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ns", default="3,10")
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/stress1c")
    args = ap.parse_args()

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__))
                                       ).decode().strip()
    except Exception:
        head = "unknown"
    print(f"[1c] HEAD = {head}", flush=True)
    print(f"[1c] BUILD deltas {BUILD_DELTAS} -> FREEZE -> TEST deltas "
          f"{TEST_DELTAS}", flush=True)
    print(f"[1c] frozen: WRITE_MARGIN {WRITE_MARGIN} NOVELTY_MAX {NOVELTY_MAX} "
          f"MIN_VOTES {MIN_VOTES} CAPACITY {CAPACITY}", flush=True)

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    Ns = [int(x) for x in args.ns.split(",")]
    all_deltas = sorted(set(BUILD_DELTAS + TEST_DELTAS))

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[1c] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    from eval_two_layer import load_models, dino_feat
    load_models()

    def dino_np(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    traj0 = synth(scene, all_deltas[0])
    frames_n = len(traj0)
    n_lat = (frames_n - 1) // 4 + 1
    d = f"examples/s1c_{scene}"
    os.makedirs(d, exist_ok=True)
    np.save(f"{d}/poses.npy", traj0)
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
    print(f"[1c] {frames_n} frames -> {n_lat} chunks; N={Ns}; "
          f"delta={all_deltas}", flush=True)

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
    ymap = {}
    for nm, bb, role, spec in ALL_SLOTS:
        corr = make_correction(spec, ref_img, bb)
        ymap[nm] = build_y(TF.to_tensor(Image.fromarray(corr)).sub_(0.5)
                           .div_(0.5).unsqueeze(0).transpose(0, 1))
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()

    def make_gen(rel_all):
        def gen(y_cond, max_chunks=None, inject=None, collect_lat=False):
            self_kv = pipe._initialize_self_kv_cache(
                num_layers=ma.num_layers,
                shape=[1, kv_size, ma.num_heads // pipe.sp_size,
                       ma.dim // ma.num_heads],
                dtype=dtype, device=dev)
            cross_kv = pipe._initialize_crossattn_cache(
                num_layers=ma.num_layers,
                shape=[1, 512, ma.num_heads, ma.dim // ma.num_heads],
                dtype=dtype, device=dev)
            pipe._cross_attn_initialized = False
            g = torch.Generator(device=dev); g.manual_seed(sd)
            outs, lats = [], []
            N = n_lat if max_chunks is None else min(n_lat, max_chunks)
            tsl = pipe.scheduler.timesteps[[0, 250, 750]]
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
                for ti in range(len(tsl)):
                    with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                        npred = pipe.model(
                            x=[cur.to(dev)], t=torch.stack([tsl[ti]]).to(dev),
                            cross_attn_first_call=not pipe._cross_attn_initialized,
                            **kw)[0]
                        pipe._cross_attn_initialized = True
                        x0 = pipe._convert_flow_pred_to_x0(
                            flow_pred=npred, xt=cur, timestep=tsl[ti],
                            scheduler=pipe.scheduler)
                        if ti < len(tsl) - 1:
                            cur = pipe.scheduler.add_noise(
                                x0, torch.randn(x0.shape, generator=g,
                                                device=x0.device, dtype=x0.dtype),
                                tsl[ti + 1])
                x0 = x0.clone()
                if inject is not None:
                    inject(cid, x0)
                with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                    pipe.model(x=[x0], t=torch.stack([tsl[-1] * 0.0]).to(dev),
                               cross_attn_first_call=False, **kw)
                with torch.no_grad():
                    fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                          parallel=False, show_progress_bar=False)
                outs.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                             * 255.0).clip(0, 255).astype(np.uint8))
                if collect_lat:
                    lats.append(x0.detach().float().cpu())
            del self_kv, cross_kv
            gc.collect(); torch.cuda.empty_cache()
            return outs, lats
        return gen

    results = []
    for N in Ns:
        names = ORDER[:N]
        slots = {s[0]: s for s in ALL_SLOTS}
        LR = {}
        for nm in names:
            bb = slots[nm][1]
            LR[nm] = (int(bb[1] * lat_h), int(bb[3] * lat_h),
                      int(bb[0] * lat_w), int(bb[2] * lat_w))
        # ---- render every view once (shared by BOTH arms) ----
        view = {}
        for dlt in all_deltas:
            traj = synth(scene, dlt)
            c2w = interpolate_camera_poses(
                np.linspace(0, frames_n - 1, frames_n),
                torch.from_numpy(traj[:, :3, :3]).float(),
                torch.from_numpy(traj[:, :3, 3]).float(),
                np.linspace(0, frames_n - 1, n_lat)).to(dev)
            rel = compute_relative_poses(c2w, framewise=True)
            gen = make_gen(rel)
            A_anc, TPL = {}, {}
            for nm in names:
                y0, y1, x0_, x1_ = LR[nm]
                f, l = gen(ymap[nm], max_chunks=args.canon_chunks,
                           collect_lat=True)
                ci = min(2, len(l) - 1)
                A_anc[nm] = l[ci][:, :, y0:y1, x0_:x1_].clone().to(dev)
                # placeholder; the VIEW-SPECIFIC template is set below from the
                # return frame. Taking it from the canonicalisation frame (chunk
                # 2 = baseline pose) makes every dYaw's template identical --
                # the same pitfall as §43P-0's first version, which showed up as
                # max_sim = 1.000 for every BUILD session and blocked all writes.
                TPL[nm] = patch(f[ci], slots[nm][1])
            c2w_np = c2w.cpu().numpy()
            bb = slots[TARGET][1]
            uc = (bb[0] + bb[2]) / 2 * w; vc = (bb[1] + bb[3]) / 2 * h
            Xc = np.array([(uc - cx) / fx, (vc - cy) / fy, 1.0]) * 5.0
            Xw = c2w_np[2][:3, :3] @ Xc + c2w_np[2][:3, 3]

            def c_vis(cid):
                P = c2w_np[cid]
                X = P[:3, :3].T @ (Xw - P[:3, 3])
                if X[2] <= 1e-3:
                    return False
                u = (fx * X[0] / X[2] + cx) / w; v = (fy * X[1] / X[2] + cy) / h
                return (0.02 <= u <= 0.98) and (0.02 <= v <= 0.98)
            vis = [c_vis(c) for c in range(n_lat)]
            outc = [c for c in range(n_lat) if not vis[c]]
            retc = [c for c in range(n_lat) if vis[c] and outc and
                    c > min(outc)] if outc else []

            def mk_inject(A_anc):
                def inject(cid, x0):
                    a_dx = int(round(6 * min(1.0, max(0.0, (cid - BASE_N) / 20.0))))
                    for nm in names:
                        role = slots[nm][2]
                        if role == "occlude" and 8 <= cid <= 16:
                            continue
                        y0, y1, x0_, x1_ = LR[nm]
                        dx = a_dx if role == "moving" else 0
                        b0, b1 = x0_ + dx, x1_ + dx
                        if 0 <= y0 and y1 <= lat_h and 0 <= b0 and b1 <= lat_w:
                            x0[:, :, y0:y1, b0:b1] = A_anc[nm]
                return inject
            F, _ = gen(y, inject=mk_inject(A_anc))
            # templates captured AT THIS VIEW (the return frame), so different
            # dYaw really yields different templates
            for nm in names:
                TPL[nm] = patch(F[-1], slots[nm][1])
            view[dlt] = dict(A=A_anc, TPL=TPL, F=F, retc=retc, outc=outc)
            dif = float(np.abs(TPL[TARGET].astype(float)
                               - view[all_deltas[0]]["TPL"][TARGET].astype(float)
                               ).mean()) if all_deltas[0] in view else float("nan")
            print(f"[1c]   N={N} delta={dlt}: out {len(outc)} ret {len(retc)} "
                  f"tpl_diff_vs_first {dif:.2f}", flush=True)

        # ---- arms differ ONLY in bank capacity/writes ----
        for mode in ("single", "causal"):
            bank = [view[BUILD_DELTAS[0]]["TPL"]]
            per_obj = {nm: dict(size=1, writes=0, rej_margin=0, rej_novelty=0)
                       for nm in names}
            # ---------- BUILD (no measurement) ----------
            for si, dlt in enumerate(BUILD_DELTAS):
                if mode != "causal":
                    continue
                v = view[dlt]
                anchors = list(bank)
                for nm in names:
                    cf = [dino_np(patch(v["F"][c], slots[nm][1]))
                          for c in v["retc"]]
                    s_m = float(np.mean([max(float((x * dino_np(a[nm])).sum())
                                             for a in anchors) for x in cf]))
                    o_m = float(np.mean([max(float((x * dino_np(a[o])).sum())
                                             for a in anchors for o in names
                                             if o != nm) for x in cf]))
                    mx = max(float((dino_np(v["TPL"][nm]) * dino_np(b[nm])).sum())
                             for b in anchors)
                    if (s_m - o_m) < WRITE_MARGIN:
                        per_obj[nm]["rej_margin"] += 1
                    elif mx >= NOVELTY_MAX:
                        per_obj[nm]["rej_novelty"] += 1
                    elif len(cf) >= MIN_VOTES:
                        # per-identity slot: store this object's own view anchor
                        per_obj[nm]["writes"] += 1
                        per_obj[nm]["size"] = min(CAPACITY,
                                                  per_obj[nm]["size"] + 1)
                # the bank itself is the target-object bank used for matching
                cf_t = [dino_np(patch(v["F"][c], slots[TARGET][1]))
                        for c in v["retc"]]
                s_t = float(np.mean([max(float((x * dino_np(a[TARGET])).sum())
                                         for a in anchors) for x in cf_t]))
                o_t = float(np.mean([max(float((x * dino_np(a[o])).sum())
                                         for a in anchors for o in names
                                         if o != TARGET) for x in cf_t]))
                mx_t = max(float((dino_np(v["TPL"][TARGET]) *
                                  dino_np(b[TARGET])).sum()) for b in anchors)
                if (s_t - o_t) >= WRITE_MARGIN and mx_t < NOVELTY_MAX \
                        and len(cf_t) >= MIN_VOTES:
                    bank.append(v["TPL"])
                    if len(bank) > CAPACITY:
                        bank.pop(0)
                    print(f"[1c]   BUILD N={N} dYaw={dlt}: WRITE bank size "
                          f"{len(bank)}", flush=True)
                else:
                    print(f"[1c]   BUILD N={N} dYaw={dlt}: no write "
                          f"(margin {s_t-o_t:+.3f}, max_sim {mx_t:.3f})",
                          flush=True)
            bank_frozen = list(bank)
            print(f"[1c]   N={N} {mode}: bank FROZEN at size {len(bank_frozen)}",
                  flush=True)

            # ---------- TEST ----------
            for dlt in TEST_DELTAS:
                v = view[dlt]
                margins, amargs = [], []
                pos_hit = pos_tot = wrong = 0
                fb = fb_tot = 0
                reacq = None
                for cid in v["retc"]:
                    vis_ent = [nm for nm in names
                               if not (slots[nm][2] == "occlude" and 8 <= cid <= 16)]
                    cands = [(nm, slots[nm][1]) for nm in vis_ent]
                    nC = len(cands)
                    cvis = np.zeros((len(names), nC))
                    cgeo = np.zeros((len(names), nC))
                    for j, (cn, cb) in enumerate(cands):
                        fv = dino_np(patch(v["F"][cid], cb))
                        ctr = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
                        for i, en in enumerate(names):
                            cvis[i, j] = max(float((fv * dino_np(a[en])).sum())
                                             for a in bank_frozen)
                            cgeo[i, j] = dgeo(
                                ctr,
                                ((slots[en][1][0] + slots[en][1][2]) / 2,
                                 (slots[en][1][1] + slots[en][1][3]) / 2),
                                (slots[en][1][2] - slots[en][1][0],
                                 slots[en][1][3] - slots[en][1][1]))
                    bound, am = global_assign(cvis, cgeo, len(names), nC)
                    amargs.append(am)
                    ti = names.index(TARGET)
                    pos_tot += 1
                    if bound.get(ti) is not None and cands[bound[ti]][0] == TARGET:
                        pos_hit += 1
                        if reacq is None:
                            reacq = cid - min(v["retc"])
                    for i, en in enumerate(names):
                        j = bound.get(i)
                        if j is None:
                            continue
                        if cands[j][0] != en:
                            wrong += 1
                        margins.append(cvis[i, j] - max(
                            cvis[k, j] for k in range(len(names)) if k != i))
                    keep = [j for j in range(nC) if cands[j][0] != TARGET]
                    if keep:
                        b2, _ = global_assign(cvis[:, keep], cgeo[:, keep],
                                              len(names), len(keep))
                        fb_tot += 1
                        if b2.get(ti) is not None:
                            fb += 1
                mg = np.array(margins) if margins else np.array([0.0])
                am = np.array(amargs) if amargs else np.array([0.0])
                r = dict(n=N, delta=dlt, mode=mode,
                         id_mean=float(mg.mean()),
                         id_p10=float(np.percentile(mg, 10)),
                         id_min=float(mg.min()),
                         neg_frac=float(np.mean(mg < 0)),
                         as_mean=float(np.nanmean(am)),
                         as_min=float(np.nanmin(am)),
                         wrong=int(wrong),
                         recall=pos_hit / max(pos_tot, 1),
                         false_bind=fb / max(fb_tot, 1),
                         reacq=reacq,
                         bank_size=len(bank_frozen),
                         per_obj=per_obj if mode == "causal" else None)
                results.append(r)
                print(f"[1c] N={N:2d} dYaw={dlt:3.0f} {mode:6s}: "
                      f"id_mean {r['id_mean']:+.3f} p10 {r['id_p10']:+.3f} "
                      f"min {r['id_min']:+.3f} neg {r['neg_frac']*100:3.0f}% | "
                      f"as_min {r['as_min']:+.3f} wrong {wrong} "
                      f"recall {r['recall']*100:3.0f}% fb "
                      f"{r['false_bind']*100:3.0f}% | bank {len(bank_frozen)}",
                      flush=True)
        for dlt in all_deltas:
            del view[dlt]["F"]
        gc.collect(); torch.cuda.empty_cache()

    print("\n[1c] ===== N x dYaw, single vs FROZEN-bank causal =====")
    print(f"  {'N':>3s} {'dYaw':>5s} {'mode':>7s} {'id_mean':>8s} "
          f"{'id_p10':>8s} {'id_min':>8s} {'neg%':>5s} {'as_min':>8s} "
          f"{'wrong':>5s} {'recall':>7s} {'fb%':>4s} {'bank':>4s}")
    for r in results:
        print(f"  {r['n']:3d} {r['delta']:5.0f} {r['mode']:>7s} "
              f"{r['id_mean']:+8.3f} {r['id_p10']:+8.3f} {r['id_min']:+8.3f} "
              f"{r['neg_frac']*100:4.0f}% {r['as_min']:+8.3f} {r['wrong']:5d} "
              f"{r['recall']*100:6.0f}% {r['false_bind']*100:3.0f}% "
              f"{r['bank_size']:4d}")

    # ---- decisive cell ----
    print("\n[1c] ===== decisive cell N=10 x dYaw=7 =====")
    by = {(r["n"], r["delta"], r["mode"]): r for r in results}
    for N in Ns:
        s = by.get((N, 7.0, "single"))
        c = by.get((N, 7.0, "causal"))
        if s and c:
            print(f"  N={N}: id_mean {s['id_mean']:+.3f} -> {c['id_mean']:+.3f} "
                  f"({c['id_mean']-s['id_mean']:+.3f}) | p10 "
                  f"{s['id_p10']:+.3f} -> {c['id_p10']:+.3f} | min "
                  f"{s['id_min']:+.3f} -> {c['id_min']:+.3f} | neg "
                  f"{s['neg_frac']*100:.0f}% -> {c['neg_frac']*100:.0f}%")
            print(f"        as_min {s['as_min']:+.3f} -> {c['as_min']:+.3f} | "
                  f"wrong {s['wrong']} -> {c['wrong']} | recall "
                  f"{s['recall']*100:.0f}% -> {c['recall']*100:.0f}% | "
                  f"fb {s['false_bind']*100:.0f}% -> {c['false_bind']*100:.0f}%")

    # ---- per-identity bank audit ----
    print("\n[1c] ===== per-identity bank audit (causal arm) =====")
    for r in results:
        if r["mode"] != "causal" or r["per_obj"] is None:
            continue
        print(f"  N={r['n']} dYaw={r['delta']:.0f}:")
        for nm, v in r["per_obj"].items():
            print(f"    {nm:9s} size {v['size']} writes {v['writes']} "
                  f"rej_margin {v['rej_margin']} rej_novelty {v['rej_novelty']}")

    json.dump(dict(results=results, head=head,
                   build=BUILD_DELTAS, test=TEST_DELTAS,
                   frozen=dict(WRITE_MARGIN=WRITE_MARGIN,
                               NOVELTY_MAX=NOVELTY_MAX, MIN_VOTES=MIN_VOTES,
                               CAPACITY=CAPACITY)),
              open(f"{args.out_dir}/stress1c.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
