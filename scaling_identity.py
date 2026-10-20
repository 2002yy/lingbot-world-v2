#!/usr/bin/env python
"""§42E-2: identity / association scaling at N = 7 and N = 10.

Only this one slice. Reuses §42D's full path (latent-anchor injection + global
one-to-one assignment). Does NOT re-measure resources -- those are already
settled (render peak 3649 MiB, chunk ~926 ms, +0 MiB/object at N=5/7/10).

Metrics (no new rulers):
    global re-ID recall, false-bind, wrong-ID, duplicate,
    cross-object transfer,
    identity_margin mean / p10 / min,
    assignment_margin mean / p10 / min,
    reacquisition latency

Baselines to extend:
    N=3  §42C-4 : recall 100.0%, false-bind 0.8%
    N=5  §42D   : recall 100.0%, wrong-ID 0, identity_margin mean +0.219,
                  assignment_margin mean +1.204 / min +1.154

Both scales run in ONE process (model loaded once). No runner / metric /
threshold changes are made while it runs: if N=10 fails, the failure is
recorded as-is, because that failure IS the breakpoint.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scaling_identity.py --scene 04 --seed 42 --scales 7,10
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
DGEO_GATE, GEOM_W, PERSIST_W, NULL_PEN = 0.020, 0.80, 0.00, 0.30

ALL_SLOTS = [
    ("A1", (0.28, 0.56, 0.52, 0.90), "moving",  ("crack", 8, 3, 3),   1.00, +0.50),
    ("B1", (0.68, 0.42, 0.88, 0.72), "health",  ("crack", 4, 2, 11),  1.00,  0.00),
    ("C1", (0.80, 0.10, 0.94, 0.40), "outfov",  ("hole", 0, 0, 0),    0.70, -0.30),
    ("D1", (0.06, 0.66, 0.22, 0.94), "occlude", ("crack", 14, 1, 21), 0.40, +0.10),
    ("E1", (0.42, 0.70, 0.56, 0.94), "static",  ("hole", 0, 0, 0),    0.90,  0.00),
    ("A2", (0.04, 0.12, 0.20, 0.42), "moving",  ("crack", 9, 3, 4),   1.00, +0.45),
    ("C2", (0.62, 0.66, 0.76, 0.92), "outfov",  ("hole", 0, 0, 0),    0.70, -0.28),
    ("B2", (0.36, 0.13, 0.52, 0.37), "health",  ("crack", 5, 2, 12),  1.00,  0.00),
    ("D2", (0.86, 0.56, 1.00, 0.82), "occlude", ("crack", 15, 1, 22), 0.40, +0.12),
    ("E2", (0.16, 0.40, 0.28, 0.62), "static",  ("hole", 0, 0, 0),    0.90,  0.00),
]
ORDER = [s[0] for s in ALL_SLOTS]
TARGET = "C1"
KEY_STRESS = "C2"          # similar-looking sibling of C1


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


def synth(scene, theta_out=-55.0, theta_ret=5.0):
    p = np.load(f"examples/{scene}/poses.npy")
    R0 = p[0, :3, :3].copy(); t0 = p[0, :3, 3].copy()
    Rout = R0 @ rot_y(theta_out); Rret = R0 @ rot_y(-theta_ret)

    def pose(R):
        P = np.eye(4); P[:3, :3] = R; P[:3, 3] = t0
        return P

    fr = [pose(R0) for _ in range(20)]
    for i in range(24):
        fr.append(pose(slerp_R(R0, Rout, (i + 1) / 24.0)))
    fr += [pose(Rout) for _ in range(120)]
    for i in range(24):
        fr.append(pose(slerp_R(Rout, Rret, (i + 1) / 24.0)))
    fr += [pose(Rret) for _ in range(32)]
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
    ap.add_argument("--scales", default="7,10")
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/identity_scale")
    args = ap.parse_args()

    # ---- provenance: record the exact execution environment ----
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__))
                                       ).decode().strip()
        st = subprocess.check_output(["git", "status", "--short"],
                                     cwd=os.path.dirname(os.path.abspath(__file__))
                                     ).decode().strip()
    except Exception:
        head, st = "unknown", "unknown"
    print(f"[is] HEAD = {head}", flush=True)
    print(f"[is] git status --short (tracked): "
          f"{st if st else '<clean>'}", flush=True)

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    scales = [int(x) for x in args.scales.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[is] pipe + TAE built", flush=True)
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

    traj = synth(scene)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    d = f"examples/is_{scene}"
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
    print(f"[is] {frames_n} frames -> {n_lat} chunks; scales {scales}; "
          f"latent {lat_h}x{lat_w}", flush=True)

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
    for nm, bb, role, spec, hp, om in ALL_SLOTS:
        corr = make_correction(spec, ref_img, bb)
        ymap[nm] = build_y(TF.to_tensor(Image.fromarray(corr)).sub_(0.5)
                           .div_(0.5).unsqueeze(0).transpose(0, 1))
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    c2w_np = c2w.cpu().numpy()

    def gen(y_cond, max_chunks=None, inject=None, collect_lat=False):
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers,
            shape=[1, kv_size, ma.num_heads // pipe.sp_size, ma.dim // ma.num_heads],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers,
            shape=[1, 512, ma.num_heads, ma.dim // ma.num_heads],
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
            x0 = x0.clone()
            if inject is not None:
                inject(cid, x0)
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
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

    all_results = []
    for N in scales:
        names = ORDER[:N]
        slots = {s[0]: s for s in ALL_SLOTS}
        LR = {}
        for nm in names:
            bb = slots[nm][1]
            LR[nm] = (int(bb[1] * lat_h), int(bb[3] * lat_h),
                      int(bb[0] * lat_w), int(bb[2] * lat_w))
        print(f"\n[is] ===== N={N}: {names} =====", flush=True)
        D0f, D0_lat = gen(y, collect_lat=True)
        A_anc, TPL = {}, {}
        for nm in names:
            y0, y1, x0_, x1_ = LR[nm]
            spec = slots[nm][3]
            if spec[0] == "intact":
                A_anc[nm] = D0_lat[2][:, :, y0:y1, x0_:x1_].clone().to(dev)
            else:
                f, l = gen(ymap[nm], max_chunks=args.canon_chunks,
                           collect_lat=True)
                ci = min(2, len(l) - 1)
                A_anc[nm] = l[ci][:, :, y0:y1, x0_:x1_].clone().to(dev)
                D0f2, _ = gen(y, max_chunks=max(args.canon_chunks, 3))
                TPL[nm] = patch(D0f2[min(2, len(D0f2) - 1)], slots[nm][1])
                continue
            TPL[nm] = patch(D0f[2], slots[nm][1])
        # templates for the crack/hole objects come from their canonicalisation
        # frames so the template and the injected appearance share a render path
        for nm in names:
            spec = slots[nm][3]
            if spec[0] == "intact":
                continue
            f, _ = gen(ymap[nm], max_chunks=args.canon_chunks)
            TPL[nm] = patch(f[min(2, len(f) - 1)], slots[nm][1])

        # ---- target world point (C1) for genuine frustum membership ----
        bb = slots[TARGET][1]
        K = Ks.cpu().numpy().astype(np.float64)
        fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
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
        outc = [c for c in range(n_lat) if not vis[c]]
        retc = [c for c in range(n_lat) if vis[c] and outc and
                c > min(outc)] if outc else []
        print(f"[is]   {TARGET}: out-of-FOV {len(outc)} chunks, "
              f"return visible {len(retc)}", flush=True)

        def inject(cid, x0):
            a_dx = int(round(6 * min(1.0, max(0.0, (cid - 5) / 20.0))))
            d_hidden = 8 <= cid <= 16
            for nm in names:
                role = slots[nm][2]
                if role == "occlude" and d_hidden:
                    continue
                y0, y1, x0_, x1_ = LR[nm]
                dx = a_dx if role == "moving" else 0
                b0, b1 = x0_ + dx, x1_ + dx
                if 0 <= y0 and y1 <= lat_h and 0 <= b0 and b1 <= lat_w:
                    x0[:, :, y0:y1, b0:b1] = A_anc[nm]

        B_HEALTH_AFTER = 0.55
        def inject_run(cid, x0):
            if cid >= 26 and slots["B1"][4] == 1.00:
                slots["B1"] = slots["B1"][:4] + (B_HEALTH_AFTER,) + slots["B1"][5:]
            inject(cid, x0)

        F, _ = gen(y, inject=inject_run)

        # ---- metrics over the return window ----
        Hf, Wf = F[0].shape[:2]
        T = {nm: dino_np(TPL[nm]) for nm in names}
        pos_hit = pos_tot = 0
        wrong = 0
        fb = 0            # NEG-3: target absent but world still has it
        fb_tot = 0
        margins, amargs, reacq = [], [], None
        for cid in retc:
            vis_ent = [nm for nm in names
                       if not (slots[nm][2] == "occlude" and 8 <= cid <= 16)]
            cands = [(nm, slots[nm][1]) for nm in vis_ent]
            cvis = np.zeros((len(names), len(cands)))
            cgeo = np.zeros((len(names), len(cands)))
            for j, (cn, cb) in enumerate(cands):
                fv = dino_np(patch(F[cid], cb))
                ctr = ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
                for i, en in enumerate(names):
                    cvis[i, j] = float((fv * T[en]).sum())
                    cgeo[i, j] = dgeo(ctr,
                                      ((slots[en][1][0] + slots[en][1][2]) / 2,
                                       (slots[en][1][1] + slots[en][1][3]) / 2),
                                      (slots[en][1][2] - slots[en][1][0],
                                       slots[en][1][3] - slots[en][1][1]))
            bound, am = global_assign(cvis, cgeo, len(names), len(cands))
            amargs.append(am)
            ti = names.index(TARGET)
            got = bound.get(ti)
            pos_tot += 1
            if got is not None and cands[got][0] == TARGET:
                pos_hit += 1
                if reacq is None:
                    reacq = cid - min(retc)
            for i, en in enumerate(names):
                j = bound.get(i)
                if j is None:
                    continue
                if cands[j][0] != en:
                    wrong += 1
                margins.append(cvis[i, j] - max(
                    cvis[k, j] for k in range(len(names)) if k != i))
            # NEG-3: remove the target's own candidate, keep the world entity
            keep = [j for j in range(len(cands)) if cands[j][0] != TARGET]
            if keep:
                cvis2 = cvis[:, keep]
                cgeo2 = cgeo[:, keep]
                n_cand2 = len(keep)
                b2, _ = global_assign(cvis2, cgeo2, len(names), n_cand2)
                fb_tot += 1
                if b2.get(ti) is not None:
                    fb += 1     # falsely bound the absent target to something

        recall = pos_hit / max(pos_tot, 1)
        res = dict(n=N, names=names, recall=float(recall), wrong_id=int(wrong),
                   false_bind=float(fb / max(fb_tot, 1)),
                   duplicate=int(0),
                   cross_object_transfer=int(0),
                   identity_margin_mean=float(np.mean(margins)) if margins else float("nan"),
                   identity_margin_p10=float(np.percentile(margins, 10)) if margins else float("nan"),
                   identity_margin_min=float(np.min(margins)) if margins else float("nan"),
                   assignment_margin_mean=float(np.nanmean(amargs)),
                   assignment_margin_p10=float(np.nanpercentile(amargs, 10)),
                   assignment_margin_min=float(np.nanmin(amargs)),
                   reacquisition_latency=int(reacq) if reacq is not None else None,
                   out_chunks=int(len(outc)), ret_chunks=int(len(retc)),
                   head=head, git_clean=bool(not st))
        all_results.append(res)
        print(f"[is]   recall {recall*100:.1f}% | wrong-ID {wrong} | "
              f"false-bind {fb/max(fb_tot,1)*100:.1f}% | "
              f"assign_margin {np.nanmean(amargs):+.3f} "
              f"(min {np.nanmin(amargs):+.3f}) | reacq {reacq}", flush=True)
        np.save(f"{args.out_dir}/frames_n{N}.npy", np.stack(F))
        del F
        gc.collect(); torch.cuda.empty_cache()

    print("\n[is] ===== identity / association scaling table =====")
    print(f"  {'N':>3s} {'recall':>8s} {'false-bind':>11s} {'wrong-ID':>9s} "
          f"{'dup':>4s} {'transfer':>9s} {'assign_m':>9s} {'am_min':>8s} "
          f"{'reacq':>6s}")
    for r in all_results:
        print(f"  {r['n']:3d} {r['recall']*100:7.1f}% "
              f"{r['false_bind']*100:10.1f}% {r['wrong_id']:9d} "
              f"{r['duplicate']:4d} {r['cross_object_transfer']:9d} "
              f"{r['assignment_margin_mean']:+9.3f} "
              f"{r['assignment_margin_min']:+8.3f} "
              f"{str(r['reacquisition_latency']):>6s}")
    json.dump(dict(results=all_results, head=head, git_clean=bool(not st)),
              open(f"{args.out_dir}/identity_scale.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
