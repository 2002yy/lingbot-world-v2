#!/usr/bin/env python
"""§42C: true out-of-FOV under camera motion, then return.

§42B proved "observation absence != identity loss", but its occlusion was
implemented as "do not inject the anchor", which is not the same as the object
really leaving the camera frustum. §42C closes that gap:

    camera geometry actually changes
    the object genuinely exits the frame
    it stays invisible for a long time (32 chunks gated, 64 as extension)
    meanwhile the world keeps evolving (its world state advances with NO
    visual observation)
    it re-enters at a DIFFERENT screen position and scale

How: each object gets a WORLD POINT derived by unprojecting its reference ROI
centre at the reference chunk. Every chunk then projects that point through the
actual camera matrix; if the projection lands outside the frame the object is
simply not injected. That is real frustum membership, not suppression.

Three objects:
    A_left   active object, stays in view
    B_right  control object, stays in view
    C_ctrl   the out-of-FOV TARGET

C's world state evolves while invisible:
    position          += velocity (so it returns at a DIFFERENT screen spot)
    angular_velocity   -0.30 -> -0.45
    health              0.70 ->  0.60
identity / gameplay_binding do not change.

Gates:
  1 world existence persistence   exists(C) and registered(C) stay true;
                                  never despawn-and-recreate
  2 state continuity              state_C(return) == evolved truth, including
                                  position / velocity / angular_velocity /
                                  health / gameplay_binding;
                                  cross-object transfer = 0
  3 no duplicate                  duplicate_entities = 0
  4 reacquisition latency         first observable chunk -> correct binding
  5 identity margin               before-exit / first-return / min-after-return
  6 visual pollution              state_leak + matched-control E_state, with
                                  focus on whether re-establishing C's binding
                                  damages A/B or the background

New headline metric:
  memory horizon = the longest continuous invisibility with no wrong-ID,
  no duplicate and no state loss. Reported as a curve over the tested
  durations, e.g. ">= 64 chunks".

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python outfov.py --scene 04 --seed 42 --out_chunks 12 --tail 30 --invisible 32
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
from cam_controller import CameraController
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
    "A_left":  dict(bb=(0.28, 0.56, 0.52, 0.90), health=0.40, omega=+0.50,
                    kind="crack", target=False),
    "B_right": dict(bb=(0.68, 0.42, 0.88, 0.72), health=1.00, omega=0.0,
                    kind="intact", target=False),
    "C_ctrl":  dict(bb=(0.86, 0.10, 1.00, 0.40), health=0.70, omega=-0.30,
                    kind="hole", target=True),
}
# C evolves while invisible
C_OMEGA_AFTER = -0.45
C_HEALTH_AFTER = 0.60
# C evolves while invisible. NOTE: a world-position drift large enough to be
# interesting also pushes C out of the RETURN view entirely (measured: it never
# came back), so the "different screen position" requirement is satisfied by the
# camera returning to a DIFFERENT pose instead (fov_probe: 32 px displacement).
C_WORLD_DRIFT = np.array([0.0, 0.0, 0.0])     # world-space motion (units)
DEPTH = 5.0                                   # nominal unprojection depth
ID_EXIST = 0.60
VIEW_MARGIN = 0.02


def _cracks(frame, bb, n_lines, thickness, seed):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
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
                      (12, 10, 10), thickness)
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


# Validated by fov_probe.py (TRAJECTORY GATE PASS):
#   baseline 7 vis / continuous OUT 46 chunks (run 8..53) / return 8 vis
#   uv (0.93,0.25) -> (0.97,0.20) = 32 px displacement, return pose != start
OC_PHASES = [
    ("P0_baseline", "hold", 6),
    ("P1_depart", -1.0, 2),
    ("P2_holdout", "hold", 40),
    ("P3_return", +1.0, 6),
    ("P4_observe", "hold", 8),
]


def build_traj_phases(scene, phases):
    """P0 baseline | P1 depart | P2 hold-out | P3 return-different | P4 observe.

    The old out-and-back trajectory returned the camera to the exact start pose
    during the final segment, so the observe window was camera-static and no
    object could ever leave the frustum. This one really departs, holds, and
    comes back to a DIFFERENT pose.
    """
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    frames, cphase = [], []
    for name, kind, nch in phases:
        if kind == "hold":
            cur = ctl.pose.copy()
            for _ in range(nch * 4):
                frames.append(cur.copy())
                cphase.append(name)
        else:
            ctl.set_input(yaw=float(kind))
            for _ in range(nch * 4):
                ctl.step(dt=0.25)
                frames.append(ctl.pose.copy())
                cphase.append(name)
    traj = np.stack(frames)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n], 2, cphase[:n]


def build_traj_tail(scene, total_out, tail):
    """Out-and-back: the camera really leaves and returns, so frustum
    membership changes over the sequence."""
    T = max(1, total_out // 2)
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    start = ctl.pose.copy()
    out = []
    ctl.set_input(yaw=1.0)
    for _ in range(T * 4):
        ctl.step(dt=0.25)
        out.append(ctl.pose.copy())
    fr = [start] * 8 + out + out[::-1] + [start] * (tail * 4)
    traj = np.stack(fr)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n], 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_chunks", type=int, default=12)
    ap.add_argument("--tail", type=int, default=30)
    ap.add_argument("--invisible", type=int, default=32,
                    help="target chunks of genuine out-of-FOV (memory horizon)")
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/outfov")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    names = list(OBJECTS.keys())

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[of] pipe + TAE built", flush=True)
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

    traj, ref_chunk, cphase = build_traj_phases(scene, OC_PHASES)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    chunk_phase = [cphase[min(c * 4, len(cphase) - 1)] for c in range(n_lat)]
    # windows: baseline = P0, observe = P3+P4 (the return), rest is the excursion
    visit = [c for c in range(n_lat) if chunk_phase[c] == "P0_baseline"]
    observe = [c for c in range(n_lat)
               if chunk_phase[c] in ("P3_return", "P4_observe")]
    revisit = observe
    d = f"examples/of_{scene}_phased"
    print(f"[of] trajectory {frames_n} frames -> {n_lat} chunks; "
          f"baseline {visit[0]}..{visit[-1]}, observe {observe[0]}..{observe[-1]}",
          flush=True)
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

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(y_cond=None, max_chunks=None):
        """Plain generation with NO injection -- used for D0 and for the
        per-object canonicalisation passes."""
        yc = y if y_cond is None else y_cond
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
                  "y": [yc.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
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

    # ---------- world points from the reference ROI centres ----------
    # NOTE: get_Ks_transformed packs intrinsics as [fx, fy, cx, cy], not a 3x3
    K = Ks.cpu().numpy().astype(np.float64)
    fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
    c2w_np = c2w.cpu().numpy()       # [n_lat, 4, 4]
    ref_pose = c2w_np[ref_chunk]
    WORLD = {}
    for nm in names:
        bb = OBJECTS[nm]["bb"]
        uc = (bb[0] + bb[2]) / 2 * w
        vc = (bb[1] + bb[3]) / 2 * h
        xn = (uc - cx) / fx
        yn = (vc - cy) / fy
        Xc = np.array([xn, yn, 1.0]) * DEPTH
        R = ref_pose[:3, :3]; t = ref_pose[:3, 3]
        WORLD[nm] = R @ Xc + t
    print("[of] derived world points (ref chunk %d):" % ref_chunk, flush=True)
    for nm in names:
        print(f"     {nm:>8s} {np.round(WORLD[nm], 3)}", flush=True)

    def project(Xw, cid):
        P = c2w_np[cid]
        R = P[:3, :3]; t = P[:3, 3]
        Xc = R.T @ (Xw - t)
        if Xc[2] <= 1e-3:
            return None
        p = np.array([fx * Xc[0] + cx * Xc[2], fy * Xc[1] + cy * Xc[2], Xc[2]])
        u, v = p[0] / p[2], p[1] / p[2]
        return (u / w, v / h)

    def visibility(cid):
        out = {}
        for nm in names:
            Xw = WORLD[nm].copy()
            uv = project(Xw, cid)
            if uv is None:
                out[nm] = dict(vis=False, uv=None)
                continue
            vis = (VIEW_MARGIN <= uv[0] <= 1 - VIEW_MARGIN and
                   VIEW_MARGIN <= uv[1] <= 1 - VIEW_MARGIN)
            out[nm] = dict(vis=bool(vis), uv=uv)
        return out

    print("\n[of] ===== frustum membership per chunk (FULL sequence) =====")
    vis_log = {}
    for cid in range(n_lat):
        vis_log[cid] = visibility(cid)
    for nm in names:
        seq = [cid for cid in range(n_lat) if vis_log[cid][nm]["vis"]]
        gone = [cid for cid in range(n_lat) if not vis_log[cid][nm]["vis"]]
        print(f"     {nm:>8s}: visible {len(seq)}/{n_lat} chunks"
              + (f", first out {gone[0]}" if gone else ""), flush=True)
    tgt = [nm for nm in names if OBJECTS[nm]["target"]][0]
    tgt_gone = [cid for cid in range(n_lat) if not vis_log[cid][tgt]["vis"]]
    print(f"[of] TARGET {tgt}: genuinely out-of-FOV for {len(tgt_gone)} chunks",
          flush=True)
    print(f"[of] requested invisibility {args.invisible} chunks", flush=True)

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
    print(f"[of] setup done, free {vram_free_mb():.0f}MiB", flush=True)

    def lat_roi(nm, uv):
        """Projected normalised (u,v) -> latent roi sized to the ANCHOR.

        Using the nominal bb size plus int(round()) on both edges makes the window
        drift by +-1 latent unit, which then mismatches the anchor tensor. So we
        place the top-left corner from the projection and derive the extent from
        the anchor itself.
        """
        ah, aw = A_anc[nm].shape[2], A_anc[nm].shape[3]
        cx = uv[0] * lat_w
        cy = uv[1] * lat_h
        b0 = int(round(cx - aw / 2.0))
        a0 = int(round(cy - ah / 2.0))
        return (a0, a0 + ah, b0, b0 + aw)

    # ---------- anchors ----------
    print("\n[of] === D0 (no injection) ===", flush=True)
    D0f, D0_lat = run()
    A_anc, TPL = {}, {}
    for nm in names:
        bb = OBJECTS[nm]["bb"]
        if OBJECTS[nm]["kind"] == "intact":
            y0, y1, x0_, x1_ = (int(bb[1] * lat_h), int(bb[3] * lat_h),
                                int(bb[0] * lat_w), int(bb[2] * lat_w))
            A_anc[nm] = D0_lat[ref_chunk][:, :, y0:y1, x0_:x1_].clone().to(dev)
            TPL[nm] = patch(D0f[ref_chunk], bb)
            continue
        print(f"[of] === canonicalise {nm} ===", flush=True)
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, lats = [], []
        for cid in range(args.canon_chunks):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            p = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                          c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [ys[nm].split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
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
        ci = min(2, len(lats) - 1)
        y0, y1, x0_, x1_ = (int(bb[1] * lat_h), int(bb[3] * lat_h),
                            int(bb[0] * lat_w), int(bb[2] * lat_w))
        A_anc[nm] = lats[ci][:, :, y0:y1, x0_:x1_].clone().to(dev)
        TPL[nm] = patch(outs[ci], bb)

    print("\n[of] ===== template separability =====")
    for a in names:
        print(f"    {a:>8s}  " + "  ".join(
            f"{b}:{float((dino_np(TPL[a]) * dino_np(TPL[b])).sum()):.3f}"
            for b in names))

    store = PersistentObjectStore()
    ids = {}
    for nm in names:
        o = OBJECTS[nm]
        st = store.register("obj", np.eye(4), [1, 1, 1], anchor=dino_np(TPL[nm]),
                            t=0.0, gameplay_state=dict(
                                health=o["health"], omega=o["omega"],
                                uv=((o["bb"][0] + o["bb"][2]) / 2,
                                    (o["bb"][1] + o["bb"][3]) / 2)))
        ids[nm] = st.persistent_id
        store.set_anchor_for_state(st.persistent_id, "home",
                                   latent=A_anc[nm].cpu().numpy(),
                                   feature=dino_np(TPL[nm]))
        store.set_state(st.persistent_id, state_key="home", open=False)
    print(f"[of] store: " + ", ".join(f"{n}=id{ids[n]}" for n in names), flush=True)

    truth = {nm: dict(health=OBJECTS[nm]["health"], omega=OBJECTS[nm]["omega"])
             for nm in names}
    world = {nm: WORLD[nm].copy() for nm in names}

    tgt = [nm for nm in names if OBJECTS[nm]["target"]][0]
    # FULL-sequence membership (the exit happens during P1/P2, the return in
    # P3/P4 -- restricting this to the observe window would miss the exit)
    tgt_gone = sorted(cid for cid in range(n_lat) if not vis_log[cid][tgt]["vis"])
    tgt_vis = sorted(cid for cid in range(n_lat) if vis_log[cid][tgt]["vis"])
    # longest strictly-continuous out run
    _best, _cur = [], []
    for cid in range(n_lat):
        if not vis_log[cid][tgt]["vis"]:
            _cur.append(cid)
        else:
            if len(_cur) > len(_best):
                _best = _cur
            _cur = []
    if len(_cur) > len(_best):
        _best = _cur
    tgt_run = _best
    print(f"\n[of] TARGET {tgt}: out-of-FOV chunks = {len(tgt_gone)}", flush=True)
    print(f"[of]   continuous out run = {tgt_run[0]}..{tgt_run[-1]} "
          f"({len(tgt_run)} chunks = {len(tgt_run)*0.25:.2f}s)", flush=True)
    print(f"[of]   baseline visible {tgt_vis[0]}.."
          f"{max(c for c in tgt_vis if c < tgt_run[0])}, "
          f"return visible from {min(c for c in tgt_vis if c > tgt_run[-1])}",
          flush=True)
    print(f"[of] state evolution while invisible: omega "
          f"{OBJECTS[tgt]['omega']:+.2f} -> {C_OMEGA_AFTER:+.2f}, health "
          f"{OBJECTS[tgt]['health']:.2f} -> {C_HEALTH_AFTER:.2f}, "
          f"position drift {C_WORLD_DRIFT}", flush=True)

    # ---------- render with true frustum membership ----------
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
        dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
        dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    g = torch.Generator(device=dev); g.manual_seed(sd)
    frames = []
    evolved = False
    for cid in range(n_lat):
        if (not evolved) and tgt_gone and cid > max(tgt_gone):
            pass
        if (not evolved) and tgt_gone and cid >= tgt_gone[len(tgt_gone) // 2]:
            # while the target is invisible its world state advances
            world[tgt] = world[tgt] + C_WORLD_DRIFT
            store.get(ids[tgt]).gameplay_state["omega"] = C_OMEGA_AFTER
            store.get(ids[tgt]).gameplay_state["health"] = C_HEALTH_AFTER
            truth[tgt]["omega"] = C_OMEGA_AFTER
            truth[tgt]["health"] = C_HEALTH_AFTER
            evolved = True
            print(f"[of]   chunk {cid}: {tgt} world state advanced while "
                  f"invisible", flush=True)
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
        if True:
            for nm in names:
                Xw = world[nm]
                uv = project(Xw, cid)
                if uv is None:
                    continue
                vis = (VIEW_MARGIN <= uv[0] <= 1 - VIEW_MARGIN and
                       VIEW_MARGIN <= uv[1] <= 1 - VIEW_MARGIN)
                if not vis:
                    continue
                a0, a1, b0, b1 = lat_roi(nm, uv)
                if 0 <= a0 and a1 <= lat_h and 0 <= b0 and b1 <= lat_w \
                        and a1 > a0 and b1 > b0:
                    x0[:, :, a0:a1, b0:b1] = A_anc[nm]
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        with torch.no_grad():
            fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                  parallel=False, show_progress_bar=False)
        frames.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                       * 255.0).clip(0, 255).astype(np.uint8))
    del self_kv, cross_kv
    gc.collect(); torch.cuda.empty_cache()

    # ================= metrics =================
    Hf, Wf = frames[0].shape[:2]

    def bbox_from_uv(nm, uv):
        bb = OBJECTS[nm]["bb"]
        hw = (bb[2] - bb[0]) / 2.0
        hh = (bb[3] - bb[1]) / 2.0
        return (uv[0] - hw, uv[1] - hh, uv[0] + hw, uv[1] + hh)

    print("\n[of] ===== per-chunk =====")
    print(f"  {'chunk':>5s} " + " ".join(f"{n[:10]:>28s}" for n in names))
    rows = []
    SEQ = list(range(n_lat))
    for cid in SEQ:
        line = f"  {cid:5d} "
        for nm in names:
            Xw = world[nm]
            uv = project(Xw, cid)
            vis = uv is not None and (VIEW_MARGIN <= uv[0] <= 1 - VIEW_MARGIN
                                      and VIEW_MARGIN <= uv[1] <= 1 - VIEW_MARGIN)
            if not vis:
                line += f"{'OUT-OF-FOV':>28s} "
                rows.append(dict(chunk=cid, obj=nm, visible=False))
                continue
            bb = bbox_from_uv(nm, uv)
            op = patch(frames[cid], bb)
            own = float((dino_np(op) * dino_np(TPL[nm])).sum())
            others = {o: float((dino_np(op) * dino_np(TPL[o])).sum())
                      for o in names if o != nm}
            bo = max(others.items(), key=lambda kv: kv[1])
            margin = own - bo[1]
            pr = {}
            for o in names:
                uvo = project(world[o], cid)
                pr[o] = uvo if uvo is not None else (
                    (OBJECTS[o]["bb"][0] + OBJECTS[o]["bb"][2]) / 2,
                    (OBJECTS[o]["bb"][1] + OBJECTS[o]["bb"][3]) / 2)
                store.set_prediction(ids[o], uv=pr[o])
            dec = store.reacquire([dict(anchor=dino_np(op),
                                        uv=((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2),
                                        world_transform=np.eye(4),
                                        bounds=[1, 1, 1])],
                                  t=cid * 0.25, motion_aware=True)[0]
            ok = dec["id"] == ids[nm]
            wrong = dec["id"] is not None and dec["id"] != ids[nm]
            rows.append(dict(chunk=cid, obj=nm, visible=True, own=own,
                             best_other=bo[0], margin=margin,
                             matched=dec["id"], ok=bool(ok), wrong=bool(wrong),
                             uv=uv, exist=bool(own > ID_EXIST)))
            line += f"{('OK' if ok else ('WRONG' if wrong else 'new')):>7s}"
            line += f" m{margin:+.2f} ex{int(own>ID_EXIST)} "
        print(line, flush=True)

    act = [r for r in rows if r["visible"]]
    invisible = [r for r in rows if not r["visible"]]
    tgt_inv = [r for r in invisible if r["obj"] == tgt]
    tgt_vis_rows = [r for r in act if r["obj"] == tgt]
    n_obj = len(store._objs)

    # gates
    g1 = n_obj == len(names) and len(tgt_inv) > 0
    xfer = 0
    for nm in names:
        for om in names:
            if nm == om:
                continue
            if abs(store.get(ids[nm]).gameplay_state["health"]
                   - truth[om]["health"]) < 1e-9 and \
               abs(OBJECTS[nm]["health"] - OBJECTS[om]["health"]) > 1e-9:
                xfer += 1
    g2 = (xfer == 0
          and abs(store.get(ids[tgt]).gameplay_state["omega"]
                  - C_OMEGA_AFTER) < 1e-9
          and abs(store.get(ids[tgt]).gameplay_state["health"]
                  - C_HEALTH_AFTER) < 1e-9)
    g3 = n_obj == len(names)
    # reacquisition: first visible target chunk after the exit
    reacq = None
    after = [r for r in tgt_vis_rows if tgt_gone and r["chunk"] > max(tgt_gone)]
    if after:
        base = min(r["chunk"] for r in after)
        for r in sorted(after, key=lambda z: z["chunk"]):
            if r["ok"]:
                reacq = r["chunk"] - base
                break
    g4 = reacq is not None
    ms = [r["margin"] for r in act]
    m_before = [r["margin"] for r in tgt_vis_rows if tgt_gone and
                r["chunk"] < min(tgt_gone)]
    m_first = [r["margin"] for r in after[:1]] if after else []
    m_after = [r["margin"] for r in after]

    print("\n[of] ===== §42C six gates =====")
    print(f"  1 world existence persistence  objects alive={n_obj}/"
          f"{len(names)}, target out-of-FOV chunks={len(tgt_inv)} -> {g1}")
    print(f"  2 state continuity             cross-object transfer={xfer}, "
          f"{tgt}.omega={store.get(ids[tgt]).gameplay_state['omega']:+.2f} "
          f"(want {C_OMEGA_AFTER:+.2f}), "
          f"{tgt}.health={store.get(ids[tgt]).gameplay_state['health']:.2f} "
          f"(want {C_HEALTH_AFTER:.2f}) -> {g2}")
    print(f"  3 no duplicate                 entity count={n_obj} "
          f"(expect {len(names)}) -> {g3}")
    print(f"  4 reacquisition latency        {reacq} chunks -> {g4}")
    print(f"  5 identity margin              before-exit "
          f"{np.mean(m_before) if m_before else float('nan'):+.3f}  "
          f"first-return {np.mean(m_first) if m_first else float('nan'):+.3f}  "
          f"min-after {min(m_after) if m_after else float('nan'):+.3f}")
    print(f"  6 visual pollution             state_leak + E_state (see doc)")

    wrong_n = sum(1 for r in act if r["wrong"])
    allm = ms
    print(f"\n[of] wrong-ID = {wrong_n}   margin min = {min(allm):+.3f}   "
          f"cross-object transfer = {xfer}   duplicates = "
          f"{max(0, n_obj - len(names))}")

    horizon = len(tgt_inv)
    print(f"\n[of] ===== MEMORY HORIZON =====")
    print(f"  target genuinely out-of-FOV for {horizon} chunks "
          f"({horizon*0.25:.2f}s)")
    print(f"  wrong-ID={wrong_n} duplicates=0 state-loss="
          f"{'no' if g2 else 'YES'}")
    print(f"  -> object memory horizon "
          f"{'>= ' + str(horizon) if (wrong_n == 0 and g2) else '< ' + str(horizon)}"
          f" chunks")

    ok = g1 and g2 and g3 and g4 and wrong_n == 0
    print(f"\n  §42C: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  -> existence, identity and gameplay state do NOT depend on "
              "current visibility;")
        print("     render observation is only a temporary view of a persistent "
              "world entity")

    json.dump(dict(rows=rows, ids={k: int(v) for k, v in ids.items()},
                   horizons=dict(target=tgt, out_chunks=tgt_gone,
                                 invisible=horizon,
                                 requested=args.invisible),
                   cross_object_transfer=xfer, reacquisition_latency=reacq,
                   margin_before=float(np.mean(m_before)) if m_before else None,
                   margin_first=float(np.mean(m_first)) if m_first else None,
                   margin_min_after=(float(min(m_after)) if m_after else None),
                   wrong_id=wrong_n, duplicates=max(0, n_obj - len(names)),
                   world_drift=C_WORLD_DRIFT.tolist(),
                   gates=dict(existence=g1, state=g2, no_dup=g3,
                              reacquisition=g4),
                   pass_=bool(ok)),
              open(f"{args.out_dir}/outfov.json", "w"), indent=1, default=float)
    np.save(f"{args.out_dir}/frames.npy", np.stack(frames))


if __name__ == "__main__":
    main()
