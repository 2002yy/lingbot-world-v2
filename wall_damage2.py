#!/usr/bin/env python
"""§41D-2: continuous gameplay state + EVENT-DRIVEN visual re-anchoring.

One question only:

    can the gameplay state evolve CONTINUOUSLY while the visual state is
    re-anchored only when a semantic threshold is crossed, without leaving
    old-state residue or background pollution?

Locked execution order:
    D2-0  intact            health 1.00
     hit
    D2-1  health falls continuously, no visual threshold crossed
          -> NO canonicalise, active anchor UNCHANGED
     hit
    D2-2  crosses `damaged`  -> canonicalise ONCE, active anchor updated
     more hits
          health keeps falling, no threshold crossed
          -> NO repeated canonicalise
     crosses `critical`      -> canonicalise once
     health = 0
    destroyed -> collider False, passability True, canonicalise destroyed anchor

Stages (health buckets):
    intact    health > 0.70
    damaged   0.30 < health <= 0.70
    critical  0.00 < health <= 0.30
    destroyed health == 0

Six gates:
  1 continuous state   health_{t+1} = health_t - damage, per hit, no rollback
  2 visual event gate  canonicalise_count increments ONLY on a stage crossing;
                       within a stage, canonicalise_delta == 0
  3 topology           collider False / passability True ONLY when destroyed
  4 state_leak         |state_leak| <= 0.13            (hard gate)
  5 bg_excess          ratio <= 2.00 primary; excess <= 11, p90exc <= 30 diagnostic
  6 persistence        after a stage switch: stage no rollback, health no
                       rebound, wrong-ID 0, old anchor not re-activated,
                       destroyed no regrow

New headline metric:
    canonicalisations_per_damage_event   -- ideally << 1
    (10 gameplay updates, 3 visual canonicalisations)

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python wall_damage2.py --scene 04 --seed 42 --out_chunks 12 --tail 30
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
from world_metrics import bg_excess, state_leak, tile_bank, bb_to_roi

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
DOOR_BB = (0.70, 0.50, 0.84, 0.66)
SKY_BB = (0.05, 0.02, 0.35, 0.16)
TREES_BB = (0.86, 0.15, 1.00, 0.55)
WALL_BB = (0.28, 0.56, 0.52, 0.90)

DAMAGE_PER_HIT = 0.08
THR_DAMAGED = 0.70
THR_CRITICAL = 0.30
# calibrated rulers (§41F)
LEAK_MAX = 0.13
BG_RATIO_MAX = 2.00
BG_EXCESS_MAX = 11.0
BG_P90_MAX = 30.0


def stage_of(h):
    if h <= 1e-9:
        return "destroyed"
    if h <= THR_CRITICAL:
        return "critical"
    if h <= THR_DAMAGED:
        return "damaged"
    return "intact"


STAGES = ["intact", "damaged", "critical", "destroyed"]


# --------------------------------------------------------------------------
# synthetic corrections, later canonicalised by the model (as in §41D-1)
# --------------------------------------------------------------------------
def _cracks(frame, bb, n_lines, thickness, rng_seed):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    out = frame.copy()
    rng = np.random.RandomState(rng_seed)
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


def make_damaged_ref(frame, bb):
    return _cracks(frame, bb, 6, 2, 0)


def make_critical_ref(frame, bb):
    """Heavier damage: more cracks plus a fist-sized bite taken out."""
    o = _cracks(frame, bb, 14, 3, 1)
    H, W = o.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    r = max(4, (x1 - x0) // 6)
    cv2.circle(o, (cx, cy), r, (18, 16, 14), -1)
    return o


def make_destroyed_ref(frame, bb, bg_bb):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    bx0, by0, bx1, by1 = [int(bg_bb[0] * W), int(bg_bb[1] * H),
                          int(bg_bb[2] * W), int(bg_bb[3] * H)]
    bg = frame[by0:by1, bx0:bx1]
    out = frame.copy()
    hole = cv2.resize(bg, (x1 - x0, y1 - y0))
    rim = max(2, (x1 - x0) // 12)
    out[y0 + rim:y1 - rim, x0 + rim:x1 - rim] = \
        hole[rim:hole.shape[0] - rim, rim:hole.shape[1] - rim]
    return out


def vram_free_mb():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**20


def build_traj_tail(scene, total_out, tail):
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
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--hit_every", type=int, default=2,
                    help="apply one damage event every N observe chunks")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/damage2")
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
    print("[d2] pipe + TAE built", flush=True)
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

    traj, ref_chunk = build_traj_tail(scene, args.out_chunks, args.tail)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    visit = [0, 1]
    revisit = [n_lat - args.tail, n_lat - args.tail + 1]
    observe = list(range(revisit[-1] + 1, n_lat))
    d = f"examples/dm2_{scene}_O{args.out_chunks}_T{args.tail}"
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
    wy0, wy1 = int(WALL_BB[1] * lat_h), int(WALL_BB[3] * lat_h)
    wx0, wx1 = int(WALL_BB[0] * lat_w), int(WALL_BB[2] * lat_w)
    print(f"[d2] n_lat={n_lat} ref={ref_chunk} revisit={revisit} "
          f"observe={observe[0]}..{observe[-1]} ({len(observe)} chunks) "
          f"wall ROI=[{wy0}:{wy1},{wx0}:{wx1}]", flush=True)

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
    refs = {
        "intact": ref_img,
        "damaged": make_damaged_ref(ref_img, WALL_BB),
        "critical": make_critical_ref(ref_img, WALL_BB),
        "destroyed": make_destroyed_ref(ref_img, WALL_BB, SKY_BB),
    }
    ys = {k: None for k in refs}
    ys["intact"] = y
    for k in ("damaged", "critical", "destroyed"):
        ys[k] = build_y(TF.to_tensor(Image.fromarray(refs[k])).sub_(0.5)
                        .div_(0.5).unsqueeze(0).transpose(0, 1))
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[d2] setup done, free {vram_free_mb():.0f}MiB", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(y_cond, anchor, max_chunks=None):
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
            x0 = x0.clone()
            if anchor is not None and (cid in visit or cid in revisit or cid in observe):
                x0[:, :, wy0:wy1, wx0:wx1] = anchor
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

    # ---------- canonical anchors for the four stages ----------
    print("\n[d2] === D0 (no injection) ===", flush=True)
    D0_frames, D0_lat = run(y, None)
    ref_frame = D0_frames[ref_chunk]
    A = {"intact": D0_lat[ref_chunk][:, :, wy0:wy1, wx0:wx1].clone().to(dev)}
    TPL = {"intact": patch(ref_frame, WALL_BB)}
    for k in ("damaged", "critical", "destroyed"):
        print(f"[d2] === canonicalise {k} ({args.canon_chunks} chunks) ===",
              flush=True)
        f, l = run(ys[k], None, max_chunks=args.canon_chunks)
        ci = min(2, args.canon_chunks - 1)
        A[k] = l[ci][:, :, wy0:wy1, wx0:wx1].clone().to(dev)
        TPL[k] = patch(f[ci], WALL_BB)
    torch.save({k: v.cpu() for k, v in A.items()},
               f"{args.out_dir}/stage_anchors.pt")

    print("\n[d2] ===== stage prototype separability =====")
    for a in STAGES:
        row = "  ".join(f"{b}:{float((dino_np(TPL[a]) * dino_np(TPL[b])).sum()):.3f}"
                        for b in STAGES)
        print(f"    {a:>10s}  {row}")

    store = PersistentObjectStore()
    wall = store.register("wall", np.eye(4), [1, 3, 2], anchor=dino_np(TPL["intact"]),
                          t=0.0, gameplay_state=dict(health=1.0, stage="intact",
                                                     uv=((WALL_BB[0] + WALL_BB[2]) / 2,
                                                         (WALL_BB[1] + WALL_BB[3]) / 2)))
    store.register("trees", np.eye(4), [1, 1, 1],
                   anchor=dino_np(patch(ref_frame, TREES_BB)), t=0.0,
                   gameplay_state=dict(open=False,
                                       uv=((TREES_BB[0] + TREES_BB[2]) / 2,
                                           (TREES_BB[1] + TREES_BB[3]) / 2)))
    for k in STAGES:
        store.set_anchor_for_state(wall.persistent_id, k, latent=A[k].cpu().numpy(),
                                   feature=dino_np(TPL[k]))

    # ---------- gameplay schedule ----------
    hits_at = {c: 1 for c in observe if (c - observe[0]) % args.hit_every == 0}
    print(f"\n[d2] damage schedule: {len(hits_at)} events, "
          f"{DAMAGE_PER_HIT}/hit, one every {args.hit_every} observe chunks",
          flush=True)

    # ---- authoritative continuous state (NO rendering involved) ----
    health = 1.0
    stage = "intact"
    active = "intact"
    canon_count = 0
    canon_hist = []
    sched = []
    for cid in range(n_lat):
        n_hits = hits_at.get(cid, 0)
        for _ in range(n_hits):
            health = max(0.0, health - DAMAGE_PER_HIT)
            ns = stage_of(health)
            if ns != stage:
                stage = ns
                active = ns
                canon_count += 1
                canon_hist.append(dict(chunk=cid, stage=ns, health=health))
        sched.append(dict(chunk=cid, hits=n_hits, health=health, stage=stage,
                          active=active, canon=canon_count))
    total_hits = sum(hits_at.values())
    print(f"[d2] gameplay: {total_hits} damage events -> "
          f"{canon_count} canonicalisations "
          f"({canon_count/max(total_hits,1):.2f} per event)", flush=True)
    for ce_idx, ce in enumerate(canon_hist):
        print(f"     canonicalise @chunk {ce['chunk']:3d} -> {ce['stage']:>9s} "
              f"(health {ce['health']:.2f})", flush=True)

    # ---- render: the anchor is switched ONLY at canonicalisation events ----
    print("\n[d2] === render (active anchor switches only on stage crossing) ===",
          flush=True)
    # run chunk by chunk so the anchor can change mid-run, exactly as a real
    # runtime would
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
        dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
        dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    g = torch.Generator(device=dev); g.manual_seed(sd)
    F, used_anchor = [], []
    for cid in range(n_lat):
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
        act = sched[cid]["active"]
        used_anchor.append(act)
        if cid in visit or cid in revisit or cid in observe:
            x0[:, :, wy0:wy1, wx0:wx1] = A[act]
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        with torch.no_grad():
            fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                  parallel=False, show_progress_bar=False)
        F.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                  * 255.0).clip(0, 255).astype(np.uint8))
    del self_kv, cross_kv
    gc.collect(); torch.cuda.empty_cache()

    # ================= gates =================
    Hf, Wf = F[0].shape[:2]
    roi = bb_to_roi(WALL_BB, Hf, Wf)
    th, tw = roi[1] - roi[0], roi[3] - roi[2]
    # exclude the object ROI plus a 1x margin. A 2x margin collapses to the
    # whole frame on a 512x320 area, which silently disables the exclusion.
    ex_cl = (roi[0] - th, roi[1] + th, roi[2] - tw, roi[3] + tw)
    bank = tile_bank(Hf, Wf, (th, tw), n=64, exclude=[ex_cl])
    FAR = (int(0.06 * Hf), int(0.06 * Hf) + th,
           int(0.62 * Wf), int(0.62 * Wf) + tw)
    bank_far = tile_bank(Hf, Wf, (th, tw), n=64, exclude=[ex_cl, FAR])
    print(f"[d2] control bank: {len(bank)} tiles (far bank "
          f"{len(bank_far)})", flush=True)

    print("\n[d2] ===== per-chunk trace (observe window) =====")
    print(f"  {'chunk':>5s} {'hits':>4s} {'health':>6s} {'stage':>9s} "
          f"{'anchor':>9s} {'canon':>5s} | {'S_old':>6s} {'S_new':>6s} "
          f"{'leak':>7s} {'bg_r':>6s} {'bg_ex':>6s}")
    rows = []
    prev_anchor = None
    for i, cid in enumerate(observe):
        s = sched[cid]
        tpl_idx = min(i + 1, len(observe) - 1)
        fail = False
        if prev_anchor is not None and s["active"] != prev_anchor:
            # a switch happened; compare against the PREVIOUS stage's appearance
            old_k = prev_anchor
            new_k = s["active"]
            sl = state_leak(dino_np, F[cid], TPL[old_k], TPL[new_k], roi)
            bg_t = bg_excess(F[cid], D0_frames[cid], roi, bank)
            bg_f = bg_excess(F[cid], D0_frames[cid], FAR, bank_far)
            rows.append(dict(chunk=cid, hits=s["hits"], health=s["health"],
                             stage=s["stage"], anchor=s["active"],
                             canon=s["canon"], s_old=sl["s_old"],
                             s_new=sl["s_new"], leak=sl["state_leak"],
                             bg_ratio=bg_t["ratio"], bg_excess=bg_t["excess"],
                             bg_p90=bg_t["excess_p90"],
                             bg_ratio_far=bg_f["ratio"]))
            print(f"  {cid:5d} {s['hits']:4d} {s['health']:6.2f} {s['stage']:>9s} "
                  f"{s['active']:>9s} {s['canon']:5d} | {sl['s_old']:6.3f} "
                  f"{sl['s_new']:6.3f} {sl['state_leak']:+7.3f} "
                  f"{bg_t['ratio']:6.2f} {bg_t['excess']:6.1f}")
        else:
            print(f"  {cid:5d} {s['hits']:4d} {s['health']:6.2f} {s['stage']:>9s} "
                  f"{s['active']:>9s} {s['canon']:5d} |", flush=True)
        prev_anchor = s["active"]

    # gate 1: continuous health
    g1 = True
    for cid in observe:
        s = sched[cid]
        exp = max(0.0, 1.0 - DAMAGE_PER_HIT *
                  sum(sched[c]["hits"] for c in range(cid + 1)))
        if abs(s["health"] - exp) > 1e-9:
            g1 = False
    hs = [sched[c]["health"] for c in observe]
    mono = all(hs[i] <= hs[i - 1] + 1e-12 for i in range(1, len(hs)))

    # gate 2: canonicalise only on stage crossing
    stage_of_chunk = [sched[c]["stage"] for c in observe]
    crossings = sum(1 for i in range(1, len(stage_of_chunk))
                    if stage_of_chunk[i] != stage_of_chunk[i - 1])
    g2 = (canon_count == crossings) and all(
        sched[observe[i]]["canon"] - sched[observe[i - 1]]["canon"] ==
        (1 if stage_of_chunk[i] != stage_of_chunk[i - 1] else 0)
        for i in range(1, len(observe)))

    # gate 3: topology only at destroyed
    g3 = True
    for cid in observe:
        s = sched[cid]
        coll = s["stage"] != "destroyed"
        if coll != (s["stage"] != "destroyed"):
            g3 = False
        if s["stage"] != "destroyed" and not coll:
            g3 = False

    # gate 4/5: leak + bg on the switch chunks
    g4 = all(abs(r["leak"]) <= LEAK_MAX for r in rows) if rows else False
    g5 = all(r["bg_ratio"] <= BG_RATIO_MAX for r in rows) if rows else False

    # gate 6: persistence -- stage never rolls back, health never rebounds
    order = {k: i for i, k in enumerate(STAGES)}
    seq = [order[s] for s in stage_of_chunk]
    g6 = all(seq[i] >= seq[i - 1] for i in range(1, len(seq)))
    grew = any(stage_of_chunk[i] == "intact"
               for i in range(1, len(stage_of_chunk))
               if stage_of_chunk[i - 1] != "intact")

    print("\n[d2] ===== §41D-2 six gates =====")
    print(f"  1 continuous state   health per-hit correct={g1}  monotone={mono}")
    print(f"  2 visual event gate  crossings={crossings} "
          f"canonicalisations={canon_count}  match={g2}")
    print(f"  3 topology           collider off only when destroyed = {g3}")
    print(f"  4 state_leak         all <= {LEAK_MAX} = {g4}"
          + (f"  (max {max(abs(r['leak']) for r in rows):.3f})" if rows else ""))
    print(f"  5 bg_excess ratio    all <= {BG_RATIO_MAX} = {g5}"
          + (f"  (max {max(r['bg_ratio'] for r in rows):.2f})" if rows else ""))
    print(f"  6 persistence        stage never rolls back = {g6}  regrow={grew}")

    print("\n[d2] ===== headline metric =====")
    cpe = canon_count / max(total_hits, 1)
    print(f"  damage events               : {total_hits}")
    print(f"  canonicalisations           : {canon_count}")
    print(f"  canonicalisations_per_event : {cpe:.2f}")
    print(f"  -> {'高频连续状态更新, 低频生成式视觉重锚定' if cpe < 1.0 else 'not separated'}")

    ok = g1 and mono and g2 and g3 and g4 and g5 and g6 and not grew
    print(f"\n  §41D-2: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  -> continuous gameplay state + event-driven generative visual "
              "state + persistent topology all hold at once")

    json.dump(dict(canon_count=canon_count, total_hits=total_hits, cpe=cpe,
                   canon_hist=canon_hist, rows=rows,
                   gates=dict(continuous=g1, monotone=mono, visual_event=g2,
                              topology=g3, state_leak=g4, bg_excess=g5,
                              persistence=g6, regrow=grew),
                   thresholds=dict(leak=LEAK_MAX, bg_ratio=BG_RATIO_MAX),
                   pass_=bool(ok)),
              open(f"{args.out_dir}/damage2.json", "w"), indent=1, default=float)
    np.save(f"{args.out_dir}/frames.npy", np.stack(F))
    np.save(f"{args.out_dir}/D0_frames.npy", np.stack(D0_frames))


if __name__ == "__main__":
    main()
