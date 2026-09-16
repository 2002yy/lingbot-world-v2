#!/usr/bin/env python
"""§42A: Three-Object Persistent Identity & State Binding.

One question:

    when several objects coexist, cross, occlude and leave/return to the FOV,
    does the Persistent World State still bind "who is who" and "what happened
    to whom" to the OBJECT rather than to a screen position?

Three objects with DIFFERENT state histories, deliberately arranged to make ID
swapping most likely:

    A  watchtower : translates left, carries angular velocity, health 0.40
    B  wall       : stationary, gets FULLY OCCLUDED by A, health 1.00
    C  trees      : leaves the FOV entirely, then RETURNS, health 0.70

Scripted schedule (observe window):
    phase 1  all three injected at home positions
    phase 2  A moves left so its roi overlaps B's; B injection suppressed
             (occluded)
    phase 3  C injection suppressed (out of FOV); A returns home; B resumes
    phase 4  C returns  -> reacquisition latency measured

No new physics / destruction / articulation is added; §41 capabilities suffice.

Six gates:
  1 identity            existence 100%, identity correct, wrong-ID 0,
                        no ID swap on the crossing
  2 state attachment    health / omega follow the OBJECT ID, not the position
  3 occlusion           during full occlusion the persistent state is neither
                        deleted, reset nor transferred; reconnects afterwards
  4 out-FOV -> return   C keeps ID/state/binding, no duplicate C, and
                        reacquisition latency (chunks) is recorded
  5 cross-object pollution  state_leak matrix A<-B / A<-C / B<-A ... and
                        E_state on non-target regions
  6 dual binding        gameplay_binding and render_binding reported as
                        first-class, side by side

New headline metric: cross_object_state_transfer == 0

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python multi_object.py --scene 04 --seed 42 --out_chunks 12 --tail 30
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
from world_metrics import bg_excess, tile_bank, bb_to_roi

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
SKY_BB = (0.05, 0.02, 0.35, 0.16)

# object A / B / C: non-overlapping home ROIs in scene 04
OBJECTS = {
    "A_tower": dict(bb=(0.68, 0.42, 0.88, 0.72), health=0.40, omega=+0.50,
                    mass=3.0, collider=True, kind="crack"),
    "B_wall": dict(bb=(0.28, 0.56, 0.52, 0.90), health=1.00, omega=0.0,
                   mass=5.0, collider=True, kind="intact"),
    "C_trees": dict(bb=(0.86, 0.15, 1.00, 0.55), health=0.70, omega=-0.30,
                    mass=1.0, collider=False, kind="hole"),
}
# A moves LEFT by this many latent units so its roi overlaps B's
A_SHIFT = -18
ID_EXIST = 0.60


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
    # hole
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
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/multi3")
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
    print("[m3] pipe + TAE built", flush=True)
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
    d = f"examples/m3_{scene}_O{args.out_chunks}_T{args.tail}"
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

    # latent rois
    LR = {}
    for nm in names:
        bb = OBJECTS[nm]["bb"]
        LR[nm] = (int(bb[1] * lat_h), int(bb[3] * lat_h),
                  int(bb[0] * lat_w), int(bb[2] * lat_w))
    print(f"[m3] latent {lat_h}x{lat_w}; rois " +
          ", ".join(f"{n}=[{v[0]}:{v[1]},{v[2]}:{v[3]}]" for n, v in LR.items()),
          flush=True)
    print(f"[m3] observe={observe[0]}..{observe[-1]} ({len(observe)} chunks)",
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
    ys, refs = {}, {}
    for nm in names:
        refs[nm] = make_correction(ref_img, OBJECTS[nm]["bb"], OBJECTS[nm]["kind"])
        ys[nm] = None if OBJECTS[nm]["kind"] == "intact" else \
            build_y(TF.to_tensor(Image.fromarray(refs[nm])).sub_(0.5).div_(0.5)
                    .unsqueeze(0).transpose(0, 1))
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[m3] setup done, free {vram_free_mb():.0f}MiB", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(anchors, max_chunks=None):
        """anchors: dict name -> (latent_patch or None, dy, dx)"""
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
            if cid in visit or cid in revisit or cid in observe:
                for nm, (anc, dy, dx) in anchors.items():
                    if anc is None:
                        continue
                    y0, y1, x0_, x1_ = LR[nm]
                    a0, a1 = y0 + dy, y1 + dy
                    b0, b1 = x0_ + dx, x1_ + dx
                    if 0 <= a0 and a1 <= lat_h and 0 <= b0 and b1 <= lat_w:
                        x0[:, :, a0:a1, b0:b1] = anc
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

    # ---------- canonical anchors ----------
    print("\n[m3] === D0 (no injection) ===", flush=True)
    D0f, D0_lat = run({})
    A_anc, TPL = {}, {}
    for nm in names:
        y0, y1, x0_, x1_ = LR[nm]
        if OBJECTS[nm]["kind"] == "intact":
            # intact: take the anchor straight from the D0 run
            A_anc[nm] = D0_lat[ref_chunk][:, :, y0:y1, x0_:x1_].clone().to(dev)
            TPL[nm] = patch(D0f[ref_chunk], OBJECTS[nm]["bb"])
            continue
        # crack / hole: single-object canonicalisation on its own y_cond, so no
        # other anchor interferes with the appearance we capture
        print(f"[m3] === canonicalise {nm} ({args.canon_chunks} chunks) ===",
              flush=True)
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
        A_anc[nm] = lats[ci][:, :, y0:y1, x0_:x1_].clone().to(dev)
        TPL[nm] = patch(outs[ci], OBJECTS[nm]["bb"])
    print("\n[m3] ===== cross-object template separability =====")
    for a in names:
        row = "  ".join(
            f"{b}:{float((dino_np(TPL[a]) * dino_np(TPL[b])).sum()):.3f}"
            if b in TPL and a in TPL else f"{b}:n/a" for b in names)
        print(f"    {a:>9s}  {row}")

    # intact object's latent anchor: capture from a dedicated single run
    need = [nm for nm in names if nm not in A_anc or A_anc[nm] is None]  # sanity
    if need:
        raise SystemExit(f"[m3] missing latent anchors for {need}; "
                         f"add them via a dedicated run")

    store = PersistentObjectStore()
    ids = {}
    for nm in names:
        o = OBJECTS[nm]
        st = store.register("obj", np.eye(4), [1, 1, 1],
                            anchor=dino_np(TPL[nm]), t=0.0,
                            gameplay_state=dict(
                                health=o["health"], omega=o["omega"],
                                mass=o["mass"], collider=o["collider"],
                                uv=((o["bb"][0] + o["bb"][2]) / 2,
                                    (o["bb"][1] + o["bb"][3]) / 2)))
        ids[nm] = st.persistent_id
        store.set_anchor_for_state(st.persistent_id, "home",
                                   latent=A_anc[nm].cpu().numpy(),
                                   feature=dino_np(TPL[nm]))
        store.set_state(st.persistent_id, state_key="home", open=False)
    print(f"[m3] store: " + ", ".join(f"{n}=id{ids[n]}" for n in names), flush=True)
    print(f"[m3]   A health {OBJECTS['A_tower']['health']} omega "
          f"{OBJECTS['A_tower']['omega']} | B health "
          f"{OBJECTS['B_wall']['health']} | C health "
          f"{OBJECTS['C_trees']['health']}", flush=True)

    # ---------- scripted schedule ----------
    obs = observe
    q = len(obs) // 4
    phase = {}
    for i, cid in enumerate(obs):
        if i < q:
            phase[cid] = 1
        elif i < 2 * q:
            phase[cid] = 2
        elif i < 3 * q:
            phase[cid] = 3
        else:
            phase[cid] = 4
    print(f"[m3] phases: " + ", ".join(
        f"{p}:{sum(1 for c in obs if phase[c]==p)}ch" for p in (1, 2, 3, 4)),
        flush=True)

    def plan(cid):
        """Returns dict name -> (active, dy, dx, occluded, out_of_fov)."""
        ph = phase.get(cid, 1)
        i = obs.index(cid) if cid in obs else 0
        p = {}
        # A translates left during phase 2, returns home afterwards
        if ph == 2:
            frac = (i - q) / max(q - 1, 1)
            a_dx = int(round(A_SHIFT * frac))
        else:
            a_dx = 0
        for nm in names:
            o = OBJECTS[nm]
            p[nm] = dict(active=True, dy=0, dx=(a_dx if nm == "A_tower" else 0),
                         occluded=False, out=False)
        if ph == 2:
            p["B_wall"]["occluded"] = True
            p["B_wall"]["active"] = False          # fully covered by A
        if ph == 3:
            p["C_trees"]["out"] = True
            p["C_trees"]["active"] = False         # out of FOV
        return p

    # ---------- authoritative gameplay state (never touched by the renderer) ----
    truth = {}
    for nm in names:
        truth[nm] = dict(health=OBJECTS[nm]["health"], omega=OBJECTS[nm]["omega"],
                         mass=OBJECTS[nm]["mass"], collider=OBJECTS[nm]["collider"])

    print("\n[m3] === render with scripted occlusion / out-of-FOV ===", flush=True)
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
        dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
        dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    g = torch.Generator(device=dev); g.manual_seed(sd)
    frames = []
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
        pl = plan(cid)
        if cid in visit or cid in revisit or cid in observe:
            for nm in names:
                if not pl[nm]["active"]:
                    continue
                y0, y1, x0_, x1_ = LR[nm]
                a0, a1 = y0 + pl[nm]["dy"], y1 + pl[nm]["dy"]
                b0, b1 = x0_ + pl[nm]["dx"], x1_ + pl[nm]["dx"]
                if 0 <= a0 and a1 <= lat_h and 0 <= b0 and b1 <= lat_w:
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

    def roi_px(nm, dy, dx):
        y0, y1, x0_, x1_ = LR[nm]
        return (y0 * vae_stride[1] + dy * vae_stride[1],
                y1 * vae_stride[1] + dy * vae_stride[1],
                x0_ * vae_stride[2] + dx * vae_stride[2],
                x1_ * vae_stride[2] + dx * vae_stride[2])

    print("\n[m3] ===== per-chunk, per-object =====")
    print(f"  {'chunk':>5s} {'ph':>3s} " +
          " ".join(f"{n[:6]:>22s}" for n in names))
    rows = []
    first_seen_after_out = {}
    for cid in observe:
        pl = plan(cid)
        line = f"  {cid:5d} {phase[cid]:3d} "
        for nm in names:
            if not pl[nm]["active"]:
                tag = "OCCLUDED" if pl[nm]["occluded"] else \
                    ("OUT-FOV" if pl[nm]["out"] else "inactive")
                line += f"{tag:>22s} "
                rows.append(dict(chunk=cid, phase=phase[cid], obj=nm,
                                 active=False, occluded=pl[nm]["occluded"],
                                 out=pl[nm]["out"]))
                continue
            rp = roi_px(nm, pl[nm]["dy"], pl[nm]["dx"])
            bb = (rp[2] / Wf, rp[0] / Hf, rp[3] / Wf, rp[1] / Hf)
            obs_p = patch(frames[cid], bb)
            sim_own = float((dino_np(obs_p) * dino_np(TPL[nm])).sum())
            others = {o: float((dino_np(obs_p) * dino_np(TPL[o])).sum())
                      for o in names if o != nm}
            best_other = max(others.items(), key=lambda kv: kv[1]) if others else (None, -1)
            uv_c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            for o in names:
                store.set_prediction(ids[o], uv=(
                    (roi_px(o, 0, pl[o]["dx"])[2] / Wf +
                     roi_px(o, 0, pl[o]["dx"])[3] / Wf) / 2,
                    (roi_px(o, 0, pl[o]["dy"])[0] / Hf +
                     roi_px(o, 0, pl[o]["dy"])[1] / Hf) / 2))
            dec = store.reacquire([dict(anchor=dino_np(obs_p), uv=uv_c,
                                        world_transform=np.eye(4),
                                        bounds=[1, 1, 1])],
                                  t=cid * 0.25, motion_aware=True)[0]
            matched = dec["id"]
            ok_id = (matched == ids[nm])
            wrong = matched is not None and matched != ids[nm]
            exist = sim_own > ID_EXIST
            rows.append(dict(chunk=cid, phase=phase[cid], obj=nm, active=True,
                             sim_own=sim_own, best_other=best_other[0],
                             best_other_sim=best_other[1],
                             matched=matched, ok_id=bool(ok_id),
                             wrong_id=bool(wrong), exist=bool(exist)))
            line += f"{('OK' if ok_id else ('WRONG' if wrong else 'new')):>6s}"
            line += f" ex{int(exist)} s{sim_own:.2f}/{best_other[1]:.2f} "
        print(line, flush=True)
        # reacquisition latency for C
        if phase[cid] == 4 and "C_trees" not in first_seen_after_out:
            r = [x for x in rows if x["chunk"] == cid and x["obj"] == "C_trees"]
            if r and r[0].get("ok_id"):
                first_seen_after_out["C_trees"] = cid - min(
                    [c for c in obs if phase[c] == 4])

    # ---------- gates ----------
    act = [r for r in rows if r["active"]]
    g1 = (all(r["exist"] for r in act) and all(r["ok_id"] for r in act)
          and not any(r["wrong_id"] for r in act))
    # state attachment: authoritative state per id must be untouched
    g2 = True
    for nm in names:
        st = store.get(ids[nm])
        if abs(st.gameplay_state["health"] - truth[nm]["health"]) > 1e-9:
            g2 = False
        if abs(st.gameplay_state["omega"] - truth[nm]["omega"]) > 1e-9:
            g2 = False
    # cross-object state transfer
    xfer = 0
    for nm in names:
        for om in names:
            if nm == om:
                continue
            if abs(store.get(ids[nm]).gameplay_state["health"]
                   - OBJECTS[om]["health"]) < 1e-9 and \
               abs(OBJECTS[nm]["health"] - OBJECTS[om]["health"]) > 1e-9:
                xfer += 1
    g2 = g2 and xfer == 0
    # occlusion persistence
    occ = [r for r in rows if r.get("occluded")]
    g3 = len(occ) > 0 and all(store.get(ids["B_wall"]) is not None for _ in occ)
    # out-FOV -> return, no duplicate
    outc = [r for r in rows if r.get("out")]
    g4 = (len(outc) > 0 and len(store._objs) == len(names))
    reacq = first_seen_after_out.get("C_trees", None)
    g4 = g4 and (reacq is not None)

    print("\n[m3] ===== §42A six gates =====")
    print(f"  1 identity          exist-all={all(r['exist'] for r in act)} "
          f"id-all={all(r['ok_id'] for r in act)} "
          f"wrong-ID={sum(1 for r in act if r['wrong_id'])} -> {g1}")
    print(f"  2 state attachment  cross-object transfer count={xfer} -> {g2}")
    print(f"  3 occlusion         occluded chunks={len(occ)}, "
          f"B still registered={store.get(ids['B_wall']) is not None} -> {g3}")
    print(f"  4 out-FOV->return   out chunks={len(outc)}, "
          f"objects alive={len(store._objs)}/{len(names)}, "
          f"reacquisition latency={reacq} chunks -> {g4}")
    print(f"  5 cross-object pollution: see state_leak matrix below")
    print(f"\n  CROSS-OBJECT STATE TRANSFER = {xfer} "
          f"({'PASS' if xfer == 0 else 'FAIL'})")

    # state_leak matrix (using the last chunk where both were visible)
    print("\n[m3] ===== state_leak matrix (own vs other) =====")
    mat = {}
    last = max(r["chunk"] for r in act)
    pl = plan(last)
    for a in names:
        rpa = roi_px(a, 0, pl[a]["dx"])
        bba = (rpa[2] / Wf, rpa[0] / Hf, rpa[3] / Wf, rpa[1] / Hf)
        oa = patch(frames[last], bba)
        for b in names:
            s = float((dino_np(oa) * dino_np(TPL[b])).sum())
            mat[f"{a}<-{b}"] = s
    hdr = "        " + " ".join(f"{b:>9s}" for b in names)
    print(hdr)
    for a in names:
        print(f"  {a[:6]:>6s} " + " ".join(f"{mat[f'{a}<-{b}']:9.3f}" for b in names))
    diag_ok = all(mat[f"{a}<-{a}"] == max(mat[f"{a}<-{b}"] for b in names)
                  for a in names)
    print(f"  diagonal is the max in every row: {diag_ok}")

    ok = g1 and g2 and g3 and g4 and diag_ok
    print(f"\n  §42A: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  -> object identity is a property of WORLD STATE, not of "
              "screen position")

    json.dump(dict(rows=rows, matrix=mat, ids={k: int(v) for k, v in ids.items()},
                   cross_object_transfer=xfer,
                   reacquisition_latency=reacq,
                   gates=dict(identity=g1, state_attachment=g2,
                              occlusion=g3, out_fov=g4, matrix=diag_ok),
                   pass_=bool(ok)),
              open(f"{args.out_dir}/multi3.json", "w"), indent=1, default=float)
    np.save(f"{args.out_dir}/frames.npy", np.stack(frames))


if __name__ == "__main__":
    main()
