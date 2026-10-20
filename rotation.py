#!/usr/bin/env python
"""§41B-3: Rotation -- is the 6x8 latent anchor an IDENTITY representation or a
VIEW-CONDITIONED appearance snapshot?

Translation never stressed the relative view much, so this stayed hidden. The
test: keep the camera fixed and make the object's orientation change, with
P1 refresh (translation is now closed, so cadence is not a variable).

    R0       inject the canonical anchor UNCHANGED (no pose handling)
    R1_15    inject the latent patch rotated so that 15 deg accumulates
    R1_30    ... 30 deg
    R1_60    ... 60 deg
    R1_90    ... 90 deg

Two things are measured:
  * existence / identity / wrong-ID / pos_err  -- does the object survive
  * appearance drift vs the original anchor     -- did the view actually change

If existence/identity fall systematically with angle, the anchor is a
view-conditioned snapshot and the render binding needs multi-view prototypes:
    render_binding.view_anchors{yaw_0, yaw_45, yaw_90, ...}
    anchor = f(object_state, relative_view)

NOTE on scope: a multi-view PROTOTYPE arm (R2) needs anchors captured from
several real relative views, which this scene does not provide. R1 is the cheap
control the user asked for; whether latent-space rotation is geometrically
meaningful is exactly what the appearance-drift column reveals.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python rotation.py --scene 04 --seed 42 --out_chunks 12 --tail 30
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from cam_controller import CameraController
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange
from object_permanence import patch
from door_state_probe import make_open_ref, surround_brightness_of, \
    structural_features
from object_state import PersistentObjectStore

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
DOOR_BB = (0.70, 0.50, 0.84, 0.66)
SKY_BB = (0.05, 0.02, 0.35, 0.16)
TREES_BB = (0.86, 0.15, 1.00, 0.55)

ARMS = [("R0", 0.0), ("R1_15", 15.0), ("R1_30", 30.0),
        ("R1_60", 60.0), ("R1_90", 90.0)]


def vram_free_mb():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**20


def rotate_latent(a, deg):
    """Rotate a [C,1,h,w] latent patch about its centre by `deg` degrees."""
    if abs(deg) < 1e-3:
        return a
    C, T, hh, ww = a.shape
    x = a.permute(1, 0, 2, 3)                       # [1, C, h, w]
    th = math.radians(deg)
    c, s = math.cos(th), math.sin(th)
    theta = torch.tensor([[c, -s, 0.0], [s, c, 0.0]], dtype=x.dtype,
                         device=x.device).unsqueeze(0)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    y = F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                      align_corners=False)
    return y.permute(1, 0, 2, 3)


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


def best_match_loc(ref_patch, img, bb, dino_np, search=0.08, steps=13):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = bb
    pw, ph = int((x1 - x0) * W), int((y1 - y0) * H)
    fr = dino_np(ref_patch)
    best = (-1.0, None)
    for dy in np.linspace(-search, search, steps):
        for dx in np.linspace(-search, search, steps):
            bx = int((x0 + dx) * W); by = int((y0 + dy) * H)
            if bx < 0 or by < 0 or bx + pw > W or by + ph > H:
                continue
            cand = img[by:by + ph, bx:bx + pw]
            if cand.size == 0:
                continue
            s = float((fr * dino_np(cand)).sum())
            if s > best[0]:
                best = (s, (bx, by))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_chunks", type=int, default=12)
    ap.add_argument("--tail", type=int, default=30)
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--arms", default=",".join(a for a, _ in ARMS))
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/rot")
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
    print("[ro] pipe + TAE built", flush=True)
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
    revisit = [n_lat - args.tail, n_lat - args.tail + 1]
    observe = list(range(revisit[-1] + 1, n_lat))
    d = f"examples/ro_{scene}_O{args.out_chunks}_T{args.tail}"
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
    print(f"[ro] n_lat={n_lat} ref={ref_chunk} revisit={revisit} "
          f"observe={observe[0]}..{observe[-1]} ({len(observe)} chunks)", flush=True)

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
    open_img = make_open_ref(ref_img, DOOR_BB, SKY_BB)
    open_t = TF.to_tensor(Image.fromarray(open_img)).sub_(0.5).div_(0.5) \
        .unsqueeze(0).transpose(0, 1)
    y_open = build_y(open_t)
    dy0, dy1 = int(DOOR_BB[1] * lat_h), int(DOOR_BB[3] * lat_h)
    dx0, dx1 = int(DOOR_BB[0] * lat_w), int(DOOR_BB[2] * lat_w)
    dh, dw = dy1 - dy0, dx1 - dx0
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[ro] setup done, free {vram_free_mb():.0f}MiB, object {dw}x{dh} latent",
          flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(y_cond, anchor=None, deg_total=0.0, max_chunks=None):
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, ang_log, lats = [], [], []
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
            # accumulated authoritative rotation over the observe window
            if cid in revisit:
                deg = 0.0
            elif cid in observe:
                t = cid - revisit[-1]
                deg = deg_total * t / max(len(observe), 1)
            else:
                deg = 0.0
            ang_log.append(deg)
            if anchor is not None and (cid in revisit or cid in observe):
                a = rotate_latent(anchor, deg) if deg_total > 0 else anchor
                x0[:, :, dy0:dy1, dx0:dx1] = a
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
        return outs, ang_log, lats

    print("\n[ro] === D0 ===", flush=True)
    D0_frames, _, D0_lat = run(y, None)
    ref_frame = D0_frames[ref_chunk]
    ref_door = patch(ref_frame, DOOR_BB)
    print(f"[ro] === canonicalisation ===", flush=True)
    canon_frames, _, canon_lat = run(y_open, None, max_chunks=args.canon_chunks)
    ci = min(2, args.canon_chunks - 1)
    canon_door = patch(canon_frames[ci], DOOR_BB)
    CANON_FEAT = dino_np(canon_door)
    Z_OPEN = canon_lat[ci][:, :, dy0:dy1, dx0:dx1].clone().to(dev)

    store = PersistentObjectStore()
    door = store.register("door", np.eye(4), [1, 2, 0.2], anchor=dino_np(ref_door),
                          t=0.0, gameplay_state=dict(open=False,
                                                     uv=((DOOR_BB[0] + DOOR_BB[2]) / 2,
                                                         (DOOR_BB[1] + DOOR_BB[3]) / 2)))
    store.register("trees", np.eye(4), [1, 1, 1],
                   anchor=dino_np(patch(ref_frame, TREES_BB)), t=0.0,
                   gameplay_state=dict(open=False,
                                       uv=((TREES_BB[0] + TREES_BB[2]) / 2,
                                           (TREES_BB[1] + TREES_BB[3]) / 2)))
    store.set_anchor_for_state(door.persistent_id, "closed", feature=dino_np(ref_door))
    store.set_anchor_for_state(door.persistent_id, "open", feature=CANON_FEAT)
    store.set_state(door.persistent_id, open=True)

    # ---- canonical OPEN latent anchor (rotation operates on this) ----
    print(f"[ro] canonical latent anchor captured: {tuple(Z_OPEN.shape)}", flush=True)

    summary = {}
    for aname in args.arms.split(","):
        deg_total = dict(ARMS).get(aname)
        if deg_total is None:
            continue
        print(f"\n[ro] === {aname}: accumulated {deg_total:.0f} deg, P1 ===",
              flush=True)
        F, ang, _ = run(y, Z_OPEN, deg_total=deg_total)
        # measure
        dinos, exs, poss, ids, wids, apps = [], [], [], [], 0, []
        for cid in observe:
            bb = (DOOR_BB[0], DOOR_BB[1], DOOR_BB[2], DOOR_BB[3])
            fr = F[cid]
            sim, loc = best_match_loc(canon_door, fr, bb, dino_np)
            dinos.append(sim)
            exs.append(bool(sim > 0.6))
            exp = (int(bb[0] * fr.shape[1]), int(bb[1] * fr.shape[0]))
            poss.append(math.hypot(loc[0] - exp[0], loc[1] - exp[1]) if loc else 1e3)
            uv_c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            store.set_prediction(door.persistent_id, uv=uv_c)
            dec = store.reacquire([dict(anchor=dino_np(patch(fr, bb)), uv=uv_c,
                                        world_transform=np.eye(4),
                                        bounds=[1, 2, 0.2])],
                                  t=cid * 0.25, motion_aware=True)[0]
            ids.append(dec["id"] == door.persistent_id)
            if dec["id"] not in (None, door.persistent_id):
                wids += 1
            # appearance drift: how far the ROI moved from the original anchor
            apps.append(float((dino_np(patch(fr, bb)) * CANON_FEAT).sum()))
        summary[aname] = dict(deg=deg_total,
                              existence=float(np.mean(exs)),
                              identity=float(np.mean(ids)), wrong_id=wids,
                              pos_err=float(np.mean(poss)),
                              dino=float(np.mean(dinos)),
                              appearance=float(np.mean(apps)),
                              collapse=float(max(dinos[:2]) /
                                             max(min(dinos[-2:]), 1e-6)))
        s = summary[aname]
        print(f"    exist {s['existence']*100:3.0f}% id {s['identity']*100:3.0f}% "
              f"wID {wids} pos {s['pos_err']:5.1f} | DINO {s['dino']:.3f} "
              f"appearance {s['appearance']:.3f} collapse {s['collapse']:.1f}",
              flush=True)

    print("\n[ro] ===== §41B-3 Rotation (P1, camera fixed) =====")
    print(f"  {'arm':>7s} {'deg':>5s} {'exist':>6s} {'ident':>6s} {'wID':>4s} "
          f"{'pos':>6s} {'DINO':>6s} {'appear':>7s} {'collapse':>9s}")
    for a, s in summary.items():
        print(f"  {a:>7s} {s['deg']:5.0f} {s['existence']*100:5.0f}% "
              f"{s['identity']*100:5.0f}% {s['wrong_id']:4d} {s['pos_err']:6.1f} "
              f"{s['dino']:6.3f} {s['appearance']:7.3f} {s['collapse']:9.1f}")

    print("\n[ro] ===== interpretation =====")
    print("  appearance column: does rotating the latent patch actually change")
    print("  the rendered view?  If it stays ~constant the anchor is a fixed")
    print("  appearance snapshot and view conditioning must be added elsewhere.")
    base = summary.get("R0", {}).get("appearance")
    for a, s in summary.items():
        if base is not None and a != "R0":
            print(f"    {a}: appearance {s['appearance']:.3f} vs R0 {base:.3f} "
                  f"(delta {s['appearance']-base:+.3f})")
    json.dump(summary, open(f"{args.out_dir}/rot.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
