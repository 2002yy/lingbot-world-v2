#!/usr/bin/env python
"""§41A: Anchor Relocation -- can one identity anchor be moved to a NEW ROI?

The door anchor so far has always been injected back at its ORIGINAL latent
ROI. A dynamic object (box_03 pushed from A to B) needs the same anchor to
work at a DIFFERENT ROI. That has never been proven.

Arms (all at the revisit, same camera pose, same canonical anchor):
    D0        model's own output (no injection)
    T1_raw    inject the raw anchor at the SHIFTED ROI
    T2_fg     inject only the FOREGROUND part (anchor minus its own local
              mean), i.e. structure without the absolute background level

Shifts: 0 (control), small (2 latent), medium (5), large (10)
  (1 latent unit = 8 px, so 16 / 40 / 80 px)

Metrics:
    Existence at the NEW roi      DINO(ref object, new-roi patch)
    Identity at the NEW roi       §39 reacquire -> must be door_01
    Position error                requested vs observed object centre
    Background collateral         change at the OLD roi vs D0
    Appearance adaptation         how much the pasted object is re-rendered

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python relocate.py --scene 04 --seed 42 --out_chunks 40 --tail 10
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
SHIFTS = {"s0": (0, 0), "s2": (2, 0), "s5": (5, 0), "s10": (10, 0)}  # (dx,dy) latent


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


def best_match_loc(ref_patch, img, bb, dino_np, search=0.10, steps=15):
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
    ap.add_argument("--out_chunks", type=int, default=40)
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--arms", default="T1_raw_s2,T1_raw_s5,T1_raw_s10,T2_fg_s5,T2_fg_s10")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/reloc")
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
    print("[rl] pipe + TAE built", flush=True)
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
    d = f"examples/rl_{scene}_O{args.out_chunks}_T{args.tail}"
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
    print(f"[rl] n_lat={n_lat} ref={ref_chunk} revisit={revisit} "
          f"observe={observe[0]}..{observe[-1]}", flush=True)

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
    print(f"[rl] setup done, driver free {vram_free_mb():.0f}MiB, "
          f"door latent ROI=[{dy0}:{dy1},{dx0}:{dx1}]", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(y_cond, anchor=None, shift=(0, 0), fg_only=False, max_chunks=None):
        """anchor: [16,1,dh,dw]; injected at the SHIFTED ROI on revisit."""
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, latents = [], []
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
            if anchor is not None and cid in revisit:
                sx, sy = shift
                ty0, ty1 = dy0 + sy, dy1 + sy
                tx0, tx1 = dx0 + sx, dx1 + sx
                if 0 <= ty0 and ty1 <= lat_h and 0 <= tx0 and tx1 <= lat_w:
                    a = anchor
                    if fg_only:
                        a = anchor - anchor.mean(dim=(2, 3), keepdim=True)
                    x0[:, :, ty0:ty1, tx0:tx1] = a
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            with torch.no_grad():
                fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                      parallel=False, show_progress_bar=False)
            outs.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                         * 255.0).clip(0, 255).astype(np.uint8))
            latents.append(x0.detach().float().cpu())
        del self_kv, cross_kv
        gc.collect(); torch.cuda.empty_cache()
        return outs, latents

    # ---------- D0 + anchors ----------
    print("\n[rl] === D0 (no injection) ===", flush=True)
    D0_frames, D0_lat = run(y, None)
    ref_frame = D0_frames[ref_chunk]
    ref_door = patch(ref_frame, DOOR_BB)
    z_closed = D0_lat[ref_chunk][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    print(f"\n[rl] === canonicalisation ({args.canon_chunks} chunks on R_open) ===",
          flush=True)
    canon_frames, canon_lat = run(y_open, None, max_chunks=args.canon_chunks)
    ci = min(2, len(canon_lat) - 1)
    z_open = canon_lat[ci][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    canon_door = patch(canon_frames[ci], DOOR_BB)

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
    store.set_anchor_for_state(door.persistent_id, "closed",
                               latent=z_closed.cpu().numpy(), feature=dino_np(ref_door))
    store.set_anchor_for_state(door.persistent_id, "open",
                               latent=z_open.cpu().numpy(), feature=dino_np(canon_door))
    store.set_state(door.persistent_id, open=True)

    def rgb_bb(shift):
        sx, sy = shift
        return ((dx0 + sx) * vae_stride[2] / w, (dy0 + sy) * vae_stride[1] / h,
                (dx1 + sx) * vae_stride[2] / w, (dy1 + sy) * vae_stride[1] / h)

    results = {"D0": D0_frames}
    for arm in args.arms.split(","):
        # arm format: T1_raw_s5 / T2_fg_s10
        parts = arm.split("_")
        kind = parts[1]
        skey = parts[-1]
        shift = SHIFTS[skey]
        fg = (kind == "fg")
        print(f"\n[rl] === {arm}: shift={shift} latent, fg_only={fg} ===", flush=True)
        f, _ = run(y, z_open, shift=shift, fg_only=fg)
        results[arm] = f

    # ---------- metrics ----------
    print("\n[rl] ===== §41A Anchor Relocation =====")
    print(f"  {'arm':>11s} {'shift_px':>8s} {'new_DINO':>8s} {'new_resp':>8s} "
          f"{'old_DINO':>8s} {'pos_err':>7s} {'bg_collat':>9s} {'id':>4s}")
    summary = {}
    for arm, frames in results.items():
        if arm == "D0":
            shift, fg = (0, 0), False
        else:
            parts = arm.split("_")
            fg = (parts[1] == "fg")
            shift = SHIFTS[parts[-1]]
        newbb = rgb_bb(shift)
        oldbb = DOOR_BB
        rows = []
        for cid in revisit + observe:
            f = frames[cid]
            new_patch = patch(f, newbb)
            old_patch = patch(f, oldbb)
            new_sim = float((dino_np(new_patch) * dino_np(canon_door)).sum())
            old_sim = float((dino_np(old_patch) * dino_np(canon_door)).sum())
            sb = surround_brightness_of(f, newbb)
            new_resp = abs(structural_features(new_patch, sb)["contrast"]) / \
                (abs(structural_features(canon_door, surround_brightness_of(
                    canon_frames[ci], DOOR_BB))["contrast"]) + 1e-9)
            sim, loc = best_match_loc(canon_door, f, newbb, dino_np)
            exp = (int(newbb[0] * f.shape[1]), int(newbb[1] * f.shape[0]))
            pe = math.hypot(loc[0] - exp[0], loc[1] - exp[1]) if loc else 1e3
            # §39 identity at the NEW location
            cand = dict(anchor=dino_np(new_patch),
                        uv=((newbb[0] + newbb[2]) / 2, (newbb[1] + newbb[3]) / 2),
                        world_transform=np.eye(4), bounds=[1, 2, 0.2])
            dec = store.reacquire([cand], t=cid * 0.25)[0]
            # background collateral: change vs D0 outside the NEW roi
            if arm != "D0":
                g = results["D0"][cid]
                diff = np.abs(f.astype(float) - g.astype(float)).mean(-1)
                Hf, Wf = diff.shape
                m = np.ones((Hf, Wf), bool)
                m[int(newbb[1] * Hf):int(newbb[3] * Hf),
                  int(newbb[0] * Wf):int(newbb[2] * Wf)] = False
                bg = float(diff[m].mean())
            else:
                bg = 0.0
            rows.append(dict(chunk=cid, new_sim=new_sim, old_sim=old_sim,
                             new_resp=new_resp, pose=pe, bg=bg,
                             match_id=dec["id"], proto=dec.get("matched_proto")))
        rv = [r for r in rows if r["chunk"] in revisit]
        summary[arm] = dict(
            shift_px=shift[0] * vae_stride[2], fg=fg,
            new_dino=float(np.mean([r["new_sim"] for r in rv])),
            new_resp=float(np.mean([r["new_resp"] for r in rv])),
            old_dino=float(np.mean([r["old_sim"] for r in rv])),
            pos_err=float(np.mean([r["pose"] for r in rv])),
            bg=float(np.mean([r["bg"] for r in rows])),
            identity=float(np.mean([r["match_id"] == door.persistent_id for r in rv])),
            proto=[r["proto"] for r in rv])
        s = summary[arm]
        print(f"  {arm:>11s} {s['shift_px']:8d} {s['new_dino']:8.3f} "
              f"{s['new_resp']:8.2f} {s['old_dino']:8.3f} {s['pos_err']:7.1f} "
              f"{s['bg']:9.2f} {s['identity']*100:3.0f}%")

    print("\n[rl] ===== §41A GATE =====")
    print("  relocation works if: new_DINO high, new_resp high, old_DINO LOW")
    print("  (the object moved, the old spot is free), identity=door_01, bg low")
    for arm, s in summary.items():
        if arm == "D0":
            continue
        ok = (s["new_dino"] > 0.6 and s["new_resp"] > 1.0
              and s["old_dino"] < 0.55 and s["identity"] >= 0.9)
        print(f"    {arm}: new_DINO {s['new_dino']:.3f} new_resp {s['new_resp']:.2f} "
              f"old_DINO {s['old_dino']:.3f} identity {s['identity']*100:.0f}% "
              f"bg {s['bg']:.2f} -> {'PASS' if ok else 'FAIL'}")

    json.dump(summary, open(f"{args.out_dir}/reloc.json", "w"), indent=1, default=float)
    for arm, frames in results.items():
        np.save(f"{args.out_dir}/{arm}_frames.npy", np.stack(frames))


if __name__ == "__main__":
    main()
