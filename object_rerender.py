#!/usr/bin/env python
"""§40D: Object Re-render / resurrection.

§40C proved the commit mechanism works (and §40C-B proved KV-only memory
write), but the revisit fails because the BASE MODEL does not re-render the
object at all: the door ROI response collapses to ~0.04 and stays there.
Without a host instance there is nothing for OPEN/CLOSED to attach to.

So §40D attacks existence/identity FIRST, not state.

    D0  no anchor          the current baseline: leave FOV -> return -> absent
    D1  identity anchor    inject the REGISTERED LOCAL LATENT PATCH of the door
                           ROI at the revisit chunks (no state change at all)
    D2  D1 + state commit  only if D1 works

The anchor is cheap: the registration run already produces x0, so we simply
crop the door ROI's latent tokens and store them. No extra VAE encode.

    visual_anchor = { latent_patch [16,1,dh,dw], latent_roi, registration_pose }

Injection (before the KV-writing forward, same slot as the §40C commit):
    x0[door_roi] = blend * anchor_latent + (1 - blend) * x0[door_roi]

Metrics (per the §40D spec):
    Object Re-render Rate   is the object present where it should be
    Identity Match          DINO similarity to the registration patch
    Pose Error              expected ROI (world_transform + camera pose) vs
                            the best-match location in a local window
    State Compliance        ONLY computed when the object is confirmed present

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python object_rerender.py --scene 04 --seed 42 --horizon 40 --blends 0.5,1.0
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange
from object_permanence import build_traj, patch
from door_state_probe import make_open_ref, surround_brightness_of, \
    structural_features

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
DOOR_BB = (0.70, 0.50, 0.84, 0.66)
SKY_BB = (0.05, 0.02, 0.35, 0.16)


def vram_free_mb():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**20


def visible_chunks(frames_n, n_lat):
    vis = set()
    for cid in range(n_lat):
        f0, f1 = 4 * cid, 4 * cid + 3
        if f1 < 8 or f0 >= frames_n - 8:
            vis.add(cid)
    return vis


def best_match_loc(ref_patch, img, bb, dino_np, search=0.10, steps=15):
    """Local search around the expected ROI; returns (sim, (bx,by))."""
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
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--blends", default="0.5,1.0")
    ap.add_argument("--n_chunks", type=int, default=0)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/rerender")
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
    print("[rd] pipe + TAE built", flush=True)
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

    traj, ref_chunk, out_chunks = build_traj(scene, args.horizon)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    if args.n_chunks:
        n_lat = min(n_lat, args.n_chunks)
    vis = visible_chunks(frames_n, n_lat)
    d = f"examples/rd_{scene}_H{args.horizon}"
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

    revisit = sorted([c for c in vis if c > 1])
    print(f"[rd] H={args.horizon} n_lat={n_lat} ref={ref_chunk} "
          f"visible={sorted(vis)} revisit={revisit}", flush=True)

    base_free = vram_free_mb()
    with torch.no_grad():
        y = pipe.vae.encode([torch.concat([
            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                            mode='bicubic').transpose(0, 1),
            torch.zeros(3, frames_n - 1, h, w)], dim=1).to(dev)])[0]
    msk = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y]).detach()
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[rd] setup done, driver free {vram_free_mb():.0f}MiB", flush=True)

    dy0, dy1 = int(DOOR_BB[1] * lat_h), int(DOOR_BB[3] * lat_h)
    dx0, dx1 = int(DOOR_BB[0] * lat_w), int(DOOR_BB[2] * lat_w)
    print(f"[rd] door latent ROI = [{dy0}:{dy1}, {dx0}:{dx1}]", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(anchor=None, blend=0.0):
        """anchor: registered latent patch [16,1,dh,dw] or None."""
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, latents = [], []
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
            # ---- §40D identity anchor injection (revisit chunks only) ----
            if anchor is not None and blend > 0 and cid in revisit:
                x0 = x0.clone()
                x0[:, :, dy0:dy1, dx0:dx1] = (
                    blend * anchor[:, :, :dy1 - dy0, :dx1 - dx0]
                    + (1 - blend) * x0[:, :, dy0:dy1, dx0:dx1])
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

    # ---------- D0: no anchor, capture the registered latent patch ----------
    print("\n[rd] === D0 (no anchor) ===", flush=True)
    D0_frames, D0_lat = run(anchor=None, blend=0.0)
    anchor = D0_lat[ref_chunk][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    # D0_lat[cid] is [16,1,lat_h,lat_w] -> anchor is [16,1,dh,dw] on dev
    print(f"[rd] registered latent anchor from chunk {ref_chunk}: "
          f"{tuple(anchor.shape)} mean|a|={float(anchor.abs().mean()):.4f}", flush=True)

    results = {"D0": dict(frames=D0_frames, latents=D0_lat)}
    for bl in [float(x) for x in args.blends.split(",")]:
        print(f"\n[rd] === D1 (identity anchor, blend={bl}) ===", flush=True)
        f, l = run(anchor=anchor, blend=bl)
        results[f"D1_b{bl}"] = dict(frames=f, latents=l)

    # ---------- metrics ----------
    ref_p = patch(D0_frames[ref_chunk], DOOR_BB)
    c_reg = abs(structural_features(
        ref_p, surround_brightness_of(D0_frames[ref_chunk], DOOR_BB))["contrast"]) + 1e-9

    print("\n[rd] ===== §40D Object Re-render metrics =====")
    print(f"  {'tag':>10s} {'chunk':>5s} {'vis':>4s} {'DINO':>6s} "
          f"{'resp':>6s} {'pose_err':>9s} {'present':>8s}")
    summary = {}
    for tag, v in results.items():
        rows = []
        for cid in revisit:
            f = v["frames"][cid]
            sim, loc = best_match_loc(ref_p, f, DOOR_BB, dino_np)
            sb = surround_brightness_of(f, DOOR_BB)
            resp = abs(structural_features(patch(f, DOOR_BB), sb)["contrast"]) / c_reg
            exp_x = int(DOOR_BB[0] * f.shape[1])
            exp_y = int(DOOR_BB[1] * f.shape[0])
            pe = (math.hypot(loc[0] - exp_x, loc[1] - exp_y)
                  if loc else float("nan"))
            present = bool(sim > 0.5 and resp > 0.35)
            rows.append(dict(chunk=cid, dino=sim, resp=resp, pose_err=pe,
                             present=present))
            print(f"  {tag:>10s} {cid:5d} {'Y':>4s} {sim:6.3f} {resp:6.2f} "
                  f"{pe:9.1f} {'YES' if present else 'no':>8s}")
        summary[tag] = rows

    print("\n[rd] ===== summary =====")
    print(f"  {'tag':>10s} {'rerender_rate':>13s} {'mean_DINO':>10s} "
          f"{'mean_resp':>10s} {'mean_pose_err':>14s}")
    for tag, rows in summary.items():
        rr = np.mean([r["present"] for r in rows])
        print(f"  {tag:>10s} {rr*100:12.0f}% {np.mean([r['dino'] for r in rows]):10.3f} "
              f"{np.mean([r['resp'] for r in rows]):10.2f} "
              f"{np.nanmean([r['pose_err'] for r in rows]):14.1f}")

    json.dump({k: v for k, v in summary.items()},
              open(f"{args.out_dir}/rerender.json", "w"), indent=1, default=float)
    for tag, v in results.items():
        np.save(f"{args.out_dir}/{tag}_frames.npy", np.stack(v["frames"]))
    torch.save(anchor, f"{args.out_dir}/anchor.pt")


if __name__ == "__main__":
    main()
