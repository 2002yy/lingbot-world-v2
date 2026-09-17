#!/usr/bin/env python
"""§40C-2: minimal-cost, sustainable, controllable state commit.

Order of work (locked):
    1. continuous probe          (state_response, not a single OPEN/CLOSED threshold)
    2. alpha x K sweep           (find the SMALLEST dose that gives absorption)
    3. commit paths  A vs B      (A: x0 pre-final-forward; B: KV-only)
    4. FOV verification          (door_visible is KNOWN BY CONSTRUCTION here)
    5. stop-commit cross-T50 revisit  (the graduation experiment)

FOV BY CONSTRUCTION (from object_permanence.build_traj):
    frames = [start]*8 + out(T*4) + out[::-1](T*4) + [start]*8
  => the door is visible ONLY at chunks 0,1 and the last 2 chunks.
     Everything in between is out of FOV. This means the earlier §40C-1
     "persistence at chunks 6,7" was measured on a region with NO door in it
     and is INVALID. It is corrected here.

Commit paths:
    A  x0 + alpha*delta  before the KV-writing forward; decode the corrected x0
       -> affects BOTH the picture and the KV
    B  KV-only: the KV-writing forward sees the corrected x0, but the DECODE
       uses the ORIGINAL x0 -> the picture is untouched, only memory is written.
       This separates "display correction" (§40B) from "memory commit".

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python state_commit2.py --scene 04 --seed 42 --mode sweep --horizon 12
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
    """Chunks whose 4 frames all lie in the start-pose blocks (0..7, last 8)."""
    vis = set()
    for cid in range(n_lat):
        f0, f1 = 4 * cid, 4 * cid + 3
        if f1 < 8 or f0 >= frames_n - 8:
            vis.add(cid)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--mode", default="sweep", choices=["sweep", "revisit"])
    ap.add_argument("--path", default="A", choices=["A", "B"])
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--ks", default="1,2,4")
    ap.add_argument("--n_chunks", type=int, default=0)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/commit2")
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
    print("[c2] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lh_ = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads

    traj, ref_chunk, out_chunks = build_traj(scene, args.horizon)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    if args.n_chunks:
        n_lat = min(n_lat, args.n_chunks)
    vis = visible_chunks(frames_n, n_lat)
    d = f"examples/c2_{scene}_H{args.horizon}"
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

    print(f"[c2] H={args.horizon} n_lat={n_lat} ref_chunk={ref_chunk} "
          f"visible chunks = {sorted(vis)}", flush=True)

    # ---------- condition latent + OPEN direction (encoder used then dropped) ----------
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

    ref_img = np.array(img_pil.resize((w, h), Image.BICUBIC))
    open_img = make_open_ref(ref_img, DOOR_BB, SKY_BB)
    t0 = time.perf_counter()
    with torch.no_grad():
        z_ref = pipe.vae.encode([
            TF.to_tensor(Image.fromarray(ref_img)).sub_(0.5).div_(0.5)
            .unsqueeze(0).transpose(0, 1).to(dev)])[0]
        z_open = pipe.vae.encode([
            TF.to_tensor(Image.fromarray(open_img)).sub_(0.5).div_(0.5)
            .unsqueeze(0).transpose(0, 1).to(dev)])[0]
    enc_ms = (time.perf_counter() - t0) * 1000.0
    enc_peak = base_free - vram_free_mb()
    dy0, dy1 = int(DOOR_BB[1] * lat_h), int(DOOR_BB[3] * lat_h)
    dx0, dx1 = int(DOOR_BB[0] * lat_w), int(DOOR_BB[2] * lat_w)
    delta = torch.zeros_like(z_open)
    delta[:, :, dy0:dy1, dx0:dx1] = (z_open - z_ref)[:, :, dy0:dy1, dx0:dx1]
    delta = delta.detach()
    del z_ref, z_open
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[c2] setup: 2 VAE encodes {enc_ms:.0f}ms, peak +{enc_peak:.0f}MiB, "
          f"reclaimed -> free {vram_free_mb():.0f}MiB", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(alpha, K, path):
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, latents, flags = [], [], []
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
            on = alpha > 0 and cid < K
            x0_commit = x0 + alpha * delta if on else x0
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0_commit], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            # path B: KV gets the commit, the DECODE does not
            x0_dec = x0 if (on and path == "B") else x0_commit
            with torch.no_grad():
                fr = tae.decode_video(x0_dec.permute(1, 0, 2, 3).unsqueeze(0),
                                      parallel=False, show_progress_bar=False)
            outs.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                         * 255.0).clip(0, 255).astype(np.uint8))
            latents.append(x0_commit.detach().float().cpu())
            flags.append(bool(on))
        del self_kv, cross_kv
        gc.collect(); torch.cuda.empty_cache()
        return outs, latents, flags

    def measure(frames, latents, flags, c_reg):
        """Continuous per-chunk response + known visibility.

        c_reg MUST be a single fixed constant (the CONTROL's reference
        contrast), otherwise each branch normalises by its own possibly
        committed reference frame and the comparison is meaningless.
        """
        rows = []
        for cid in range(len(frames)):
            f = patch(frames[cid], DOOR_BB)
            sb = surround_brightness_of(frames[cid], DOOR_BB)
            sf = structural_features(f, sb)
            rows.append(dict(
                chunk=cid, visible=cid in vis, commit=flags[cid],
                state_response=abs(sf["contrast"]) / c_reg,
                contrast=sf["contrast"],
                roi_brightness=float(f.mean() / 255.0)))
        return rows

    results = {}

    # ---------- control (no commit) ----------
    print("\n[c2] === control (alpha=0) ===", flush=True)
    A_frames, A_lat, A_flags = run(0.0, 0, args.path)
    _rp = patch(A_frames[ref_chunk], DOOR_BB)
    C_REG = abs(structural_features(
        _rp, surround_brightness_of(A_frames[ref_chunk], DOOR_BB))["contrast"]) + 1e-9
    print(f"[c2] fixed normaliser C_REG = {C_REG:.4f} (control ref chunk {ref_chunk})",
          flush=True)
    A_rows = measure(A_frames, A_lat, A_flags, C_REG)
    results["control"] = A_rows

    # ---------- configs ----------
    alphas = [float(x) for x in args.alphas.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    if args.mode == "sweep":
        configs = [(a, k) for a in alphas for k in ks]
    else:
        configs = [(a, k) for a in alphas for k in ks]
    for alpha, K in configs:
        tag = f"{args.path}_a{alpha}_K{K}"
        print(f"\n[c2] === {tag} ===", flush=True)
        B_frames, B_lat, B_flags = run(alpha, K, args.path)
        B_rows = measure(B_frames, B_lat, B_flags, C_REG)
        # collateral in latent space over the committed chunks
        coll = []
        for cid in range(len(B_lat)):
            dl = (B_lat[cid] - A_lat[cid]).abs()
            door = dl[:, :, dy0:dy1, dx0:dx1]
            out = dl.clone(); out[:, :, dy0:dy1, dx0:dx1] = 0
            n_out = out.numel() - door.numel()
            coll.append(float(out.sum() / max(n_out, 1)))
        results[tag] = dict(rows=B_rows, collateral=coll, alpha=alpha, K=K)
        vis_rows = [r for r in B_rows if r["visible"]]
        rev_rows = [r for r in B_rows if r["visible"] and r["chunk"] > 1]
        peak = max((r["state_response"] for r in vis_rows), default=0)
        base_peak = max((r["state_response"] for r in A_rows if r["visible"]),
                        default=0)
        rev = max((r["state_response"] for r in rev_rows), default=0)
        base_rev = max((r["state_response"] for r in A_rows
                        if r["visible"] and r["chunk"] > 1), default=0)
        print(f"    visible-chunk peak response = {peak:.2f} (control {base_peak:.2f})"
              f"   revisit = {rev:.2f} (control {base_rev:.2f})"
              f"   collateral = {np.mean(coll):.6f}", flush=True)

    # ---------- report ----------
    print("\n[c2] ===== §40C-2 per-chunk (visible = door in FOV) =====")
    print(f"  {'chunk':>5s} {'vis':>4s} {'ctrl':>7s} "
          + " ".join(f"{t[:14]:>14s}" for t in results if t != "control"))
    n = len(A_rows)
    for cid in range(n):
        line = f"  {cid:5d} {'Y' if cid in vis else '.':>4s} "
        line += f"{A_rows[cid]['state_response']:7.2f} "
        for t, v in results.items():
            if t == "control":
                continue
            line += f"{v['rows'][cid]['state_response']:14.2f} "
        print(line)

    print("\n[c2] ===== summary =====")
    print(f"  {'config':>16s} {'peak_vis':>9s} {'revisit':>8s} "
          f"{'ctrl_rev':>9s} {'gain':>6s} {'collat':>9s}")
    for t, v in results.items():
        if t == "control":
            continue
        vis_rows = [r for r in v["rows"] if r["visible"]]
        rev_rows = [r for r in v["rows"] if r["visible"] and r["chunk"] > 1]
        peak = max((r["state_response"] for r in vis_rows), default=0)
        rev = max((r["state_response"] for r in rev_rows), default=0)
        ctrl_rev = max((r["state_response"] for r in A_rows
                        if r["visible"] and r["chunk"] > 1), default=0)
        print(f"  {t:>16s} {peak:9.2f} {rev:8.2f} {ctrl_rev:9.2f} "
              f"{rev/max(ctrl_rev,1e-6):6.1f} {np.mean(v['collateral']):9.6f}")

    json.dump(results, open(f"{args.out_dir}/commit2_{args.mode}_{args.path}.json", "w"),
              indent=1, default=float)
    np.save(f"{args.out_dir}/ctrl_frames.npy", np.stack(A_frames))


if __name__ == "__main__":
    main()
