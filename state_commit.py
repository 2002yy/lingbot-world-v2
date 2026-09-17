#!/usr/bin/env python
"""§40C-1: does a corrected-frame feedback have CAUSAL power?

Decisive paired experiment, same prefix / same seed:

    Branch A (control)  generate chunk 0 normally, then keep going
    Branch B (commit)   at --commit_chunk, correct the latent toward OPEN
                        before the KV-writing forward, then keep going

Pipeline order is  denoise -> final forward (writes KV) -> decode .
Correcting x0 *before* the final forward puts the corrected content into BOTH
the decoded frame and the KV cache that conditions every later chunk. So NO
VAE encode is needed inside the loop -- the correction is a latent-space
addition of a direction computed ONCE at setup:

    delta = vae.encode(open_frame) - vae.encode(ref_frame)   (door ROI only)

The VAE encoder is used twice at setup, then dropped immediately (§17/§36C
discipline: never leave the encoder resident).

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python state_commit.py --scene 04 --seed 42 --n_chunks 6
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
from door_state_probe import DoorStateProbe, surround_brightness_of, make_open_ref

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
DOOR_BB = (0.70, 0.50, 0.84, 0.66)
SKY_BB = (0.05, 0.02, 0.35, 0.16)


def vram_free_mb():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--commit_chunk", type=int, default=0)
    ap.add_argument("--commit_until", type=int, default=None,
                    help="commit on every chunk in [commit_chunk, commit_until)")
    ap.add_argument("--n_chunks", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/commit")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    cu = args.commit_until if args.commit_until is not None else args.commit_chunk + 1

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[cm] pipe + TAE built", flush=True)
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
    n_lat = min((frames_n - 1) // 4 + 1, args.n_chunks)
    d = f"examples/cm_{scene}_H{args.horizon}"
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

    # ---------- condition latent (needs the encoder) ----------
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

    # ---------- the OPEN direction (2 more encodes, then drop) ----------
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
    # NOTE: vae.encode returns [C, T, H, W] -> the region is on dims 2 and 3
    delta = torch.zeros_like(z_open)
    delta[:, :, dy0:dy1, dx0:dx1] = (z_open - z_ref)[:, :, dy0:dy1, dx0:dx1]
    delta = delta.detach()
    dnorm = float(delta[:, :, dy0:dy1, dx0:dx1].abs().mean())
    print(f"[cm] setup: 2 VAE encodes in {enc_ms:.0f}ms, peak +{enc_peak:.0f}MiB, "
          f"door latent {dy1-dy0}x{dx1-dx0}, mean|delta|_door={dnorm:.4f}", flush=True)
    del z_ref, z_open
    pipe.vae = None                      # §17/§36C discipline
    gc.collect(); torch.cuda.empty_cache()
    free_after = vram_free_mb()
    print(f"[cm] VAE dropped: driver free {base_free:.0f} -> {free_after:.0f}MiB "
          f"(encode peak was +{enc_peak:.0f}MiB; fully reclaimed = "
          f"{free_after >= base_free - 64})", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run_branch(commit):
        """Returns the list of decoded frames (uint8 HWC RGB)."""
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
            # ---- §40C commit: correct BEFORE the KV-writing forward ----
            if commit and args.commit_chunk <= cid < cu:
                x0 = x0 + args.alpha * delta
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

    # ---------- run both branches ----------
    print("\n[cm] === branch A (control, no commit) ===", flush=True)
    tA = time.perf_counter()
    A_frames, A_lat = run_branch(commit=False)
    tA = time.perf_counter() - tA
    print(f"[cm] === branch B (commit at chunk {args.commit_chunk}, "
          f"alpha={args.alpha}) ===", flush=True)
    tB = time.perf_counter()
    B_frames, B_lat = run_branch(commit=True)
    tB = time.perf_counter() - tB
    print(f"[cm] A {tA:.1f}s  B {tB:.1f}s", flush=True)

    # ---------- measure with the §40A-2 probe ----------
    # calibrate on the model's own pre-commit rendering (chunk 0, branch A)
    cal_patch = patch(A_frames[0], DOOR_BB)
    cal_sb = surround_brightness_of(A_frames[0], DOOR_BB)
    probe = DoorStateProbe().calibrate([cal_patch], [cal_sb])

    print("\n[cm] ===== §40C-1 State Compliance after commit =====")
    print(f"  {'chunk':>5s} {'A(control)':>22s} {'B(commit)':>22s}")
    rows = []
    for cid in range(n_lat):
        ra = probe.scores(patch(A_frames[cid], DOOR_BB),
                          surround_brightness_of(A_frames[cid], DOOR_BB))
        rb = probe.scores(patch(B_frames[cid], DOOR_BB),
                          surround_brightness_of(B_frames[cid], DOOR_BB))
        rows.append(dict(chunk=cid, A=ra["decision"], A_margin=ra["margin"],
                         A_ratio=ra["ratio"],
                         B=rb["decision"], B_margin=rb["margin"],
                         B_ratio=rb["ratio"]))
        print(f"  {cid:5d} {ra['decision']:>10s} (r={ra['ratio']:.2f}) "
              f"{rb['decision']:>10s} (r={rb['ratio']:.2f})")

    # latent-space divergence outside the door region (collateral)
    cid = args.commit_chunk
    if cid < len(A_lat) and cid < len(B_lat):
        dl = (B_lat[cid] - A_lat[cid]).abs()
        door_region = dl[:, :, dy0:dy1, dx0:dx1].mean()
        outside = dl.clone()
        outside[:, :, dy0:dy1, dx0:dx1] = 0
        out_mean = outside.sum() / max(outside.numel() - dl[:, :, dy0:dy1, dx0:dx1].numel(), 1)
        print(f"\n  latent delta at commit chunk: door={door_region:.4f} "
              f"outside={out_mean:.4f} ratio={door_region/max(out_mean,1e-9):.1f}x")

    post = [r for r in rows if r["chunk"] > args.commit_chunk]
    if post:
        comp_A = np.mean([r["A"] == "OPEN" for r in post])
        comp_B = np.mean([r["B"] == "OPEN" for r in post])
        print(f"\n  §40C-1 compliance over chunks after commit:")
        print(f"    A control : {comp_A*100:.0f}%  (expect ~0)")
        print(f"    B commit  : {comp_B*100:.0f}%  (expect > 0 if feedback has causal power)")
        tta = next((r["chunk"] for r in post if r["B"] == "OPEN"), None)
        print(f"    time-to-assimilate = {tta if tta is not None else 'never'} chunks")
        persist = 0
        for r in reversed(post):
            if r["B"] == "OPEN":
                persist += 1
            else:
                break
        print(f"    persistence (trailing OPEN run) = {persist} chunks "
              f"({persist*0.25:.2f}s)")

    json.dump(dict(rows=rows, encode_ms=enc_ms, encode_peak_mib=enc_peak,
                   alpha=args.alpha, commit_chunk=args.commit_chunk,
                   n_chunks=n_lat),
              open(f"{args.out_dir}/commit.json", "w"), indent=1)
    np.save(f"{args.out_dir}/A_frames.npy", np.stack(A_frames))
    np.save(f"{args.out_dir}/B_frames.npy", np.stack(B_frames))


if __name__ == "__main__":
    main()
