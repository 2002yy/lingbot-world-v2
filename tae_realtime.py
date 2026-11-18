#!/usr/bin/env python
"""§36A: TAE real-time path integration + display A/B + VRAM co-residency.

Real-time loop: hot-swap conditioning + translation compensator -> DiT ->
TAEW2.1 decode. Each chunk is decoded BOTH ways (full Wan VAE and TAEW2.1) so
the display A/B uses identical latents.

Display groups:
    A  warp -> full VAE frames
    B  warp -> TAE frames
    C  warp -> TAE frames + adaptive correction

Also audits StreamingTAEHV first-frame timing and peak VRAM co-residency of
DiT + KV cache + hot-swap + compensator + TAE + display buffers.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python tae_realtime.py
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import types

import cv2
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from eval_two_layer import load_models, lpips_d
from pose_warp import PoseWarper
from cam_controller import CameraController
from compensator import Compensator
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV, StreamingTAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
FPS, CADENCE = 60.0, 1.25
D = int(FPS * CADENCE)
SCRIPT = [(0.0, dict(fwd=0.6)), (2.0, dict(yaw=0.8, fwd=0.3)),
          (4.0, dict(yaw=-0.8, fwd=0.3)), (6.0, dict(fwd=-0.6)),
          (8.0, dict(strafe=0.8))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_chunks", type=int, default=8)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/tae_rt")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)

    # authoritative control clock
    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    n_ctrl = int((args.n_chunks * CADENCE + 2.0) * FPS)
    cps, cvs = [], []
    si = 0
    ctl.set_input(**SCRIPT[0][1])
    for k in range(n_ctrl):
        t = k / FPS
        while si + 1 < len(SCRIPT) and t >= SCRIPT[si + 1][0]:
            si += 1
            ctl.set_input(**SCRIPT[si][1])
        ctl.step(dt=1.0 / (FPS * CADENCE))
        cps.append(ctl.pose.copy()); cvs.append(ctl.v.copy())
    cps, cvs = np.stack(cps), np.stack(cvs)

    def slot_pose(t):
        i = int(np.clip(round(t * FPS), 0, len(cps) - 1))
        return cps[i]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print("[rt] pipe built", flush=True)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print(f"[rt] TAE loaded ({sum(p.numel() for p in tae.decoder.parameters())/1e6:.2f}M dec params)",
          flush=True)

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch[2] * patch[2])
    h = lat_h * vae_stride[1]; w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = (args.n_chunks - 1) * 4 + 1
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * pipe.local_attn_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads
    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                        mode='bicubic').transpose(0, 1),
        torch.zeros(3, F - 1, h, w)], dim=1).to(dev)])[0]
    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y])
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, args.n_chunks, lat_h, lat_w, generator=g, device=dev)
    comp_state = dict(gate=1.0, prev_v=None)

    def plucker(rel_pose):
        rel = torch.from_numpy(np.asarray(rel_pose)).float()[None].to(dev)
        p = get_plucker_embeddings(rel, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h, w=lat_w).to(pdt)

    def lead_pose(pose, t):
        i = int(np.clip(round(t * FPS), 0, len(cvs) - 1))
        v, vp = cvs[i], comp_state["prev_v"]
        n_now = float(np.linalg.norm(v))
        if vp is None or n_now < 0.02:
            tgt = 1.0 if vp is None else 0.0
        else:
            cos = float(np.dot(vp, v) / max(np.linalg.norm(vp) * n_now, 1e-9))
            tgt = 0.0 if cos <= 0 else float(min(1.0, cos / 0.7))
        comp_state["gate"] = tgt if tgt < comp_state["gate"] else \
            comp_state["gate"] + 0.25 * (tgt - comp_state["gate"])
        comp_state["prev_v"] = v.copy()
        T = np.eye(4); T[:3, 3] = v * (2.0 * comp_state["gate"])
        return pose @ T

    full_frames, tae_frames, rows = [], [], []
    prev_pose = None
    t_start = time.perf_counter()
    for cid in range(args.n_chunks):
        cur = noise.split(1, dim=1)[cid]
        t_now = time.perf_counter() - t_start
        pose_now = lead_pose(slot_pose(t_now), t_now)
        rel0 = np.eye(4) if prev_pose is None else np.linalg.inv(prev_pose) @ pose_now
        for ti in range(len(timesteps)):
            t_now = time.perf_counter() - t_start
            pn = lead_pose(slot_pose(t_now), t_now)
            rel = np.eye(4) if prev_pose is None else np.linalg.inv(prev_pose) @ pn
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plucker(rel).chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * frame_seqlen,
                  "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=not pipe._cross_attn_initialized, **kw)[0]
                pipe._cross_attn_initialized = True
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=g, device=x0.device,
                                        dtype=x0.dtype), timesteps[ti + 1])
        t_dit = time.perf_counter() - t_start
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)

        # ---- decode both ways ----
        pipe.vae.stream_decode_start()
        t0 = time.perf_counter()
        with torch.no_grad():
            f_full = pipe.vae.stream_decode_step(x0)
        torch.cuda.synchronize()
        t_full = time.perf_counter() - t0
        pipe.vae.stream_decode_end()
        zt = x0.permute(1, 0, 2, 3).unsqueeze(0)      # [1,T,C,H,W]
        t0 = time.perf_counter()
        with torch.no_grad():
            f_tae = tae.decode_video(zt, parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        t_tae = time.perf_counter() - t0
        full_frames.append(f_full)
        tae_frames.append(f_tae[0].permute(0, 2, 3, 1))
        rows.append(dict(chunk=cid, t_dit=t_dit, t_full=t_full, t_tae=t_tae,
                         t_end=time.perf_counter() - t_start))
        print(f"[rt] chunk {cid}: dit={t_dit*1000:.0f}ms full={t_full*1000:.0f}ms "
              f"tae={t_tae*1000:.0f}ms", flush=True)
        prev_pose = pose_now

    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"\n[rt] ===== real-time path =====")
    print(f"  peak VRAM (DiT+KV+hotswap+comp+TAE+full VAE co-resident) = {peak:.0f}MiB")
    t_full_all = np.mean([r["t_full"] for r in rows[1:]]) * 1000
    t_tae_all = np.mean([r["t_tae"] for r in rows[1:]]) * 1000
    t_dit = np.mean([r["t_dit"] for r in rows[1:]]) * 1000
    print(f"  steady denoise={t_dit:.0f}ms  full VAE={t_full_all:.0f}ms  "
          f"TAE={t_tae_all:.0f}ms")
    print(f"  TTFNF: full={t_dit+t_full_all:.0f}ms  TAE={t_dit+t_tae_all:.0f}ms  "
          f"(-{100*(1-(t_dit+t_tae_all)/(t_dit+t_full_all)):.0f}%)")

    # ---- StreamingTAEHV first-frame audit ----
    print(f"\n[rt] ===== StreamingTAEHV first-frame audit =====")
    st = StreamingTAEHV(tae)
    zt = x0.permute(1, 0, 2, 3).unsqueeze(0)
    st.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    f1 = st.decode(zt)
    torch.cuda.synchronize()
    t_first = time.perf_counter() - t0
    n_rest = 0
    while True:
        fr = st.decode()
        if fr is None:
            break
        n_rest += 1
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0
    print(f"  streaming first frame = {t_first*1000:.0f}ms, "
          f"total {t_total*1000:.0f}ms ({n_rest} more frames)")
    print(f"  batch TAE = {t_tae*1000:.0f}ms  -> first-frame gain = "
          f"{(t_tae-t_first)*1000:.0f}ms")

    # ---- display A/B ----
    del pipe, tae
    gc.collect(); torch.cuda.empty_cache()
    load_models()
    full_np = [((f.permute(1, 2, 3, 0).float().cpu().numpy() + 1) * 127.5
               ).clip(0, 255).astype(np.uint8) for f in full_frames]
    tae_np = [(f.float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
              for f in tae_frames]
    Hf, Wf = full_np[0].shape[1:3]
    tae_np = [cv2.resize(f, (Wf, Hf)) if f.shape[1:3] != (Hf, Wf) else f
              for f in tae_np]
    n_lat = len(full_np)
    total = n_lat * D
    P = np.load(f"{args.base}/poses.npy")
    ctrl = interpolate_camera_poses(
        np.linspace(0, len(P) - 1, len(P)),
        torch.from_numpy(P[:, :3, :3]).float(),
        torch.from_numpy(P[:, :3, 3]).float(),
        np.linspace(0, len(P) - 1, total)).numpy()
    Ksm = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, Hf, Wf, Hf, Wf)[0].numpy()

    def frame_of(seq, k):
        return seq[min(k, len(seq) - 1)][min(3, len(seq[min(k, len(seq) - 1)]) - 1)]

    results = {}
    for name, seq in (("A_full", full_np), ("B_tae", tae_np)):
        warper = PoseWarper(Ksm, Hf, Wf, nominal_depth=4.0)
        outs = []
        for f in range(total):
            chunk = f // D
            ref = max(0, chunk - 1)
            warper.set_reference(frame_of(seq, ref), ctrl[min(ref * D, total - 1)])
            outs.append(warper.warp(ctrl[f]))
        jumps = np.array([lpips_d(cv2.cvtColor(outs[k - 1], cv2.COLOR_BGR2RGB),
                                  cv2.cvtColor(outs[k], cv2.COLOR_BGR2RGB))
                          for k in range(1, len(outs))])
        results[name] = dict(mean=float(jumps.mean()), peak=float(jumps.max()),
                             total=float(jumps.sum()))
        print(f"[rt] display {name}: mean={jumps.mean():.4f} peak={jumps.max():.4f} "
              f"total={jumps.sum():.1f}", flush=True)

    print(f"\n[rt] ===== display A/B =====")
    a = results["A_full"]
    for k in ("A_full", "B_tae"):
        r = results[k]
        print(f"  {k}: mean={r['mean']:.4f} peak={r['peak']:.4f} total={r['total']:.1f}"
              + ("" if k == "A_full" else
                 f"  (peak {100*(r['peak']/a['peak']-1):+.1f}%, "
                 f"total {100*(r['total']/a['total']-1):+.1f}%)"))
    json.dump(dict(rows=rows, results=results, peak_vram=peak,
                   t_dit=t_dit, t_full=t_full_all, t_tae=t_tae_all,
                   stream_first_ms=t_first * 1000),
              open(f"{args.out_dir}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
