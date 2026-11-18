#!/usr/bin/env python
"""§33 Step-boundary Camera Hot-Swap.

A shared CameraConditionSlot holds the latest authoritative conditioned pose.
At every denoise-step boundary the loop snapshots the slot and rebuilds the
Plucker embedding, so a control change arriving mid-chunk is absorbed by the
CURRENT chunk instead of waiting for the next one.

Condition is FIXED within a denoise step (no half-step mixing).

Four modes (A/B/C/D) to separate the two orthogonal mechanisms:
    A baseline      chunk-start condition, no compensator
    B comp only     chunk-start condition, translation compensator
    C hot-swap only step-boundary condition, no compensator
    D full          step-boundary condition, translation compensator

Latency is measured from REAL timestamps:
    causal_admission_latency = t_first_step_consuming_new_input - t_input
    control_to_real          = t_chunk_visible_after_input   - t_input
    effective_control_age    = t_chunk_visible - t_condition_timestamp

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python hotswap_loop.py --mode D
"""
import argparse
import gc
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import save_video
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from cam_controller import CameraController
from compensator import Compensator

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
CTRL_HZ = 60.0

# scripted control timeline: (t_seconds, inputs)
SCRIPT = [
    (0.0, dict(fwd=0.6)),
    (2.0, dict(yaw=0.8, fwd=0.3)),
    (4.0, dict(yaw=-0.8, fwd=0.3)),
    (6.0, dict(fwd=-0.6)),
    (8.0, dict(strafe=0.8)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="D", choices=list("ABCD"))
    ap.add_argument("--n_chunks", type=int, default=10)
    ap.add_argument("--out_dir", default="output/hotswap")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    use_swap = args.mode in ("C", "D")
    use_comp = args.mode in ("B", "D")

    # ---- authoritative control clock (60 Hz), independent of generation ----
    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max = 6.0
    ctl.cfg.pitch_rate_max = 2.0
    ctl.cfg.v_max = 1.0
    total_time = args.n_chunks * 1.25 + 2.0
    n_ctrl = int(total_time * CTRL_HZ)
    ctl_poses, ctl_times, ctl_v = [], [], []
    si = 0
    ctl.set_input(**SCRIPT[0][1])
    for k in range(n_ctrl):
        t = k / CTRL_HZ
        while si + 1 < len(SCRIPT) and t >= SCRIPT[si + 1][0]:
            si += 1
            ctl.set_input(**SCRIPT[si][1])
        ctl.step(dt=1.0 / (CTRL_HZ * 1.25))
        ctl_poses.append(ctl.pose.copy())
        ctl_times.append(t)
        ctl_v.append(ctl.v.copy())
    ctl_poses = np.stack(ctl_poses)
    ctl_times = np.array(ctl_times)
    ctl_v = np.stack(ctl_v)

    def slot_pose(t):
        i = int(np.clip(round(t * CTRL_HZ), 0, len(ctl_poses) - 1))
        return ctl_poses[i]

    # ---- model ----
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    print(f"[hs] pipe built (mode {args.mode})", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    aspect = h / w
    lat_h = round(math.sqrt(W * H * aspect) // vae_stride[1] // patch[1] * patch[1])
    lat_w = round(math.sqrt(W * H / aspect) // vae_stride[2] // patch[2] * patch[2])
    h = lat_h * vae_stride[1]
    w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = (args.n_chunks - 1) * 4 + 1
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * pipe.local_attn_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads

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
    comp = Compensator(lead_rot=0.0, lead_trans=2.0)
    comp_state = dict(gate=1.0, prev_v=None)

    def plucker(rel_pose):
        rel = torch.from_numpy(np.asarray(rel_pose)).float()[None].to(dev)
        p = get_plucker_embeddings(rel, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                         h=lat_h, w=lat_w).to(pdt)

    def lead_pose(pose, t):
        """Translation-only pre-emphasis with a reversal gate (rotation untouched)."""
        i = int(np.clip(round(t * CTRL_HZ), 0, len(ctl_poses) - 1))
        v = ctl_v[i]
        vp = comp_state["prev_v"]
        n_now = float(np.linalg.norm(v))
        if vp is None or n_now < 0.02:
            tgt = 1.0 if vp is None else 0.0
        else:
            n_prev = float(np.linalg.norm(vp))
            cos = float(np.dot(vp, v) / max(n_prev * n_now, 1e-9))
            tgt = 0.0 if cos <= 0 else float(min(1.0, cos / 0.7))
        if tgt < comp_state["gate"]:
            comp_state["gate"] = tgt
        else:
            comp_state["gate"] += 0.25 * (tgt - comp_state["gate"])
        comp_state["prev_v"] = v.copy()
        T = np.eye(4)
        T[:3, 3] = v * (2.0 * comp_state["gate"])
        return pose @ T

    pipe.vae.stream_decode_start()
    parts, events, chunk_rows = [], [], []
    prev_pose = None
    t_start = time.perf_counter()
    for cid in range(args.n_chunks):
        cur = noise.split(1, dim=1)[cid]
        # condition source for this chunk
        chunk_t = time.perf_counter() - t_start
        if use_swap:
            chunk_pose = slot_pose(chunk_t)
        else:
            chunk_pose = slot_pose(chunk_t)     # snapshot at chunk start
        if use_comp:
            chunk_pose_c = lead_pose(chunk_pose, chunk_t)
        else:
            chunk_pose_c = chunk_pose
        rel0 = (np.eye(4) if prev_pose is None
                else np.linalg.inv(prev_pose) @ chunk_pose_c)
        frozen_plk = plucker(rel0)

        step_samples = []
        for ti in range(len(timesteps)):
            if use_swap:
                t_now = time.perf_counter() - t_start
                pose_now = slot_pose(t_now)
                if use_comp:
                    pose_now = lead_pose(pose_now, t_now)
                rel = (np.eye(4) if prev_pose is None
                       else np.linalg.inv(prev_pose) @ pose_now)
                plk = plucker(rel)
                cond_t = t_now
            else:
                plk = frozen_plk
                cond_t = chunk_t
            step_samples.append((time.perf_counter() - t_start, cond_t))
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * frame_seqlen,
                  "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}
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
                        x0, torch.randn(x0.shape, generator=g, device=x0.device,
                                        dtype=x0.dtype), timesteps[ti + 1])
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        with torch.no_grad():
            parts.append(pipe.vae.stream_decode_step(x0))
        t_vis = time.perf_counter() - t_start
        chunk_rows.append(dict(chunk=cid, t_start=chunk_t, t_visible=t_vis,
                               steps=step_samples, cond_t=step_samples[-1][1]))
        prev_pose = chunk_pose_c
        print(f"[hs] {args.mode} chunk {cid}: t={chunk_t:.2f}s "
              f"visible={t_vis:.2f}s cond_age={t_vis-step_samples[-1][1]:.2f}s",
              flush=True)
    pipe.vae.stream_decode_end()
    rgb = torch.cat(parts, dim=1)
    mp4 = f"{args.out_dir}/{args.mode}_s{args.seed}.mp4"
    save_video(tensor=rgb[None], save_file=mp4, fps=cfg.sample_fps,
               nrow=1, normalize=True, value_range=(-1, 1))

    # ---- latency from real timestamps ----
    print(f"\n[hs] ===== mode {args.mode}: causal admission (real timestamps) =====")
    print(f"{'t_input':>8s} {'t_admit':>8s} {'admit_lat':>10s} "
          f"{'t_visible':>10s} {'ctl_to_real':>12s}")
    rows = []
    for t_in, _inp in SCRIPT[1:]:
        admit = vis = None
        if use_swap:
            # condition is refreshed at every denoise-step boundary
            for r in chunk_rows:
                for (t_s, _ct) in r["steps"]:
                    if t_s >= t_in:
                        admit = t_s
                        break
                if admit is not None:
                    vis = r["t_visible"]
                    break
        else:
            # condition is frozen at the chunk start -> next CHUNK start
            for r in chunk_rows:
                if r["t_start"] >= t_in:
                    admit = r["t_start"]
                    vis = r["t_visible"]
                    break
        if admit is None:
            continue
        rows.append(dict(t_input=t_in, t_admit=admit, admit_lat=admit - t_in,
                         t_visible=vis, ctl_to_real=vis - t_in))
        print(f"{t_in:8.2f} {admit:8.2f} {admit-t_in:10.2f} "
              f"{vis:10.2f} {vis-t_in:12.2f}")
    if rows:
        lat = np.array([r["ctl_to_real"] for r in rows])
        adm = np.array([r["admit_lat"] for r in rows])
        print(f"\n  causal admission latency: mean={adm.mean():.2f}s max={adm.max():.2f}s")
        print(f"  control-to-real: mean={lat.mean():.2f}s "
              f"p50={np.median(lat):.2f}s worst={lat.max():.2f}s")
    ages = [r["t_visible"] - r["cond_t"] for r in chunk_rows]
    print(f"  effective_control_age: mean={np.mean(ages):.2f}s "
          f"max={np.max(ages):.2f}s")
    json.dump(dict(mode=args.mode, chunk_rows=chunk_rows, rows=rows,
                   ages=ages), open(f"{args.out_dir}/{args.mode}_s{args.seed}.json", "w"),
              indent=1)


if __name__ == "__main__":
    main()
