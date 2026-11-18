#!/usr/bin/env python
"""§40: Authority Reconciliation -- explicit gameplay state vs the generated world.

First object: a 1-bit door (open/closed) rendered on the watchtower opening of
scene 04. Three layers, no DiT modification yet:

    A  baseline            state changed but NOT communicated to the model
    B  display-only fix    composite an 'open' anchor over the door ROI
    C  correction+feedback NOT implemented here (needs latent write-back)

Metrics:
    State Compliance Rate   does the model render the explicit state
    Collateral Damage       how much of the world OUTSIDE the ROI changes
    Correction Cost         how strong the external correction has to be

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python state_authority.py --scenes 04 --seeds 42 --horizons 4,12,40
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
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange
from object_permanence import build_traj, patch
from object_state import PersistentObjectStore

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

# the watchtower window is the "door" in scene 04
DOOR_BB = (0.70, 0.50, 0.84, 0.66)


def make_open_anchor(closed_patch):
    """Synthesise the 'open' appearance: the dark opening becomes bright."""
    g = cv2.cvtColor(closed_patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dark = g < np.percentile(g, 40)
    out = closed_patch.astype(np.float32).copy()
    out[dark] = out[dark] * 0.35 + 210 * 0.65      # brighten the opening
    return out.clip(0, 255).astype(np.uint8)


def composite(frame, anchor, bb):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = bb
    ax0, ay0, ax1, ay1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    out = frame.copy()
    a = cv2.resize(anchor, (ax1 - ax0, ay1 - ay0))
    out[ay0:ay1, ax0:ax1] = a
    return out, (ax0, ay0, ax1, ay1)


def roi_stats(frame, roi):
    x0, y0, x1, y1 = roi
    return float(frame[y0:y1, x0:x1].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scenes", default="04")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--horizons", default="4,12,40")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/authority")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",")]
    scenes = args.scenes.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[au] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads
    from eval_two_layer import load_models, dino_feat, lpips_d
    load_models()

    def dino_np(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    rows = []
    for scene in scenes:
        for sd in seeds:
            for hz in horizons:
                traj, ref_chunk, out_chunks = build_traj(scene, hz)
                frames_n = len(traj)
                d = f"examples/au_{scene}_H{hz}"
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
                n_lat = (frames_n - 1) // 4 + 1
                pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
                Ks = get_Ks_transformed(
                    torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
                    480, 832, h, w, h, w)[0].to(dev)
                y = pipe.vae.encode([torch.concat([
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                                    mode='bicubic').transpose(0, 1),
                    torch.zeros(3, frames_n - 1, h, w)], dim=1).to(dev)])[0]
                msk = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
                msk[:, 1:] = 0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                                    msk[:, 1:]], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
                y = torch.concat([msk, y])
                self_kv = pipe._initialize_self_kv_cache(
                    num_layers=ma.num_layers, shape=[1, kv_size, lh, hd],
                    dtype=dtype, device=dev)
                cross_kv = pipe._initialize_crossattn_cache(
                    num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
                    dtype=dtype, device=dev)
                pipe._cross_attn_initialized = False
                timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
                g = torch.Generator(device=dev); g.manual_seed(sd)
                c2w = interpolate_camera_poses(
                    np.linspace(0, frames_n - 1, frames_n),
                    torch.from_numpy(traj[:, :3, :3]).float(),
                    torch.from_numpy(traj[:, :3, 3]).float(),
                    np.linspace(0, frames_n - 1, n_lat)).to(dev)
                rel_all = compute_relative_poses(c2w, framewise=True)
                outs = []
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
                    with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                        pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                                   cross_attn_first_call=False, **kw)
                    with torch.no_grad():
                        fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                              parallel=False, show_progress_bar=False)
                    outs.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                                 * 255.0).clip(0, 255).astype(np.uint8))
                R = outs[min(ref_chunk, len(outs) - 1)]
                V = outs[-1]
                Hh, Ww = R.shape[:2]
                closed = patch(R, DOOR_BB)
                open_anchor = make_open_anchor(closed)

                # ---- §39: register the door and set the explicit state ----
                store = PersistentObjectStore()
                door = store.register("door", np.eye(4), [1, 2, 0.2],
                                      anchor=dino_np(closed), t=0.0,
                                      gameplay_state=dict(open=False,
                                                          uv=((DOOR_BB[0] + DOOR_BB[2]) / 2,
                                                              (DOOR_BB[1] + DOOR_BB[3]) / 2)))
                store.set_state(door.persistent_id, open=True)

                # ---- §40A: did the model comply on its own? ----
                gen_door = patch(V, DOOR_BB)
                sim_closed = float((dino_np(gen_door) * dino_np(closed)).sum())
                sim_open = float((dino_np(gen_door) * dino_np(open_anchor)).sum())
                compliance = int(sim_open > sim_closed)

                # ---- §40B: display-only correction + collateral damage ----
                fixed, roi = composite(V, open_anchor, DOOR_BB)
                fixed, roi = composite(V, open_anchor, DOOR_BB)
                # collateral: change OUTSIDE the ROI, and change INSIDE the ROI
                m = np.ones((Hh, Ww), bool)
                m[roi[1]:roi[3], roi[0]:roi[2]] = False
                collat = float(np.abs(fixed.astype(float) - V.astype(float))[m].mean())
                inside = float(np.abs(fixed.astype(float) - V.astype(float))[~m].mean())
                cost = float(np.abs(open_anchor.astype(float) - closed.astype(float)).mean())
                rows.append(dict(scene=scene, seed=sd, H=hz, out_chunks=out_chunks,
                                 seconds=out_chunks * 0.25,
                                 sim_closed=sim_closed, sim_open=sim_open,
                                 compliance=compliance,
                                 roi_delta=inside, collateral=collat,
                                 correction_cost=cost))
                print(f"[au] {scene}/H{hz}/s{sd} out={out_chunks}ch "
                      f"({out_chunks*0.25:.1f}s) sim_closed={sim_closed:.3f} "
                      f"sim_open={sim_open:.3f} COMPLIANCE={compliance} "
                      f"roi_delta={inside:.1f} collateral={collat:.3f} "
                      f"cost={cost:.1f}", flush=True)
                del y, self_kv, cross_kv, outs
                gc.collect(); torch.cuda.empty_cache()

    print("\n[au] ===== §40A/§40B summary =====")
    print(f"{'scene':6s} {'H':>3s} {'sec':>5s} {'sim_closed':>11s} "
          f"{'sim_open':>9s} {'compliant':>10s} {'roi_delta':>10s} "
          f"{'collateral':>11s} {'cost':>6s}")
    for r in rows:
        print(f"{r['scene']:6s} {r['H']:3d} {r['seconds']:5.1f} "
              f"{r['sim_closed']:11.3f} {r['sim_open']:9.3f} "
              f"{r['compliance']:10d} {r['roi_delta']:10.1f} "
              f"{r['collateral']:11.3f} {r['correction_cost']:6.1f}")
    if rows:
        print(f"\n  §40A State Compliance Rate (model complies on its own) = "
              f"{np.mean([r['compliance'] for r in rows]):.2f}")
        print(f"  §40B display fix: ROI delta = "
              f"{np.mean([r['roi_delta'] for r in rows]):.1f}, "
              f"collateral OUTSIDE ROI = "
              f"{np.mean([r['collateral'] for r in rows]):.3f} "
              f"(0 => perfectly local)")
    json.dump(rows, open(f"{args.out_dir}/authority.json", "w"), indent=1)


if __name__ == "__main__":
    main()
