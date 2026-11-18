#!/usr/bin/env python
"""§39A/§39B: identity anchor + reacquisition against the world model.

Register sparse gameplay objects at a generated reference frame, let them leave
the FOV, then try to REACQUIRE the same persistent_id at the revisit frame
instead of creating new objects.

Metrics: reacquisition accuracy, wrong-ID rate, false-new rate, pose error,
confidence. §39B repeats it across T50 (5 s / 10 s / 20 s away).

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python reacquire_test.py --scenes 04 --seeds 42 --horizons 4,8,12,20,40,80
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
from object_permanence import OBJECTS, build_traj, patch
from object_state import PersistentObjectStore

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scenes", default="04")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--horizons", default="4,8,12,20,40,80")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/reacquire")
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
    print("[rq] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads
    from eval_two_layer import load_models, dino_feat
    load_models()

    def dino_np(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    rows = []
    for scene in scenes:
        for sd in seeds:
            for hz in horizons:
                traj, ref_chunk, out_chunks = build_traj(scene, hz)
                frames_n = len(traj)
                d = f"examples/rq_{scene}_H{hz}"
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

                # ---- register sparse objects from the reference frame ----
                store = PersistentObjectStore()
                bbs = OBJECTS[scene]
                for name, bb in bbs.items():
                    rp = patch(R, bb)
                    store.register(name, np.eye(4), [1.0, 1.0, 1.0],
                                   anchor=dino_np(rp), t=0.0,
                                   gameplay_state=dict(visible=True, uv=((bb[0]+bb[2])/2, (bb[1]+bb[3])/2)))
                # ---- reacquire at the revisit frame ----
                cands = []
                for name, bb in bbs.items():
                    rp = patch(R, bb)
                    best = (-1.0, None, None)
                    fr_vec = dino_np(rp)
                    for dy in np.linspace(-0.06, 0.06, 13):
                        for dx in np.linspace(-0.06, 0.06, 13):
                            bx = int((bb[0] + dx) * Ww); by = int((bb[1] + dy) * Hh)
                            pw, ph = int((bb[2] - bb[0]) * Ww), int((bb[3] - bb[1]) * Hh)
                            if bx < 0 or by < 0 or bx + pw > Ww or by + ph > Hh:
                                continue
                            cd = V[by:by + ph, bx:bx + pw]
                            if cd.size == 0:
                                continue
                            s = float((fr_vec * dino_np(cd)).sum())
                            if s > best[0]:
                                best = (s, (bx, by), cd)
                    if best[1] is None:
                        continue
                    exp = (int(bb[0] * Ww), int(bb[1] * Hh))
                    cands.append(dict(name=name, anchor=dino_np(best[2]),
                                      world_transform=np.eye(4),
                                      bounds=[1.0, 1.0, 1.0],
                                      pose_err=float(np.hypot(best[1][0] - exp[0],
                                                              best[1][1] - exp[1])),
                                      self_sim=best[0],
                                      uv=((best[1][0]+best[1][0]+int((bb[2]-bb[0])*Ww))/2/Ww,
                                          (best[1][1]+best[1][1]+int((bb[3]-bb[1])*Hh))/2/Hh)))
                dec = store.reacquire(cands, t=out_chunks * 0.25)
                # ---- score ----
                ids = {o.persistent_id: o.semantic_type for o in store}
                correct = wrong = new = amb = 0
                pose_errs, confs = [], []
                for c, dd in zip(cands, dec):
                    if dd["decision"] == "reacquire" and ids.get(dd["id"]) == c["name"]:
                        correct += 1
                    elif dd["decision"] == "reacquire":
                        wrong += 1
                    elif dd["decision"] == "ambiguous":
                        amb += 1
                    else:
                        new += 1
                    pose_errs.append(c["pose_err"])
                    confs.append(dd["sim"])
                n = max(len(cands), 1)
                rows.append(dict(scene=scene, seed=sd, H=hz, out_chunks=out_chunks,
                                 seconds=out_chunks * 0.25, n_obj=len(cands),
                                 correct=correct, wrong=wrong, new=new, amb=amb,
                                 acc=correct / n, wrong_rate=wrong / n,
                                 new_rate=new / n, pose_err=float(np.mean(pose_errs)),
                                 conf=float(np.mean(confs))))
                print(f"[rq] {scene}/H{hz}/s{sd} out={out_chunks}ch "
                      f"({out_chunks*0.25:.2f}s) acc={correct}/{len(cands)} "
                      f"wrong={wrong} new={new} amb={amb} "
                      f"pose_err={np.mean(pose_errs):.1f}px conf={np.mean(confs):.3f}",
                      flush=True)
                del y, self_kv, cross_kv, outs
                gc.collect(); torch.cuda.empty_cache()

    print("\n[rq] ===== reacquisition vs elapsed time =====")
    print(f"{'scene':6s} {'H':>3s} {'out_ch':>6s} {'sec':>5s} {'acc':>6s} "
          f"{'wrong':>6s} {'new':>5s} {'amb':>5s} {'pose_err':>9s} {'conf':>6s}")
    for r in rows:
        print(f"{r['scene']:6s} {r['H']:3d} {r['out_chunks']:6d} {r['seconds']:5.2f} "
              f"{r['acc']:6.2f} {r['wrong_rate']:6.2f} {r['new_rate']:5.2f} "
              f"{r['amb']/max(r['n_obj'],1):5.2f} {r['pose_err']:9.1f} {r['conf']:6.3f}")
    json.dump(rows, open(f"{args.out_dir}/reacquire.json", "w"), indent=1)


if __name__ == "__main__":
    main()
