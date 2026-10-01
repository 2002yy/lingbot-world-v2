#!/usr/bin/env python
"""Generate held-out (step0, final) latent pairs for the Preview-Head generalisation gate.

Two axes, so a failure is attributable:
  H1  same scene (examples/04), DIFFERENT seed and DIFFERENT control script
      -> is the head just memorising the nine training pairs?
  H2  a different scene entirely
      -> does the stage1-feature -> final mapping hold across visual domains?

Geometry stays 304x528 for every holdout, because the pipeline pre-resizes the image
regardless of the native aspect. That keeps the comparison about content, not about a
resolution change.

Nothing here trains. It only produces data.
"""
import argparse
import gc
import hashlib
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

import wan
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

# Training used this script (profile_chunk.py). H1 uses a different one so the
# controls are genuinely unseen.
TRAIN_SCRIPT = [{"forward": 1.0}, {"yaw": 1.0}, {"forward": 1.0, "right": 1.0},
                {"right": -1.0}, {"pitch": -1.0}]
HOLDOUT_SCRIPT = [{"forward": -1.0}, {"yaw": -1.0}, {"pitch": 1.0},
                  {"right": 1.0}, {"forward": 0.7, "yaw": 0.6},
                  {"strafe": 0.9}, {"forward": 0.4, "pitch": -0.7}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--base", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weight", default="bf16", choices=["bf16", "fp8_lowmem"])
    ap.add_argument("--pixel", default="304x528")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_chunks", type=int, default=9)
    ap.add_argument("--local_window", type=int, default=6)
    ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_WEIGHT_MODE"] = args.weight
    os.environ["LINGBOT_FP8"] = "0" if args.weight == "bf16" else "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"
    os.environ.setdefault("LINGBOT_STREAM_ENCODE", "1")

    W, H = (int(x) for x in args.pixel.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    print("=" * 84)
    print(f"  holdout generation   tag={args.tag}  base={args.base}  "
          f"seed={args.seed}  chunks={args.n_chunks}")
    print("=" * 84, flush=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_window,
        sink_size=args.sink, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch = pipe.vae_stride, pipe.patch_size
    img_pil = Image.open(f"{args.base}/image.jpg").convert("RGB")
    img_pil = img_pil.resize((W, H), Image.BICUBIC)
    img = TF.to_tensor(img_pil).sub_(0.5).div_(0.5).to(dev)
    h, w = img.shape[1:]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch[1] * patch[2])
    F = (args.n_chunks - 1) * 4 + 1
    kv_size = fsl * args.local_window
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads
    print(f"  geometry {h}x{w}  latent {lat_h}x{lat_w}  fsl {fsl}", flush=True)

    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    mf.bump_cam_epoch()
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe._condition_latent(img, F, h, w)
    msk = torch.ones(1, F, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y])
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype,
        device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype,
        device=dev)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def plucker(rel):
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(dev)
        p = get_plucker_embeddings(r, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h,
                         w=lat_w).to(pdt)

    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, args.n_chunks, lat_h, lat_w, generator=g, device=dev)
    saved, ctrls = {}, []
    prev_pose = np.eye(4)
    for cid in range(args.n_chunks):
        ctrl = HOLDOUT_SCRIPT[cid % len(HOLDOUT_SCRIPT)]
        ctrls.append(ctrl)
        from control_reduce import reduce_controls
        from interactive_runtime import CameraState
        cam = reduce_controls(CameraState(pose=prev_pose, v=np.zeros(3)), [ctrl])
        chunk_pose = cam.pose
        rel0 = np.linalg.inv(prev_pose) @ chunk_pose
        plk = plucker(rel0)
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": fsl,
              "y": [y.split(1, dim=1)[cid]],
              "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
              "kv_cache": self_kv, "crossattn_cache": cross_kv,
              "current_start": cid * fsl,
              "max_attention_size": kv_size, "frame_seqlen": fsl}
        cur = noise.split(1, dim=1)[cid]
        for ti in range(len(timesteps)):
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=(ti == 0 and cid == 0), **kw)[0]
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti == 0 and cid >= 1:
                    saved[f"c{cid}_step0"] = x0.detach().float().cpu().clone()
                if ti < len(timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=g, device=dev,
                                        dtype=x0.dtype), timesteps[ti + 1])
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        if cid >= 1:
            saved[f"c{cid}_final"] = x0.detach().float().cpu().clone()
        prev_pose = chunk_pose
        print(f"  chunk {cid} done", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(saved, args.out)
    import json
    with open(args.out + ".meta.json", "w") as f:
        json.dump(dict(tag=args.tag, base=args.base, seed=args.seed,
                       n_chunks=args.n_chunks, ctrls=ctrls,
                       keys=sorted(saved.keys())), f, indent=2)
    print(f"\n  saved {args.out}  ({len(saved)} tensors)")


if __name__ == "__main__":
    main()
