#!/usr/bin/env python
"""P2.5: does `cudagraph_support_input_mutation` actually unblock CUDA Graph?

FINDING THAT MOTIVATES THIS SCRIPT
----------------------------------
Reading torch 2.8's own source:

    # torch/_inductor/config.py:1116
    cudagraph_support_input_mutation = False if is_fbcode() else True

So on our build the flag is ALREADY True by default, yet we still observed
`skipping cudagraphs due to mutated inputs` at crossattn_cache["k"].copy_(k).
That is "result C" territory: before concluding anything about the model we must
establish (a) the runtime value of the flag, (b) whether forcing it changes the
outcome, and (c) if not, what the actual unsupported mutation is.

VARIANTS (run separately, one per invocation, so attribution is unambiguous)
---------------------------------------------------------------------------
  off   : flag explicitly False                      (control)
  on    : flag explicitly True                       (P2.5)
  on_mark: flag True + torch.compiler.cudagraph_mark_step_begin() at the real
           CHUNK boundary (P2.5b). The marker is NOT placed inside the 3-step
           denoise loop -- it tells the runtime one iteration has ended, which
           for us means one chunk, not one denoise step.

WHAT IS REPORTED
----------------
  * runtime value of cudagraph_support_input_mutation
  * whether "skipping cudagraphs" still appears (and where)
  * per-chunk latency (cold + warm)
  * whether the CUDA Graph was actually captured/replayed

Run one variant per process:
  CGPU_VARIANT=on LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TORCH_LOGS=perf_hints python cg_probe.py --scene 04 --chunks 12
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
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

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--out_dir", default="output/cg")
    args = ap.parse_args()

    variant = os.environ.get("CGPU_VARIANT", "on")
    import torch._inductor.config as ind
    want = variant != "off"
    # NOTE: this is an attribute of the `triton` sub-config, NOT module-level:
    #   torch/_inductor/config.py -> class triton:
    #       cudagraph_support_input_mutation = False if is_fbcode() else True
    # Its documented scope is mutations "from prior cudagraph pool", i.e. inputs
    # that are outputs of a previously captured graph -- not arbitrary eager
    # inputs. Our KV cache buffers are plain eager inputs, so this experiment is
    # expected to land on "result C" (still skipped). We run it to be sure.
    if not hasattr(ind.triton, "cudagraph_support_input_mutation"):
        print("[cg] FATAL: triton.cudagraph_support_input_mutation absent", flush=True)
        sys.exit(2)
    default_val = ind.triton.cudagraph_support_input_mutation
    ind.triton.cudagraph_support_input_mutation = want
    print(f"[cg] variant={variant}  "
          f"triton.cudagraph_support_input_mutation: default={default_val} "
          f"-> now={ind.triton.cudagraph_support_input_mutation}", flush=True)
    print(f"[cg] triton.cudagraph_trees={ind.triton.cudagraph_trees}  "
          f"triton.cudagraphs={ind.triton.cudagraphs}  "
          f"torch={torch.__version__}", flush=True)

    repo = os.path.dirname(os.path.abspath(__file__))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()
    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    frames_n = (args.frames - 1) // 4 * 4 + 1
    n_lat = (frames_n - 1) // 4 + 1
    n_test = min(args.chunks, n_lat)

    d = f"examples/cg_{scene}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    img = (torch.nn.functional.interpolate(
        torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None].float(),
        size=(int(np.sqrt(W * H * (480 / 832)) // 8 * 8),
              int(np.sqrt(W * H / (480 / 832)) // 8 * 8)),
        mode='bicubic').squeeze(0) / 255.0 - 0.5) / 0.5
    h, w = img.shape[1:]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = int(math.ceil(fsl / pipe.sp_size)) * pipe.sp_size
    kv_size = fsl * args.local_attn_size
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        img[None].transpose(0, 1).to(dev),
        torch.zeros(3, frames_n - 1, h, w, device=dev)], dim=1)])[0]
    msk = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y]).detach()
    del img
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()

    p = np.load(f"examples/{scene}/poses.npy")
    traj = np.tile(p, (frames_n // len(p) + 1, 1, 1))[:frames_n]
    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    # Caches allocated once (address stability is NOT the variable here).
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers,
        shape=[1, kv_size, ma.num_heads // pipe.sp_size, ma.dim // ma.num_heads],
        dtype=dtype, device=dev, python_metadata=pipe._py_cache_meta)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers,
        shape=[1, 512, ma.num_heads, ma.dim // ma.num_heads],
        dtype=dtype, device=dev, python_metadata=pipe._py_cache_meta)

    def reset():
        for c in self_kv:
            c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False
            c["k"].zero_(); c["v"].zero_()

    torch._dynamo.reset()
    model = torch.compile(pipe.model, mode="reduce-overhead", fullgraph=False)

    def run(nchunks):
        reset()
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        per = []
        for cid in range(nchunks):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            pp = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                           c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * fsl,
                  "max_attention_size": kv_size, "frame_seqlen": fsl}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for ti in range(len(timesteps)):
                with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                    npred = model(x=[cur.to(dev)],
                                  t=torch.stack([timesteps[ti]]).to(dev),
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
                model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                      cross_attn_first_call=False, **kw)
            torch.cuda.synchronize()
            per.append(time.perf_counter() - t0)
            # P2.5b: step marker at the REAL chunk boundary (not inside denoise)
            if variant == "on_mark":
                torch.compiler.cudagraph_mark_step_begin()
        return per

    torch.cuda.reset_peak_memory_stats()
    cold = run(n_test)
    cold_alloc = torch.cuda.max_memory_allocated() / 2**20
    cold_resv = torch.cuda.max_memory_reserved() / 2**20
    print(f"[cg] cold ms: {[round(x*1000,1) for x in cold]}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    warm = run(n_test)
    warm_alloc = torch.cuda.max_memory_allocated() / 2**20
    warm_resv = torch.cuda.max_memory_reserved() / 2**20
    ws = sorted(warm)
    print(f"[cg] warm ms: {[round(x*1000,1) for x in warm]}", flush=True)
    print(f"[cg] warm median {statistics.median(warm)*1000:.1f} ms "
          f"(min {ws[0]*1000:.1f} / max {ws[-1]*1000:.1f})", flush=True)
    print(f"[cg] VRAM cold {cold_alloc:.0f}/{cold_resv:.0f} MiB  "
          f"warm {warm_alloc:.0f}/{warm_resv:.0f} MiB", flush=True)
    json.dump(dict(variant=variant, head=head,
                   flag_default=default_val, flag_now=bool(want),
                   cold_ms=[x*1000 for x in cold],
                   warm_ms=[x*1000 for x in warm],
                   warm_median_ms=statistics.median(warm)*1000,
                   cold_alloc=cold_alloc, cold_resv=cold_resv,
                   warm_alloc=warm_alloc, warm_resv=warm_resv),
              open(f"{args.out_dir}/cg_{variant}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
