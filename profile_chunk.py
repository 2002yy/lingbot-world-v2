#!/usr/bin/env python
"""§Latency-2A: authoritative chunk critical-path decomposition.

Answers, for the frozen performance preset: where does the ~750 ms chunk go, and
which parts are on the critical path?

Three things it does NOT do: it does not optimise anything, it does not change the
LatencyTraceRecord schema, and it does not mix CUDA and host clocks.

It also runs the instrumentation-overhead gate: the same preset with profiling off
and on, so the instrument cannot be measuring itself.
"""
import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image

import wan
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings

from chunk_phase import ChunkPhaseTrace, cuda_phase, print_accounting, summarise
from control_reduce import reduce_controls
from interactive_runtime import CameraState, InteractiveRuntime

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
MS = 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--weight", default="bf16", choices=["bf16", "fp8_lowmem"])
    ap.add_argument("--pixel", default="304x528")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_chunks", type=int, default=10)
    ap.add_argument("--local_window", type=int, default=6)
    ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--profile", type=int, default=1)
    ap.add_argument("--save_steps", type=int, default=0,
                    help="also save the step0/step1/final latents of one chunk, "
                         "so offline decoding can test whether an early step "
                         "already carries a usable picture")
    ap.add_argument("--out_dir", required=True)
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
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 96)
    print(f"  Latency-2A chunk critical-path decomposition   "
          f"weight={args.weight} pixel={W}x{H} chunks={args.n_chunks} "
          f"profile={args.profile}")
    print("=" * 96, flush=True)

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
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = (args.n_chunks - 1) * 4 + 1
    kv_size = frame_seqlen * args.local_window
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads

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

    rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, args.n_chunks, lat_h, lat_w, generator=g, device=dev)
    ctl_script = [{"forward": 1.0}, {"yaw": 1.0}, {"forward": 1.0, "right": 1.0},
                  {"right": -1.0}, {"pitch": -1.0}]

    traces, wall_ms, saved = [], [], {}
    prev_pose = None
    for cid in range(args.n_chunks):
        ctrl = ctl_script[cid % len(ctl_script)]
        rt.accept(ctrl)
        tr = ChunkPhaseTrace(chunk_index=cid, generation_id=0)

        t_begin = time.perf_counter_ns()
        tr.begin_ns = t_begin
        snap = rt.begin_chunk()
        tr.control_done_ns = time.perf_counter_ns()

        chunk_pose = snap["candidate_camera"].pose
        rel0 = (np.eye(4) if prev_pose is None
                else np.linalg.inv(prev_pose) @ chunk_pose)
        plk = plucker(rel0)
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": frame_seqlen,
              "y": [y.split(1, dim=1)[cid]],
              "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
              "kv_cache": self_kv, "crossattn_cache": cross_kv,
              "current_start": cid * frame_seqlen,
              "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}
        tr.conditioning_done_ns = time.perf_counter_ns()

        cur = noise.split(1, dim=1)[cid]
        for ti in range(len(timesteps)):
            stop = cuda_phase(dev) if args.profile else None
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=(ti == 0 and cid == 0), **kw)[0]
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
                if ti < len(timesteps) - 1:
                    cur = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=g, device=dev,
                                        dtype=x0.dtype), timesteps[ti + 1])
            if args.save_steps and cid == 1:
                saved[f"step{ti}"] = x0.detach().float().cpu().clone()
            tr.dit_step_ms.append(stop() if stop else float("nan"))

        stop = cuda_phase(dev) if args.profile else None
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        tr.kv_update_ms = stop() if stop else float("nan")
        torch.cuda.synchronize()

        meta = rt.new_frame_meta("real", snap["chunk_index"],
                                 snap["generation_id"],
                                 snap["applied_event_ids"])
        rt.commit(meta)
        tr.commit_ns = time.perf_counter_ns()

        stop = cuda_phase(dev) if args.profile else None
        with torch.no_grad():
            tae.decode_video(x0.to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                             parallel=False, show_progress_bar=False)
        tr.decode_ms = stop() if stop else float("nan")
        torch.cuda.synchronize()
        rt.mark_real_decoded(meta)
        tr.first_real_ns = time.perf_counter_ns()

        tr.end_ns = time.perf_counter_ns()
        traces.append(tr)
        wall_ms.append((tr.end_ns - tr.begin_ns) / MS)
        prev_pose = chunk_pose

    # ---------------------------------------------------------------- report
    print()
    if args.profile:
        print("=" * 96)
        print("  CHUNK CRITICAL-PATH ACCOUNTING")
        print("=" * 96)
        s = summarise(traces)
        print_accounting(s, f"weight={args.weight}")
        print()
        print("  PER-CHUNK WALL (ms): "
              + " ".join(f"{x:.0f}" for x in wall_ms))
        print()
        print("  NOTE: CUDA phase durations come from CUDA events; the host seams")
        print("        come from perf_counter_ns. The two are never subtracted.")
        print("        The host total is what the accounting above reconciles.")
    else:
        print(f"  profiling OFF: chunk wall p50 "
              f"{statistics.median(wall_ms[1:]):.1f} ms  (n={len(wall_ms)})")

    out = dict(weight=args.weight, pixel=[W, H], chunks=args.n_chunks,
               profile=args.profile,
               wall_ms=wall_ms, wall_p50=statistics.median(wall_ms[1:]),
               accounting=(summarise(traces) if args.profile else None),
               per_chunk=[t.accounting() for t in traces])
    with open(f"{args.out_dir}/phase_{'on' if args.profile else 'off'}.json",
              "w") as f:
        json.dump(out, f, indent=2)

    if args.save_steps and saved:
        torch.save(saved, f"{args.out_dir}/step_latents.pt")
        print(f"\n  saved intermediate latents for one chunk: "
              f"{list(saved.keys())}")
    print(f"\n  wrote {args.out_dir}/phase_"
          f"{'on' if args.profile else 'off'}.json")


if __name__ == "__main__":
    main()
