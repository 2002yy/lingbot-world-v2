#!/usr/bin/env python
"""§Preview-1A runner: measure the step0 preview cost and the interference it causes.

Arms (--arm):
    A  baseline, no preview
    B  step0 + synchronous preview decode
    C  step0 snapshot + separate-stream preview decode

For B and C, --clone 0/1 controls whether the step0 latent is copied before being
handed to the decoder. Both are run so the handoff price is visible rather than
assumed: reporting "preview costs 36 ms" from TAE(latent) alone would omit the copy,
its VRAM and its bandwidth contention.
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

from chunk_phase import cuda_phase as phase_stop
from control_reduce import reduce_controls
from interactive_runtime import CameraState, InteractiveRuntime
from preview_trace import PreviewTrace, assert_non_authoritative

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
MS = 1e6
CTRL = [{"forward": 1.0}, {"forward": 1.0, "right": 1.0}, {"yaw": 1.0},
        {"right": -1.0}, {"pitch": -1.0}, {"forward": 0.6, "yaw": -0.8},
        {"forward": -1.0}, {"right": 1.0, "yaw": 0.5}]


def mem():
    free, total = torch.cuda.mem_get_info()
    return dict(alloc=torch.cuda.memory_allocated() / MS,
                reserved=torch.cuda.memory_reserved() / MS,
                free=free / MS, total=total / MS)


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
    ap.add_argument("--arm", default="A", choices=list("ABC"))
    ap.add_argument("--clone", type=int, default=1)
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
    print(f"  Preview-1A   arm={args.arm}  clone={args.clone}  "
          f"weight={args.weight}  pixel={W}x{H}  chunks={args.n_chunks}")
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
    fsl = (lat_h * lat_w) // (patch[1] * patch[2])
    F = (args.n_chunks - 1) * 4 + 1
    kv_size = fsl * args.local_window
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
    preview_stream = torch.cuda.Stream() if args.arm == "C" else None

    rows, ptraces = [], []
    prev_pose = None
    torch.cuda.reset_peak_memory_stats()
    for cid in range(args.n_chunks):
        ctrl = CTRL[cid % len(CTRL)]
        ev = rt.accept(ctrl)
        t_accept = ev.t0_ns
        snap = rt.begin_chunk()
        chunk_pose = snap["candidate_camera"].pose
        rel0 = (np.eye(4) if prev_pose is None
                else np.linalg.inv(prev_pose) @ chunk_pose)
        plk = plucker(rel0)
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": fsl,
              "y": [y.split(1, dim=1)[cid]],
              "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
              "kv_cache": self_kv, "crossattn_cache": cross_kv,
              "current_start": cid * fsl,
              "max_attention_size": kv_size, "frame_seqlen": fsl}

        cur = noise.split(1, dim=1)[cid]
        step_ms = []
        pv = PreviewTrace(chunk_index=snap["chunk_index"],
                          generation_id=snap["generation_id"],
                          applied_event_ids=snap["applied_event_ids"],
                          source_step=0, accept_ns=t_accept)
        for ti in range(len(timesteps)):
            stop = phase_stop(dev)
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                npred = pipe.model(
                    x=[cur.to(dev)], t=torch.stack([timesteps[ti]]).to(dev),
                    cross_attn_first_call=(ti == 0 and cid == 0), **kw)[0]
                x0 = pipe._convert_flow_pred_to_x0(
                    flow_pred=npred, xt=cur, timestep=timesteps[ti],
                    scheduler=pipe.scheduler)
            step_ms.append(stop())

            if ti == 0 and args.arm in ("B", "C"):
                torch.cuda.synchronize()
                pv.step0_done_ns = time.perf_counter_ns()
                # ---- latent handoff ----
                th = time.perf_counter()
                pv_lat = x0.detach().clone() if args.clone else x0.detach()
                torch.cuda.synchronize()
                pv.handoff_ms = (time.perf_counter() - th) * 1000
                pv.handoff_done_ns = time.perf_counter_ns()
                if args.arm == "B":
                    td = time.perf_counter()
                    with torch.no_grad():
                        tae.decode_video(
                            pv_lat.permute(1, 0, 2, 3).unsqueeze(0),
                            parallel=False, show_progress_bar=False)
                    torch.cuda.synchronize()
                    pv.preview_decode_ms = (time.perf_counter() - td) * 1000
                    pv.preview_decoded_ns = time.perf_counter_ns()
                else:
                    # arm C: start on a side stream and let the main stream continue
                    preview_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(preview_stream):
                        with torch.no_grad():
                            tae.decode_video(
                                pv_lat.permute(1, 0, 2, 3).unsqueeze(0),
                                parallel=False, show_progress_bar=False)
                    pv.overlap = True
            if ti < len(timesteps) - 1:
                cur = pipe.scheduler.add_noise(
                    x0, torch.randn(x0.shape, generator=g, device=dev,
                                    dtype=x0.dtype), timesteps[ti + 1])

        stop = phase_stop(dev)
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        kv_ms = stop()
        torch.cuda.synchronize()

        if args.arm == "C":
            torch.cuda.current_stream().wait_stream(preview_stream)
            torch.cuda.synchronize()
            pv.preview_decoded_ns = time.perf_counter_ns()

        meta = rt.new_frame_meta("real", snap["chunk_index"],
                                 snap["generation_id"],
                                 snap["applied_event_ids"])
        rt.commit(meta)
        td = time.perf_counter()
        with torch.no_grad():
            tae.decode_video(x0.to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                             parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        auth_dec_ms = (time.perf_counter() - td) * 1000
        rt.mark_real_decoded(meta)

        # ---- correctness: a preview must be refused by every authoritative seam
        pv_meta = rt.new_frame_meta("preview", snap["chunk_index"],
                                    snap["generation_id"],
                                    snap["applied_event_ids"])
        assert_non_authoritative(rt, pv_meta)

        tr = rt.record(ev.event_id)
        m = mem()
        rows.append(dict(chunk=cid, step_ms=step_ms, kv_ms=kv_ms,
                         auth_decode_ms=auth_dec_ms,
                         dit_total_ms=sum(step_ms),
                         auth_to_first_real_ms=(tr.t3_first_real_ns - t_accept)
                         / MS if tr.t3_first_real_ns else None,
                         preview_ready_ms=pv.ready_ms(),
                         preview_step0_to_decoded_ms=pv.step0_to_preview_ms(),
                         handoff_ms=pv.handoff_ms,
                         preview_decode_ms=pv.preview_decode_ms,
                         **m))
        if args.arm in ("B", "C"):
            ptraces.append(pv)
        prev_pose = chunk_pose

    # ---------------------------------------------------------------- report
    warm = rows[1:] if len(rows) > 1 else rows
    print()
    print("=" * 96)
    print(f"  PREVIEW-1A RESULT   arm={args.arm} clone={args.clone}")
    print("=" * 96)

    def p50(key, src=None):
        vals = [r[key] for r in (src or warm) if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    auth = p50("auth_to_first_real_ms")
    dit = p50("dit_total_ms")
    kv = p50("kv_ms")
    adec = p50("auth_decode_ms")
    print(f"  authoritative input->first-real   {auth:8.1f} ms")
    print(f"    DiT total                       {dit:8.1f} ms")
    print(f"    KV update                       {kv:8.1f} ms")
    print(f"    authoritative decode            {adec:8.1f} ms")
    if args.arm in ("B", "C"):
        pr = p50("preview_ready_ms", rows)
        s0 = p50("preview_step0_to_decoded_ms", rows)
        ho = p50("handoff_ms", rows)
        pd = p50("preview_decode_ms", rows) if args.arm == "B" else None
        print()
        print(f"  preview input->decoded            {pr:8.1f} ms")
        print(f"    step0 -> preview decoded        {s0:8.1f} ms")
        print(f"    latent handoff (clone={args.clone})   {ho:8.1f} ms")
        if pd is not None:
            print(f"    preview decode                  {pd:8.1f} ms")
        else:
            print(f"    preview decode                  overlapped on a side stream")
    pk = max(r["reserved"] for r in rows)
    print()
    print(f"  peak reserved                     {pk:8.1f} MiB")
    print(f"  min free                          {min(r['free'] for r in rows):8.1f} MiB")
    print(f"  per-chunk DiT step totals (ms): "
          + " ".join(f"{sum(r['step_ms']):.0f}" for r in warm))

    out = dict(arm=args.arm, clone=args.clone, weight=args.weight, pixel=[W, H],
               n_chunks=args.n_chunks, rows=rows,
               preview_traces=[dict(chunk_index=p.chunk_index,
                                    applied=list(p.applied_event_ids),
                                    ready_ms=p.ready_ms(),
                                    handoff_ms=p.handoff_ms,
                                    preview_decode_ms=p.preview_decode_ms,
                                    overlap=p.overlap) for p in ptraces])
    with open(f"{args.out_dir}/arm_{args.arm}_clone{args.clone}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {args.out_dir}/arm_{args.arm}_clone{args.clone}.json")


if __name__ == "__main__":
    main()
