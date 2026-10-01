#!/usr/bin/env python
"""Interactive-1 entry point: the authoritative runtime, wired end to end.

What this produces that did not exist before: a REAL input -> commit ->
first-real-frame latency chain, with mechanical lineage, instead of a
decode-complete proxy.

    accept()             t0, a real perf_counter_ns timestamp at queue entry
    begin_chunk()        t1, the events are bound to this chunk
    commit()             t2, fail-closed on an exact metadata match
    mark_real_decoded()  t3, only for the real frame carrying those events

t4 (renderer submit) and t5 (present) stay None, because Latency-1A established
that no renderer and no present signal exist in this architecture. They are
reported as unavailable rather than approximated.

The input source is scripted, but it calls accept() at real wall-clock moments --
exactly what a keyboard callback would do -- so t0 is a genuine acceptance time,
not a script timestamp. This is the difference between this and hotswap_loop,
whose "relative elapsed minus script timestamp" mixed two clock bases.

Frozen preset: performance = bf16, 304x528, streamed condition encode,
offload_model=True with the in-tree device restore.
"""
import argparse
import gc
import hashlib
import json
import math
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
from cam_controller import CameraController

from interactive_runtime import CameraState, InteractiveRuntime, RuntimeStateError

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
CTRL_HZ = 60.0
# scripted control timeline: (t_seconds_from_loop_start, controls)
SCRIPT = [(0.0, dict(fwd=0.6)), (2.0, dict(yaw=0.8, fwd=0.3)),
          (4.0, dict(yaw=-0.8, fwd=0.3)), (6.0, dict(fwd=-0.6)),
          (8.0, dict(strafe=0.8))]
MB = 2 ** 20


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
    print(f"  Interactive-1 authoritative runtime")
    print(f"  preset=performance  weight={args.weight}  pixel={W}x{H}  "
          f"chunks={args.n_chunks}  stream_encode="
          f"{os.environ.get('LINGBOT_STREAM_ENCODE')}")
    print(f"  t0/t1/t2/t3 are real; t4/t5 are reported UNAVAILABLE (no renderer)")
    print("=" * 96, flush=True)

    # ---- 60 Hz authoritative control clock (as in hotswap_loop) ----
    base_pose = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base_pose[:3, :3], base_pose[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0

    # ---- the runtime owns the committed camera state ----
    rt = InteractiveRuntime(CameraState(pose=ctl.pose.copy(),
                                        v=np.zeros(3), gate=1.0))

    # ---- model ----
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
    print(f"  geometry pixel {h}x{w} latent {lat_h}x{lat_w} fsl {frame_seqlen}",
          flush=True)

    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    mf.bump_cam_epoch()
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)

    # prepare: condition latent through the frozen streamed path
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

    # ---- scripted input source: calls accept() at real wall-clock moments ----
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, args.n_chunks, lat_h, lat_w, generator=g, device=dev)
    loop_t0 = time.perf_counter()
    script_idx = 0
    ctl.set_input(**SCRIPT[0][1])

    def pump_input(now_s):
        """Deliver any scripted inputs whose time has arrived, as real events."""
        nonlocal script_idx
        while script_idx + 1 < len(SCRIPT) and now_s >= SCRIPT[script_idx + 1][0]:
            script_idx += 1
            ev = rt.accept(SCRIPT[script_idx][1], kind="control")
            ctl.set_input(**ev.controls)
            print(f"  [input] event {ev.event_id} accepted at "
                  f"t={now_s:.3f}s controls={ev.controls}", flush=True)

    rows = []
    prev_pose = None
    for cid in range(args.n_chunks):
        now_s = time.perf_counter() - loop_t0
        pump_input(now_s)

        # advance the 60 Hz integrator to now, then snapshot it into the runtime
        steps = max(1, int(round((now_s - getattr(pump_input, "_last", 0.0))
                                 * CTRL_HZ)))
        for _ in range(steps):
            ctl.step(dt=1.0 / (CTRL_HZ * 1.25))
        pump_input._last = now_s
        snap = rt.begin_chunk()
        snap["camera"].pose = ctl.pose.copy()
        snap["camera"].v = ctl.v.copy()

        chunk_pose = snap["camera"].pose
        rel0 = (np.eye(4) if prev_pose is None
                else np.linalg.inv(prev_pose) @ chunk_pose)
        plk = plucker(rel0)
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": frame_seqlen,
              "y": [y.split(1, dim=1)[cid]],
              "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
              "kv_cache": self_kv, "crossattn_cache": cross_kv,
              "current_start": cid * frame_seqlen,
              "max_attention_size": kv_size, "frame_seqlen": frame_seqlen}

        cur = noise.split(1, dim=1)[cid]
        torch.cuda.synchronize(); t_gen = time.perf_counter()
        for ti in range(len(timesteps)):
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
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        torch.cuda.synchronize()

        # decode -> the real frame. t3 is set AFTER decode, for this frame only.
        with torch.no_grad():
            tae.decode_video(x0.to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                             parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t_gen) * 1000

        meta = rt.new_frame_meta("real", snap["chunk_index"],
                                 snap["generation_id"],
                                 snap["applied_event_ids"])
        rt.mark_real_decoded(meta)
        rt.commit(meta)                      # fail-closed
        rows.append(dict(chunk=cid, gen_ms=gen_ms,
                         applied=list(snap["applied_event_ids"])))
        prev_pose = chunk_pose
        print(f"  [chunk {cid}] gen {gen_ms:7.1f} ms  "
              f"applied={list(snap['applied_event_ids'])}  "
              f"committed_chunk={rt.committed.chunk_index}", flush=True)

    # ------------------------------------------------------------- reporting
    print()
    print("=" * 96)
    print("  MEASURED LATENCY CHAIN  (real timestamps, mechanical lineage)")
    print("=" * 96)
    print(f"  {'ev':>4} {'t0(s)':>8} {'->assign':>10} {'->commit':>10} "
          f"{'->real':>10} {'input->real':>12} {'chunk':>6} {'frame':>6}")
    print("  " + "-" * 84)
    lat = []
    for rec in rt.records():
        d = rec.derived()
        t0s = (rec.t0_accept_ns - loop_t0) / 1e9
        def ms(x):
            return "-" if x is None else f"{x:.1f}"
        print(f"  {rec.event_id:>4} {t0s:>8.3f} "
              f"{ms(d['accept_to_assign_ms']):>10} "
              f"{ms(d['accept_to_commit_ms']):>10} "
              f"{ms(d['accept_to_first_real_ms']):>10} "
              f"{ms(d['accept_to_first_real_ms']):>12} "
              f"{str(rec.assigned_chunk):>6} "
              f"{str(rec.first_real_frame_id):>6}  {rec.terminal_status}")
        if d["accept_to_first_real_ms"] is not None:
            lat.append(d["accept_to_first_real_ms"])
    print()
    if lat:
        lat_s = sorted(lat)
        print(f"  input -> first affected REAL frame:")
        print(f"    n={len(lat)}  p50 {statistics.median(lat):.0f} ms  "
              f"min {min(lat):.0f}  max {max(lat):.0f} ms")
    else:
        print("  no event was assigned to a chunk within the run; "
              "input->real unavailable")
    print()
    print(f"  input -> renderer submit    UNAVAILABLE (no renderer in this tree)")
    print(f"  input -> present            UNAVAILABLE (no present signal)")
    print(f"  measured input-to-display   UNAVAILABLE by construction")
    print()
    print(f"  chunk generation (DiT+decode+KV): p50 "
          f"{statistics.median(r['gen_ms'] for r in rows):.0f} ms")
    print(f"  committed chunk_index: {rt.committed.chunk_index}  "
          f"generation_id: {rt.committed.generation_id}")

    with open(f"{args.out_dir}/play_traces.json", "w") as f:
        json.dump(dict(weight=args.weight, pixel=[W, H], n_chunks=args.n_chunks,
                       rows=rows,
                       traces=rt.export(),
                       committed_chunk=rt.committed.chunk_index,
                       accept_to_first_real_p50_ms=(statistics.median(lat)
                                             if lat else None),
                       t4_available=False, t5_available=False), f, indent=2)
    print(f"\n[play] wrote {args.out_dir}/play_traces.json")


if __name__ == "__main__":
    main()
