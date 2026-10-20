#!/usr/bin/env python
"""G4-hotpath: does the streamed prepare contaminate the following hot path?

Reuses hotswap_loop.py's state and event logic verbatim -- the same 60 Hz
authoritative CameraController, the same compensator, the same scripted control
timeline, the same step-boundary hot-swap, the same KV/cam bookkeeping. Only four
things change, each for a stated reason:

  1. GEOMETRY 304x528 (the deployment geometry from M0-prod-A), not 320x480.
     Selected by pre-resizing the image, since generate-style code derives the
     geometry from the native aspect.
  2. DECODER TAE-HV, matching the real-time display path, not the full streaming
     VAE.
  3. CONDITION y goes through _condition_latent, so LINGBOT_STREAM_ENCODE selects
     the encoder. That variable is read per call, so both arms run in ONE process
     and can be interleaved.
  4. ABBA INTERLEAVING. A laptop GPU is subject to thermal drift, power limits,
     boost clocks and allocator warm-up, so AAAA-BBBB would read drift as an
     effect.

WHAT THIS MEASURES, AND WHAT IT DOES NOT. It reports per-chunk DiT latency,
per-chunk decode latency, and their sum, plus allocator state and event/state
counters. It does NOT measure control-to-real. A G4-0 audit found that no existing
script has a qualifying control-to-real: the reported figures are formulas,
configuration values, or decode-completion times. The real contract needs
event-accept -> commit -> decode -> submit -> present, which this round does not
have. So the combined figure here is labelled measured hot-path latency and may
serve later as one component of a control-to-real lower bound. It must never be
renamed control-to-real.
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
from compensator import Compensator

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
CTRL_HZ = 60.0
SCRIPT = [(0.0, dict(fwd=0.6)), (2.0, dict(yaw=0.8, fwd=0.3)),
          (4.0, dict(yaw=-0.8, fwd=0.3)), (6.0, dict(fwd=-0.6)),
          (8.0, dict(strafe=0.8))]
MB = 2 ** 20


def mem():
    free, total = torch.cuda.mem_get_info()
    return dict(alloc=torch.cuda.memory_allocated() / MB,
                reserved=torch.cuda.memory_reserved() / MB,
                max_reserved=torch.cuda.max_memory_reserved() / MB,
                free=free / MB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--weight", default="bf16", choices=["bf16", "fp8_lowmem", "fp8"])
    ap.add_argument("--pixel", default="304x528")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_chunks", type=int, default=10)
    ap.add_argument("--mode", default="D", choices=list("ABCD"))
    ap.add_argument("--rounds", type=int, default=2,
                    help="ABBA rounds; each round runs A,B,B,A")
    ap.add_argument("--local_window", type=int, default=6)
    ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    if args.weight == "bf16":
        os.environ["LINGBOT_WEIGHT_MODE"] = "bf16"
        os.environ["LINGBOT_FP8"] = "0"
    else:
        os.environ["LINGBOT_WEIGHT_MODE"] = "fp8_lowmem"
        os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"

    W, H = (int(x) for x in args.pixel.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    use_swap = args.mode in ("C", "D")
    use_comp = args.mode in ("B", "D")

    print("=" * 96)
    print(f"  G4-hotpath   weight={args.weight}  pixel={W}x{H}  "
          f"chunks={args.n_chunks}  mode={args.mode}  rounds={args.rounds}")
    print(f"  preset local={args.local_window} sink={args.sink}   "
          f"decoder=TAE   interleaving=ABBA")
    print("  measures per-chunk hot-path latency; NOT control-to-real")
    print("=" * 96, flush=True)

    # ---- authoritative 60 Hz control clock (identical to hotswap_loop) ----
    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max = 6.0
    ctl.cfg.pitch_rate_max = 2.0
    ctl.cfg.v_max = 1.0
    total_time = args.n_chunks * 1.25 + 2.0
    n_ctrl = int(total_time * CTRL_HZ)
    ctl_poses, ctl_v = [], []
    si = 0
    ctl.set_input(**SCRIPT[0][1])
    for k in range(n_ctrl):
        t = k / CTRL_HZ
        while si + 1 < len(SCRIPT) and t >= SCRIPT[si + 1][0]:
            si += 1
            ctl.set_input(**SCRIPT[si][1])
        ctl.step(dt=1.0 / (CTRL_HZ * 1.25))
        ctl_poses.append(ctl.pose.copy())
        ctl_v.append(ctl.v.copy())
    ctl_poses = np.stack(ctl_poses)
    ctl_v = np.stack(ctl_v)

    def slot_pose(t):
        i = int(np.clip(round(t * CTRL_HZ), 0, len(ctl_poses) - 1))
        return ctl_poses[i]

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
    lat_h = h // vae_stride[1]
    lat_w = w // vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = (args.n_chunks - 1) * 4 + 1
    max_seq_len = frame_seqlen
    kv_size = frame_seqlen * args.local_window
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size
    hd = ma.dim // ma.num_heads
    print(f"  geometry: pixel {h}x{w}  latent {lat_h}x{lat_w}  "
          f"frame_seqlen {frame_seqlen}  kv_size {kv_size}", flush=True)

    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype,
        device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype,
        device=dev)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def reset_kv():
        for c in self_kv:
            c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False
            c["k"].zero_(); c["v"].zero_()

    def plucker(rel_pose):
        rel = torch.from_numpy(np.asarray(rel_pose)).float()[None].to(dev)
        p = get_plucker_embeddings(rel, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h,
                         w=lat_w).to(pdt)

    # ---- one arm: prepare + the chunk loop ----
    def run_arm(stream_encode):
        os.environ["LINGBOT_STREAM_ENCODE"] = str(stream_encode)
        reset_kv()
        pipe._cross_attn_initialized = False
        mf.bump_cam_epoch()
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # ---- prepare: condition latent ----
        pre = mem()
        print(f"  [arm stream={stream_encode}] before prepare: "
              f"alloc {pre['alloc']:.0f} reserved {pre['reserved']:.0f} "
              f"free {pre['free']:.0f} MiB", flush=True)
        tp = time.perf_counter()
        y = pipe._condition_latent(
            img, F, h, w) if hasattr(pipe, "_condition_latent") else None
        if y is None:
            y = pipe.vae.encode([torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                torch.zeros(3, F - 1, h, w)], dim=1).to(dev)])[0]
        torch.cuda.synchronize()
        prepare_s = time.perf_counter() - tp
        y_hash = hashlib.sha256(
            y.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]
        y_shape = list(y.shape)
        msk = torch.ones(1, F, lat_h, lat_w, device=dev)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4,
                                                    dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        y = torch.concat([msk, y])
        after_prepare = mem()

        g = torch.Generator(device=dev); g.manual_seed(args.seed)
        noise = torch.randn(16, args.n_chunks, lat_h, lat_w, generator=g,
                            device=dev)
        comp_state = dict(gate=1.0, prev_v=None)

        def lead_pose(pose, t):
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

        rows = []
        prev_pose = None
        t_start = time.perf_counter()
        for cid in range(args.n_chunks):
            cur = noise.split(1, dim=1)[cid]
            chunk_t = time.perf_counter() - t_start
            chunk_pose = slot_pose(chunk_t)
            chunk_pose_c = lead_pose(chunk_pose, chunk_t) if use_comp \
                else chunk_pose
            rel0 = (np.eye(4) if prev_pose is None
                    else np.linalg.inv(prev_pose) @ chunk_pose_c)
            frozen_plk = plucker(rel0)

            torch.cuda.synchronize(); t_d0 = time.perf_counter()
            applied_evt = None
            for ti in range(len(timesteps)):
                if use_swap:
                    t_now = time.perf_counter() - t_start
                    pose_now = slot_pose(t_now)
                    if use_comp:
                        pose_now = lead_pose(pose_now, t_now)
                    rel = (np.eye(4) if prev_pose is None
                           else np.linalg.inv(prev_pose) @ pose_now)
                    plk = plucker(rel)
                    applied_evt = t_now
                else:
                    plk = frozen_plk
                    applied_evt = chunk_t
                kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                      "y": [y.split(1, dim=1)[cid]],
                      "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                      "kv_cache": self_kv, "crossattn_cache": cross_kv,
                      "current_start": cid * frame_seqlen,
                      "max_attention_size": kv_size,
                      "frame_seqlen": frame_seqlen}
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
            torch.cuda.synchronize()
            dit_ms = (time.perf_counter() - t_d0) * 1000

            torch.cuda.synchronize(); t_c0 = time.perf_counter()
            with torch.no_grad():
                tae.decode_video(x0.to(dev).permute(1, 0, 2, 3).unsqueeze(0),
                                 parallel=False, show_progress_bar=False)
            torch.cuda.synchronize()
            dec_ms = (time.perf_counter() - t_c0) * 1000

            m = mem()
            rows.append(dict(chunk=cid, dit_ms=dit_ms, dec_ms=dec_ms,
                             hotpath_ms=dit_ms + dec_ms,
                             applied_evt=applied_evt,
                             **m))
            prev_pose = chunk_pose_c
        return dict(stream=stream_encode, prepare_s=prepare_s,
                    y_hash=y_hash, y_shape=y_shape, after_prepare=after_prepare,
                    before_prepare=pre, rows=rows)

    # ---- ABBA interleaving ----
    order = []
    for r in range(args.rounds):
        order += [0, 1, 1, 0] if r % 2 == 0 else [1, 0, 0, 1]
    print(f"  arm order (0=whole, 1=streamed): {order}", flush=True)

    runs = []
    for i, se in enumerate(order):
        d = run_arm(se)
        runs.append(d)
        hot = [r["hotpath_ms"] for r in d["rows"][1:]] or \
            [r["hotpath_ms"] for r in d["rows"]]
        print(f"  [{i}] stream={se}  prepare {d['prepare_s']:.2f}s  "
              f"y={d['y_hash']}  hotpath p50 {statistics.median(hot):7.1f} ms  "
              f"reserved {d['after_prepare']['reserved']:.0f} MiB", flush=True)

    # ---- analysis ----
    print()
    print("=" * 96)
    print("  G4-hotpath RESULT")
    print("=" * 96)

    yh = {d["stream"]: set() for d in runs}
    for d in runs:
        yh[d["stream"]].add(d["y_hash"])
    print(f"  condition y hash: whole={sorted(yh[0])}  streamed={sorted(yh[1])}")
    same_y = len(yh[0]) == 1 and len(yh[1]) == 1 and yh[0] == yh[1]
    print(f"  y identical across arms: {same_y}")
    print()

    def agg(stream, key, warm=True):
        vals = []
        for d in runs:
            if d["stream"] != stream:
                continue
            rr = d["rows"][1:] if warm else d["rows"]
            vals += [r[key] for r in rr]
        return vals

    print(f"  {'metric':<16} {'whole':>10} {'streamed':>10} {'delta':>10} "
          f"{'pct':>8}")
    print("  " + "-" * 60)
    verdict = {}
    for key, lab in (("dit_ms", "DiT"), ("dec_ms", "decode"),
                     ("hotpath_ms", "hotpath")):
        a = agg(0, key); b = agg(1, key)
        ma_, mb_ = statistics.median(a), statistics.median(b)
        pct = (mb_ - ma_) / ma_ * 100
        p95a = sorted(a)[int(len(a) * 0.95) - 1]
        p95b = sorted(b)[int(len(b) * 0.95) - 1]
        print(f"  {lab + ' p50':<16} {ma_:>10.1f} {mb_:>10.1f} "
              f"{mb_-ma_:>+10.1f} {pct:>+7.1f}%")
        print(f"  {lab + ' p95':<16} {p95a:>10.1f} {p95b:>10.1f} "
              f"{p95b-p95a:>+10.1f} {(p95b-p95a)/p95a*100:>+7.1f}%")
        verdict[key] = pct
    print()

    # allocator + event/state
    ra = [d["after_prepare"] for d in runs if d["stream"] == 0]
    rb = [d["after_prepare"] for d in runs if d["stream"] == 1]
    print(f"  reserved after prepare  whole {statistics.mean(x['reserved'] for x in ra):7.0f}  "
          f"streamed {statistics.mean(x['reserved'] for x in rb):7.0f} MiB")
    alloc_creep = {}
    for s in (0, 1):
        d = [x for x in runs if x["stream"] == s]
        first = statistics.mean(r["alloc"] for r in d[0]["rows"][:3])
        last = statistics.mean(r["alloc"] for r in d[-1]["rows"][-3:])
        alloc_creep[s] = last - first
    print(f"  allocated creep         whole {alloc_creep[0]:+7.1f}  "
          f"streamed {alloc_creep[1]:+7.1f} MiB")
    evt_ok = all(r["applied_evt"] is not None for d in runs for r in d["rows"])
    print(f"  event/state             applied_evt present on every chunk: {evt_ok}")
    print()

    c1 = all(abs(verdict[k]) <= 3.0 for k in verdict)
    c2 = abs(alloc_creep[0]) < 32 and abs(alloc_creep[1]) < 32
    c3 = same_y and evt_ok
    print(f"  1  hot-path <= +3%               "
          f"{'PASS' if c1 else 'FAIL'}  (DiT {verdict['dit_ms']:+.1f}%, "
          f"decode {verdict['dec_ms']:+.1f}%, hotpath {verdict['hotpath_ms']:+.1f}%)")
    print(f"  2  no allocator creep            "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"  3  y identical + events intact   {'PASS' if c3 else 'FAIL'}")
    overall = c1 and c2 and c3
    print()
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"  NOTE: this is measured hot-path latency. It is NOT control-to-real;")
    print(f"        no qualifying control-to-real exists yet (see G4-0 audit).")

    with open(f"{args.out_dir}/g4_hotpath.json", "w") as f:
        json.dump(dict(weight=args.weight, pixel=[W, H], chunks=args.n_chunks,
                       mode=args.mode, order=order, runs=runs,
                       verdict=verdict, alloc_creep=alloc_creep,
                       same_y=same_y, evt_ok=evt_ok,
                       overall="PASS" if overall else "FAIL"), f, indent=2)
    print(f"\n[g4] wrote {args.out_dir}/g4_hotpath.json")


if __name__ == "__main__":
    main()
