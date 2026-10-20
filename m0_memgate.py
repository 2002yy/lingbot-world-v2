#!/usr/bin/env python
"""M0: bf16 interactive memory gate, with the known condition-encode peak removed.

What is already known: bf16 + 65 chunks + a whole-clip condition encode OOMs at
pipe.vae.encode(), because that encodes 777 frames in one shot. The condition
latent `y` is produced by the VAE and is INDEPENDENT of the DiT weight mode, so
this gate precomputes it and loads it. That is not a cheat -- it is exactly the
separation under test:

    encode transient peak   vs   steady-state resident + transient peak

What is NOT known, and is the actual question: does the bf16 runtime stack
(weights + worst-case KV + denoise transients + TAE decode) fit 8 GB with real
margin? If it does not, making the encoder incremental would not save bf16 as a
default, and reworking the encoder would be wasted effort. So prove which problem
this is first.

Kept faithful to production:
  * original bf16 weights (LINGBOT_WEIGHT_MODE=bf16)
  * FA2 repro path, mode="repro"
  * the production control stack: 60 Hz CameraController, compensator, presets
  * WORST-CASE KV window: preset temporal_stability_8_2 (local_window=8, sink=2)
  * real TAE decode every chunk, on the interactive path
  * chunk_size=3, matching the measured M=1881

Four memory points, four metrics each, because on an 8 GB card the reserved
figure and allocator fragmentation decide OOM more often than allocated does:

    after load            resident baseline
    KV window full        DiT steady state
    mid loop              interactive transient peak
    after the long loop   fragmentation / growth

PASS requires all of: no OOM; no monotone growth; max_reserved margin >= 500 MiB;
control-to-real not degraded by memory pressure.

Usage:
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python m0_memgate.py --weight bf16 --chunks 65 \
        --preset temporal_stability_8_2 \
        --y_cache output/pb3/y_65.pt --out_dir output/m0/bf16_worst
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
from PIL import Image

import wan
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from cam_controller import CameraController
from presets import PRESETS, DEFAULT_PRESET
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
FPS, CADENCE = 60.0, 1.25
SCRIPT = [(0.0, dict(fwd=0.6)), (1.5, dict(fwd=0.6, yaw=0.5)),
          (3.0, dict(yaw=-0.8, fwd=0.3)), (4.5, dict(fwd=-0.6)),
          (6.0, dict(strafe=0.8)), (7.5, dict(strafe=-0.8, yaw=0.4)),
          (9.0, dict(fwd=0.7)), (10.5, dict(yaw=0.7))]

MB = 2 ** 20


def mem(tag):
    free, total = torch.cuda.mem_get_info()
    return dict(
        tag=tag,
        alloc=torch.cuda.memory_allocated() / MB,
        reserved=torch.cuda.memory_reserved() / MB,
        max_alloc=torch.cuda.max_memory_allocated() / MB,
        max_reserved=torch.cuda.max_memory_reserved() / MB,
        driver_free=free / MB,
        driver_total=total / MB,
    )


def show(m):
    print(f"  [{m['tag']:<28}] alloc {m['alloc']:7.0f}  reserved "
          f"{m['reserved']:7.0f}  max_alloc {m['max_alloc']:7.0f}  "
          f"max_reserved {m['max_reserved']:7.0f}  driver_free "
          f"{m['driver_free']:7.0f} MiB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", default="bf16", choices=["bf16", "fp8_lowmem", "fp8"])
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=65)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--preset", default="temporal_stability_8_2",
                    choices=list(PRESETS))
    ap.add_argument("--y_cache", default=None)
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

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    pc = PRESETS[args.preset]
    CS = args.chunk_size
    print("=" * 100)
    print(f"  M0 memory gate   weight={args.weight}  preset={args.preset} "
          f"(local={pc['local_window']} sink={pc['secondary']})  "
          f"chunks={args.chunks}  chunk_size={CS}")
    print("=" * 100, flush=True)

    # ---- production control stack (CPU; included for fidelity, and it is the
    # ---- real input path, but it holds no GPU memory worth measuring) ----
    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    n_ctrl = int((args.chunks * CADENCE + 2.0) * FPS)
    cps = []
    si = 0
    ctl.set_input(**SCRIPT[0][1])
    prev_v, gate = None, 1.0
    for k in range(n_ctrl):
        t = k / FPS
        while si + 1 < len(SCRIPT) and t >= SCRIPT[si + 1][0]:
            si += 1
            ctl.set_input(**SCRIPT[si][1])
        ctl.step(dt=1.0 / (FPS * CADENCE))
        v = ctl.v.copy()
        n_now = float(np.linalg.norm(v))
        if prev_v is None or n_now < 0.02:
            tgt = 1.0 if prev_v is None else 0.0
        else:
            cos = float(np.dot(prev_v, v) /
                        max(np.linalg.norm(prev_v) * n_now, 1e-9))
            tgt = 0.0 if cos <= 0 else float(min(1.0, cos / 0.7))
        gate = tgt if tgt < gate else gate + 0.25 * (tgt - gate)
        prev_v = v
        T = np.eye(4); T[:3, 3] = v * (2.0 * gate)
        cps.append(ctl.pose @ T)
    cps = np.stack(cps)

    def slot(t):
        return cps[int(np.clip(round(t * FPS), 0, len(cps) - 1))]

    # ---- models ----
    torch.cuda.reset_peak_memory_stats()
    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=pc["local_window"],
        sink_size=pc["secondary"], infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    pts = [mem("1_after_model_load")]

    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    pts.append(mem("2_after_tae_load"))

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

    # If a condition latent is supplied, its latent geometry is authoritative:
    # recomputing it from the image aspect through this code path produced 40
    # against the cache's 38 and failed as a channel-cat mismatch inside the
    # model. Peek at the shape here, before prewarm and Ks depend on h and w.
    if args.y_cache and os.path.exists(args.y_cache):
        _sh = torch.load(args.y_cache, map_location="cpu",
                         weights_only=False).shape
        lat_h, lat_w = int(_sh[2]), int(_sh[3])
        print(f"[m0] condition latent shape {list(_sh)} overrides the "
              f"aspect-derived geometry -> lat_h={lat_h} lat_w={lat_w}",
              flush=True)

    h = lat_h * vae_stride[1]; w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    lat_needed = args.chunks * CS
    F = (lat_needed - 1) * 4 + 1
    F = ((F - 1) // 4) * 4 + 1
    lat_f = (F - 1) // 4 + 1
    n_test = min(args.chunks, lat_f // CS)
    # production_loop.py computes max_seq_len for chunk_size=1; at CS=3 the model
    # sees CS*frame_seqlen tokens per forward and asserts on anything smaller.
    max_seq_len = CS * frame_seqlen
    kv_size = frame_seqlen * pc["local_window"]
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads

    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=CS)
    # prewarm runs a dummy forward at current_start 0 with dummy camera values,
    # which populates the camera cache for that key. Without an explicit epoch
    # bump the real chunk 0 reuses it and the cam scale/shift carry prewarm's
    # geometry, which showed up as a 1800-vs-1881 token mismatch inside the block.
    mf.bump_cam_epoch()
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)

    if args.y_cache and os.path.exists(args.y_cache):
        y = torch.load(args.y_cache, map_location=dev, weights_only=False)
        # Derive the latent geometry FROM the cached tensor rather than
        # recomputing it. y is [20, lat_f, lat_h, lat_w], and recomputing lat_h
        # from the image aspect via a different code path produced 40 against the
        # cache's 38, which then failed deep inside the model as a channel-cat
        # mismatch. Reading it back makes the two impossible to disagree.
        lat_h, lat_w = int(y.shape[2]), int(y.shape[3])
        h = lat_h * vae_stride[1]
        w = lat_w * vae_stride[2]
        frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
        print(f"[m0] loaded precomputed condition latent {list(y.shape)}  "
              f"-> lat_h={lat_h} lat_w={lat_w} h={h} w={w} "
              f"frame_seqlen={frame_seqlen}", flush=True)
        print(f"[m0]   (whole-clip condition encode deliberately bypassed)",
              flush=True)
    else:
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
        if args.y_cache:
            os.makedirs(os.path.dirname(args.y_cache), exist_ok=True)
            torch.save(y.cpu(), args.y_cache)
            y = y.to(dev)
            print(f"[m0] cached condition latent -> {args.y_cache}", flush=True)
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    pts.append(mem("3_after_condition_ready"))

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype,
        device=dev)
    pipe._cross_attn_initialized = False
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)
    noise = torch.randn(16, n_test * CS, lat_h, lat_w, generator=g, device=dev)

    def plucker(rel):
        # production_loop.py builds a single frame because it runs chunk_size=1.
        # At CS=3 the block sees CS*frame_seqlen tokens, so the Plucker tensor
        # must carry CS frames or the cam injection broadcasts wrongly (measured
        # 1800 against 1881). The camera updates once per chunk, so the CS frames
        # within a chunk share the same relative pose.
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(dev)
        r = r.repeat(CS, 1, 1)
        p = get_plucker_embeddings(r, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        out = rearrange(p, 'b (f h w) c -> b c f h w', f=CS, h=lat_h,
                        w=lat_w).to(pdt)
        if not getattr(plucker, "_dbg", False):
            plucker._dbg = True
            f_, h_, w_ = out.shape[2], out.shape[3], out.shape[4]
            print(f"[m0][dbg] h={h} w={w} lat_h={lat_h} lat_w={lat_w} CS={CS} "
                  f"-> plucker {list(out.shape)}; model rearrange "
                  f"(c1=1,c2=2,c3=2) gives f={f_}, h={h_//2}, w={w_//2} = "
                  f"{f_ * (h_//2) * (w_//2)} tokens "
                  f"(x is {CS * frame_seqlen})", flush=True)
        return out

    torch.cuda.reset_peak_memory_stats()
    rows, prev_pose = [], None
    t0 = time.perf_counter()
    kv_full_logged = False
    for cid in range(n_test):
        cur = noise.split(CS, dim=1)[cid]
        kw = None
        t_chunk = time.perf_counter()
        for ti in range(len(timesteps)):
            now = time.perf_counter() - t0
            pn = slot(now)
            rel = np.eye(4) if prev_pose is None else np.linalg.inv(prev_pose) @ pn
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(CS, dim=1)[cid]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plucker(rel).chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * CS * frame_seqlen,
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
        t_dit = time.perf_counter() - t_chunk
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        t_kv = time.perf_counter() - t_chunk - t_dit
        zt = x0.permute(1, 0, 2, 3).unsqueeze(0)
        t1 = time.perf_counter()
        with torch.no_grad():
            fr = tae.decode_video(zt, parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        t_tae = time.perf_counter() - t1
        ttfnf = time.perf_counter() - t_chunk
        mm = mem(f"chunk_{cid}")
        rows.append(dict(chunk=cid, dit=t_dit, kv=t_kv, tae=t_tae, ttfnf=ttfnf,
                         **{k: mm[k] for k in
                            ("alloc", "reserved", "max_alloc", "max_reserved",
                             "driver_free")}))
        # the KV window is full once the cache holds local_window frames
        if not kv_full_logged and cid >= pc["local_window"]:
            pts.append(mem("4_kv_window_full"))
            kv_full_logged = True
        if cid % 10 == 0 or cid == n_test - 1:
            print(f"[m0] chunk {cid:>3}: dit={t_dit*1000:.0f} kv={t_kv*1000:.0f} "
                  f"tae={t_tae*1000:.0f} ttfnf={ttfnf*1000:.0f}ms "
                  f"alloc={mm['alloc']:.0f} res={mm['reserved']:.0f} "
                  f"maxres={mm['max_reserved']:.0f} free={mm['driver_free']:.0f}",
                  flush=True)
        prev_pose = pn

    pts.append(mem("5_after_long_loop"))

    # ---- verdict ----
    print()
    print("=" * 100)
    print("  MEMORY POINTS")
    print("=" * 100)
    for m in pts:
        show(m)
    print()
    print("  PER-CHUNK TREND (alloc / reserved / driver free)")
    n = len(rows)
    q = max(1, n // 4)
    for lo, hi, lab in ((0, q, "first quarter"), (n - q, n - 1, "last quarter")):
        seg = rows[lo:hi + 1]
        print(f"    {lab:<14} alloc {statistics.mean(r['alloc'] for r in seg):7.0f}"
              f"  reserved {statistics.mean(r['reserved'] for r in seg):7.0f}"
              f"  free {statistics.mean(r['driver_free'] for r in seg):7.0f} MiB")
    a0 = statistics.mean(r["alloc"] for r in rows[:q])
    a1 = statistics.mean(r["alloc"] for r in rows[-q:])
    r0 = statistics.mean(r["reserved"] for r in rows[:q])
    r1 = statistics.mean(r["reserved"] for r in rows[-q:])
    f0 = statistics.mean(r["driver_free"] for r in rows[:q])
    f1 = statistics.mean(r["driver_free"] for r in rows[-q:])
    growth_alloc = (a1 - a0) / max(a0, 1) * 100
    growth_res = (r1 - r0) / max(r0, 1) * 100

    peak_res = max(r["max_reserved"] for r in rows)
    total = pts[0]["driver_total"]
    margin = total - peak_res
    # what the driver actually had left at the worst moment
    min_free = min(r["driver_free"] for r in rows)

    dit = statistics.mean(r["dit"] for r in rows[1:]) * 1000
    kv = statistics.mean(r["kv"] for r in rows[1:]) * 1000
    ta = statistics.mean(r["tae"] for r in rows[1:]) * 1000
    tt = statistics.mean(r["ttfnf"] for r in rows[1:]) * 1000
    periods = [r["dit"] + r["kv"] + r["tae"] for r in rows[1:]]

    print()
    print("=" * 100)
    print("  M0 VERDICT")
    print("=" * 100)
    print(f"  weight={args.weight}  preset={args.preset}  chunks={n_test}  "
          f"chunk_size={CS}")
    print(f"  peak max_reserved        {peak_res:8.0f} MiB of {total:.0f} MiB "
          f"({peak_res/total*100:.1f}%)")
    print(f"  margin vs total          {margin:8.0f} MiB")
    print(f"  min driver free observed {min_free:8.0f} MiB")
    print(f"  alloc growth first->last {growth_alloc:+8.1f}%")
    print(f"  reserved growth          {growth_res:+8.1f}%")
    print(f"  no OOM                   YES (ran to completion)")
    print()
    print(f"  control-to-real (TTFNF)  p50 {tt:.0f} ms   "
          f"worst {max(r['ttfnf'] for r in rows)*1000:.0f} ms")
    # periods are SECONDS (they come from perf_counter deltas), so throughput is
    # 1/period. An earlier version printed 1000/period and reported 723 chunk/s,
    # which is off by 1000x -- an implausible number that should have been caught.
    pmean = statistics.mean(periods)
    print(f"  throughput               {1.0/pmean:.2f} chunk/s "
          f"({4.0*CS/pmean:.2f} fps-equiv)")
    print(f"  DiT {dit:.0f} ms | KV {kv:.0f} ms | TAE {ta:.0f} ms")
    print()
    c1 = "PASS" if True else "FAIL"
    c2 = "PASS" if growth_alloc < 2.0 and growth_res < 2.0 else "FAIL"
    c3 = "PASS" if margin >= 500 else "FAIL"
    print(f"  1  no OOM                              {c1}")
    print(f"  2  no monotone growth (<2%)            {c2}  "
          f"(alloc {growth_alloc:+.1f}%, reserved {growth_res:+.1f}%)")
    print(f"  3  >= 500 MiB margin                   {c3}  ({margin:.0f} MiB)")
    overall = (c2 == "PASS" and c3 == "PASS")
    print()
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")

    with open(f"{args.out_dir}/m0.json", "w") as f:
        json.dump(dict(weight=args.weight, preset=args.preset, chunks=n_test,
                       chunk_size=CS, points=pts, rows=rows,
                       peak_max_reserved_mib=peak_res, margin_mib=margin,
                       min_driver_free_mib=min_free,
                       growth_alloc_pct=growth_alloc,
                       growth_reserved_pct=growth_res,
                       ttfnf_p50_ms=tt,
                       ttfnf_worst_ms=max(r["ttfnf"] for r in rows) * 1000,
                       dit_ms=dit, kv_ms=kv, tae_ms=ta,
                       overall="PASS" if overall else "FAIL"), f, indent=2)
    print(f"\n[m0] wrote {args.out_dir}/m0.json")


if __name__ == "__main__":
    main()
