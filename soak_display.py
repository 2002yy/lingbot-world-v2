#!/usr/bin/env python
"""§37D: generation + adaptive display joint long run (5 min stress profile).

Runs the frozen v1 production pipeline while feeding every arriving real frame
into the online adaptive display loop, and records:

    ok / soft / reset / fail shares
    mean / p90 / max refresh jump
    mean warp gain
    dropped display frames / presentation FPS
    TAE flicker + big-pop + sustained-fail event counts
    VRAM / DiT / KV / TAE (compared against the §37 generation-only soak)

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python soak_display.py --profile B --minutes 5
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from cam_controller import CameraController
from eval_two_layer import load_models, lpips_d
from pose_warp import PoseWarper
from reset_policy import predict_lpips
from wan.utils.cam_utils import get_Ks_transformed, get_plucker_embeddings
from einops import rearrange

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
FPS, CADENCE = 60.0, 1.25
D = int(FPS * CADENCE)
NAMES = ["ok", "soft", "reset", "fail"]
DEESCALATE_HOLD = 0.35
CFG = dict(soft=0.30, reset=0.40, fail=0.55)
BLEND = 6

PROFILES = {
    "B": (0.9, [dict(yaw=0.9), dict(yaw=-0.9), dict(strafe=0.9),
                dict(strafe=-0.9), dict(fwd=0.8), dict(fwd=-0.8),
                dict(fwd=0.7, yaw=-0.7), dict(fwd=0.7, yaw=0.7),
                dict(fwd=-0.7, yaw=0.7), dict()]),
}


def level_of(pred, valid):
    if valid < 0.90 or pred >= CFG["fail"]:
        return 3
    if pred >= CFG["reset"]:
        return 2
    if pred >= CFG["soft"]:
        return 1
    return 0


def warp_gain(pred, state):
    if state in ("ok", "soft"):
        return 1.0
    if state == "reset":
        span = max(CFG["fail"] - CFG["reset"], 1e-6)
        return float(np.clip(1.0 - (pred - CFG["reset"]) / span * 0.5, 0.5, 1.0))
    return float(np.clip(0.5 - (pred - CFG["fail"]) / 0.30 * 0.35, 0.15, 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", default="B", choices=list(PROFILES))
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/soak_display")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.profile}_{int(args.minutes)}min_s{args.seed}"
    rng = np.random.default_rng(args.seed)
    hold, menu = PROFILES[args.profile]
    total_t = args.minutes * 60.0
    timeline, t = [], 0.0
    while t < total_t + 5.0:
        timeline.append((t, menu[rng.integers(0, len(menu))]))
        t += hold * (0.5 + rng.random())
    print(f"[sd] profile {args.profile}, {args.minutes}min, "
          f"{len(timeline)} input segments", flush=True)

    base = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base[:3, :3], base[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    n_ctrl = int((total_t + 8.0) * FPS)
    cps, cvs, gates = [], [], []
    si, prev_v, gate = 0, None, 1.0
    ctl.set_input(**timeline[0][1])
    for k in range(n_ctrl):
        tt = k / FPS
        while si + 1 < len(timeline) and tt >= timeline[si + 1][0]:
            si += 1
            ctl.set_input(**timeline[si][1])
        ctl.step(dt=1.0 / (FPS * CADENCE))
        v = ctl.v.copy()
        n_now = float(np.linalg.norm(v))
        if prev_v is None or n_now < 0.02:
            tgt = 1.0 if prev_v is None else 0.0
        else:
            cos = float(np.dot(prev_v, v) / max(np.linalg.norm(prev_v) * n_now, 1e-9))
            tgt = 0.0 if cos <= 0 else float(min(1.0, cos / 0.7))
        gate = tgt if tgt < gate else gate + 0.25 * (tgt - gate)
        prev_v = v
        T = np.eye(4); T[:3, 3] = v * (2.0 * gate)
        cps.append(ctl.pose @ T); cvs.append(v); gates.append(gate)
    cps = np.stack(cps)

    def slot(tt):
        return cps[int(np.clip(round(tt * FPS), 0, len(cps) - 1))]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[sd] pipe + TAE built", flush=True)
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
    h = lat_h * vae_stride[1]; w = lat_w * vae_stride[2]
    frame_seqlen = (lat_h * lat_w) // (patch[1] * patch[2])
    F = 81
    max_seq_len = int(math.ceil(frame_seqlen / pipe.sp_size)) * pipe.sp_size
    kv_size = frame_seqlen * 6
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads
    pipe.prewarm(img_pil, max_area=W * H, frame_num=F, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{args.base}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    Ksm = Ks.cpu().numpy()
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
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[sd] VAE dropped, free={torch.cuda.mem_get_info()[0]/2**20:.0f}MiB",
          flush=True)
    # LPIPS is loaded AFTER generation to avoid co-residency pressure
    _lpips = None
    _frames = []
    _poses = []
    _gens = []

    def lpips_d(a, b):
        t = lambda x: torch.from_numpy(x).float().permute(2, 0, 1)[None].to(dev) / 127.5 - 1.0
        with torch.no_grad():
            return float(_lpips(t(a), t(b)).item())

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh, hd], dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd], dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
    g = torch.Generator(device=dev); g.manual_seed(args.seed)

    def plucker(rel):
        r = torch.from_numpy(np.asarray(rel)).float()[None].to(dev)
        p = get_plucker_embeddings(r, Ks[None], h, w)
        p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                      c1=int(h // lat_h), c2=int(w // lat_w))[None]
        return rearrange(p, 'b (f h w) c -> b c f h w', f=1, h=lat_h, w=lat_w).to(pdt)

    # ---- display state ----
    warper = PoseWarper(Ksm, H, W, nominal_depth=4.0)
    st_state, st_change = "ok", 0.0
    ref_img = ref_pose = None
    blend_left, blend_src = 0, None
    rows, state_counts = [], {s: 0 for s in NAMES}
    _pairs = []
    jumps, gains, fail_run, fail_max_run = [], [], 0, 0
    pop_events, flicker_events = 0, 0
    dropped = 0
    prev_pose = None
    t0 = time.perf_counter()
    cid = 0
    deadline = args.minutes * 60.0
    while (time.perf_counter() - t0) < deadline:
        tc = time.perf_counter()
        g.manual_seed(args.seed + cid)
        cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
        kw = None
        for ti in range(len(timesteps)):
            now = time.perf_counter() - t0
            pn = slot(now)
            rel = np.eye(4) if prev_pose is None else np.linalg.inv(prev_pose) @ pn
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[min(cid, F // 4 - 1)]],
                  "dit_cond_dict": {"c2ws_plucker_emb": plucker(rel).chunk(1, dim=0)},
                  "kv_cache": self_kv, "crossattn_cache": cross_kv,
                  "current_start": cid * frame_seqlen,
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
        t_dit = time.perf_counter() - tc
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        t_kv = time.perf_counter() - tc - t_dit
        t1 = time.perf_counter()
        with torch.no_grad():
            fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                  parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        t_tae = time.perf_counter() - t1
        gen_ms = (time.perf_counter() - tc) * 1000
        # dropped display frames if generation overran the chunk period
        over = max(0.0, gen_ms - CADENCE * 1000)
        dropped += int(over / (1000 / FPS))
        real = (fr[0][0].permute(1, 2, 0).float().cpu().numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)
        if real.shape[:2] != (H, W):
            real = cv2.resize(real, (W, H))

        # ---- display for this chunk period ----
        chunk_t0 = time.perf_counter() - t0
        t_now = time.perf_counter() - t0
        pose_now = slot(t_now)
        new_ref = (cid != 0)
        if new_ref:
            blend_left = BLEND
            blend_src = None
        ref_img, ref_pose = real, pose_now
        warper.set_reference(ref_img, ref_pose)
        prev_disp = None
        for f in range(D):
            tf = chunk_t0 + f / FPS
            p = slot(tf)
            Hm = warper.homography(p)
            pts = np.stack([np.arange(0, W, 8).repeat(len(range(0, H, 8))),
                            np.tile(np.arange(0, H, 8), len(range(0, W, 8))),
                            np.ones(len(range(0, W, 8)) * len(range(0, H, 8)))], 0).astype(np.float64)
            q = Hm @ pts
            q = q[:2] / q[2]
            disp = float(np.mean(np.linalg.norm(q - pts[:2], axis=0)))
            mask = cv2.warpPerspective(np.ones((H, W), np.uint8) * 255, Hm, (W, H),
                                       flags=cv2.INTER_NEAREST,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            valid = float((mask > 127).mean())
            tr = np.linalg.inv(ref_pose) @ p
            yaw = np.degrees(np.arccos(np.clip((np.trace(tr[:3, :3]) - 1) / 2, -1, 1)))
            pred = predict_lpips(disp, yaw)
            lvl = level_of(pred, valid)
            cur_i = NAMES.index(st_state)
            if lvl > cur_i:
                st_state, st_change = NAMES[lvl], tf
            elif lvl < cur_i and (tf - st_change) > DEESCALATE_HOLD:
                st_state, st_change = NAMES[cur_i - 1], tf
            state_counts[st_state] += 1
            if st_state == "fail":
                fail_run += 1
                fail_max_run = max(fail_max_run, fail_run)
            else:
                fail_run = 0
            gain = warp_gain(pred, st_state)
            gains.append(gain)
            out = warper.warp(p) if gain >= 1.0 else warper.warp(p)
            if blend_left > 0 and prev_disp is not None:
                a = 1.0 - blend_left / (BLEND + 1.0)
                out = cv2.addWeighted(prev_disp, 1.0 - a, out, a, 0.0)
                blend_left -= 1
            if prev_disp is not None and f % 15 == 0:
                _pairs.append((prev_disp.copy(), out.copy()))
            prev_disp = out
        rows.append(dict(chunk=cid, dit=t_dit, kv=t_kv, tae=t_tae, gen_ms=gen_ms,
                         state=st_state, free=torch.cuda.mem_get_info()[0] / 2**20))
        if cid % 20 == 0:
            tot = sum(state_counts.values())
            print(f"[sd] c{cid:4d} dit={t_dit*1000:5.0f} tae={t_tae*1000:4.0f} "
                  f"gen={gen_ms:5.0f}ms free={rows[-1]['free']:5.0f} "
                  f"state={st_state:5s} gain={np.mean(gains[-D:]):.2f} "
                  f"fail%={100*state_counts['fail']/max(tot,1):4.1f}", flush=True)
        prev_pose = pose_now
        cid += 1

    # ---- stage 2: LPIPS refresh jumps, with the generator freed ----
    del pipe, tae
    gc.collect(); torch.cuda.empty_cache()
    import lpips as _lp
    _lpips = _lp.LPIPS(net='alex').eval().to(dev)
    for (a, b) in _pairs:
        jv = lpips_d(cv2.cvtColor(a, cv2.COLOR_BGR2RGB),
                     cv2.cvtColor(b, cv2.COLOR_BGR2RGB))
        jumps.append(jv)
        if jv > 0.55:
            pop_events += 1
    print(f"[sd] computed {len(jumps)} LPIPS jump samples", flush=True)

    tot = sum(state_counts.values())
    j = np.array(jumps) if jumps else np.array([0.0])
    print(f"\n[sd] ===== §37D joint run: profile {args.profile}, {args.minutes}min =====")
    print(f"  chunks completed       {len(rows)}")
    print(f"  wall time              {rows[-1]['chunk'] and (time.perf_counter()-t0):.0f}s")
    print(f"  DiT / KV / TAE         {np.mean([r['dit'] for r in rows[3:]])*1000:.0f} / "
          f"{np.mean([r['kv'] for r in rows[3:]])*1000:.0f} / "
          f"{np.mean([r['tae'] for r in rows[3:]])*1000:.0f} ms")
    print(f"  driver free            start={rows[0]['free']:.0f} "
          f"end={rows[-1]['free']:.0f} min={min(r['free'] for r in rows):.0f} MiB")
    print(f"  ---- adaptive display ----")
    for s in NAMES:
        print(f"  {s:6s} {100*state_counts[s]/max(tot,1):5.1f}%")
    print(f"  mean warp gain         {np.mean(gains):.3f}")
    print(f"  refresh jump           mean={j.mean():.4f} p90={np.percentile(j,90):.4f} "
          f"max={j.max():.4f}")
    print(f"  dropped display frames {dropped}")
    print(f"  presentation FPS       {1000*D/ (np.mean([r['gen_ms'] for r in rows[3:]])):.1f} "
          f"(budget {FPS:.0f})")
    print(f"  pop events (>0.55)     {pop_events}")
    print(f"  sustained fail (max)   {fail_max_run} display frames "
          f"({fail_max_run/FPS:.2f}s)")
    ok = (fail_max_run < 2 * FPS) and (pop_events < 0.02 * max(len(jumps), 1))
    print(f"\n  GATE: sustained fail <2s = {fail_max_run < 2*FPS}; "
          f"pop rate <2% = {pop_events < 0.02*max(len(jumps),1)}")
    print(f"  §37D {'PASS' if ok else 'REVIEW'}")
    json.dump(dict(rows=rows, state_counts=state_counts, jumps=[float(x) for x in j],
                   gains=gains, dropped=dropped, pop_events=pop_events,
                   fail_max_run=fail_max_run),
              open(f"{args.out_dir}/{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
