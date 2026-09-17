#!/usr/bin/env python
"""§41B-2b: motion-aware spatial prior -- decisive paired regression.

The bug: object_state.reacquire penalised the distance from the LAST SEEN
position, so a moving object accumulated an ever-growing penalty and the
matcher killed it even when the object was rendered perfectly:
    speed 2.0 P1 -> existence 100%, identity 42%
    speed 2.5 P1 -> existence 100%, identity 33%

The fix: the spatial gate is relative to the PREDICTED position of the
authoritative transform:
    uv_pred(t) = project(transform(t), camera_pose(t))
    r          = uv_candidate - uv_pred(t)
    score      = appearance_sim - lambda * (|r| / sigma)^2
with last-seen demoted to a fallback (only when there is no prediction).

Paired design: the generation runs ONCE per config and the SAME frames are
scored by TWO independent stores -- OLD (last-seen prior) vs NEW (motion-aware)
-- so the only thing that changes is the matcher.

The observe window is extended to 30 tail chunks so that even a slow object
leaves a full object width, closing the §41B trail coverage gap.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python motion_prior.py --scene 04 --seed 42 --out_chunks 12 --tail 30
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS
from cam_controller import CameraController
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange
from object_permanence import patch
from door_state_probe import make_open_ref, surround_brightness_of, \
    structural_features
from object_state import PersistentObjectStore

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
DOOR_BB = (0.70, 0.50, 0.84, 0.66)
SKY_BB = (0.05, 0.02, 0.35, 0.16)
TREES_BB = (0.86, 0.15, 1.00, 0.55)

CONFIGS = [(f"s{int(v*100):03d}_P{p}", v, p)
           for v in (0.5, 1.0, 2.0, 2.5) for p in (1, 2, 4)]


def vram_free_mb():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**20


def build_traj_tail(scene, total_out, tail):
    T = max(1, total_out // 2)
    p = np.load(f"examples/{scene}/poses.npy")
    ctl = CameraController(p[0, :3, :3], p[0, :3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    start = ctl.pose.copy()
    out = []
    ctl.set_input(yaw=1.0)
    for _ in range(T * 4):
        ctl.step(dt=0.25)
        out.append(ctl.pose.copy())
    fr = [start] * 8 + out + out[::-1] + [start] * (tail * 4)
    traj = np.stack(fr)
    n = (len(traj) - 1) // 4 * 4 + 1
    return traj[:n], 2


def best_match_loc(ref_patch, img, bb, dino_np, search=0.08, steps=13):
    H, W = img.shape[:2]
    x0, y0, x1, y1 = bb
    pw, ph = int((x1 - x0) * W), int((y1 - y0) * H)
    fr = dino_np(ref_patch)
    best = (-1.0, None)
    for dy in np.linspace(-search, search, steps):
        for dx in np.linspace(-search, search, steps):
            bx = int((x0 + dx) * W); by = int((y0 + dy) * H)
            if bx < 0 or by < 0 or bx + pw > W or by + ph > H:
                continue
            cand = img[by:by + ph, bx:bx + pw]
            if cand.size == 0:
                continue
            s = float((fr * dino_np(cand)).sum())
            if s > best[0]:
                best = (s, (bx, by))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_chunks", type=int, default=12)
    ap.add_argument("--tail", type=int, default=30)
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/crecheck")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[mp] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lh_ = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads
    from eval_two_layer import load_models, dino_feat
    load_models()

    def dino_np(x):
        return dino_feat(x).detach().cpu().numpy().ravel()

    traj, ref_chunk = build_traj_tail(scene, args.out_chunks, args.tail)
    frames_n = len(traj)
    n_lat = (frames_n - 1) // 4 + 1
    revisit = [n_lat - args.tail, n_lat - args.tail + 1]
    observe = list(range(revisit[-1] + 1, n_lat))
    d = f"examples/mp_{scene}_O{args.out_chunks}_T{args.tail}"
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
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    print(f"[mp] n_lat={n_lat} ref={ref_chunk} revisit={revisit} "
          f"observe={observe[0]}..{observe[-1]} ({len(observe)} chunks = "
          f"{len(observe)*0.25:.2f}s)", flush=True)

    def build_y(first):
        with torch.no_grad():
            z = pipe.vae.encode([torch.concat([
                first, torch.zeros(3, frames_n - 1, h, w)], dim=1).to(dev)])[0]
        m = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
        m[:, 1:] = 0
        m = torch.concat([torch.repeat_interleave(m[:, 0:1], repeats=4, dim=1),
                          m[:, 1:]], dim=1)
        m = m.view(1, m.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        return torch.concat([m, z]).detach()

    y = build_y(torch.nn.functional.interpolate(
        img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1))
    ref_img = np.array(img_pil.resize((w, h), Image.BICUBIC))
    open_img = make_open_ref(ref_img, DOOR_BB, SKY_BB)
    open_t = TF.to_tensor(Image.fromarray(open_img)).sub_(0.5).div_(0.5) \
        .unsqueeze(0).transpose(0, 1)
    y_open = build_y(open_t)
    dy0, dy1 = int(DOOR_BB[1] * lat_h), int(DOOR_BB[3] * lat_h)
    dx0, dx1 = int(DOOR_BB[0] * lat_w), int(DOOR_BB[2] * lat_w)
    dh, dw = dy1 - dy0, dx1 - dx0
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[mp] setup done, free {vram_free_mb():.0f}MiB, object {dw}x{dh} latent",
          flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def roi_at(cid, v):
        t = 0.0 if cid in revisit else float(cid - revisit[-1])
        sx = -int(round(v * t))
        return dy0, dy1, dx0 + sx, dx1 + sx, sx

    def run(y_cond, anchor=None, v=0.0, period=1, max_chunks=None):
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, inj_log, lats = [], [], []
        N = n_lat if max_chunks is None else min(n_lat, max_chunks)
        for cid in range(N):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            p = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                          c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y_cond.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
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
            x0 = x0.clone()
            a0, a1, b0, b1, sx = roi_at(cid, v)
            inject = False
            if anchor is not None and (cid in revisit or cid in observe):
                inject = (cid in revisit) or \
                    ((cid - revisit[-1] - 1) % period == 0)
            if inject and 0 <= b0 and b1 <= lat_w and 0 <= a0 and a1 <= lat_h:
                x0[:, :, a0:a1, b0:b1] = anchor
            inj_log.append(inject)
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            with torch.no_grad():
                fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                      parallel=False, show_progress_bar=False)
            outs.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                         * 255.0).clip(0, 255).astype(np.uint8))
            lats.append(x0.detach().float().cpu())
        del self_kv, cross_kv
        gc.collect(); torch.cuda.empty_cache()
        return outs, inj_log, lats

    print("\n[mp] === D0 ===", flush=True)
    D0_frames, _, D0_lat = run(y, None)
    ref_frame = D0_frames[ref_chunk]
    ref_door = patch(ref_frame, DOOR_BB)
    print(f"[mp] === canonicalisation ===", flush=True)
    canon_frames, _, canon_lat = run(y_open, None, max_chunks=args.canon_chunks)
    ci = min(2, args.canon_chunks - 1)
    canon_door = patch(canon_frames[ci], DOOR_BB)
    Z_OPEN = canon_lat[ci][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    CANON_FEAT = dino_np(canon_door)

    def make_store(motion_aware):
        s = PersistentObjectStore()
        dd = s.register("door", np.eye(4), [1, 2, 0.2], anchor=dino_np(ref_door),
                        t=0.0, gameplay_state=dict(open=False,
                                                   uv=((DOOR_BB[0] + DOOR_BB[2]) / 2,
                                                       (DOOR_BB[1] + DOOR_BB[3]) / 2)))
        s.register("trees", np.eye(4), [1, 1, 1],
                   anchor=dino_np(patch(ref_frame, TREES_BB)), t=0.0,
                   gameplay_state=dict(open=False,
                                       uv=((TREES_BB[0] + TREES_BB[2]) / 2,
                                           (TREES_BB[1] + TREES_BB[3]) / 2)))
        s.set_anchor_for_state(dd.persistent_id, "closed", feature=dino_np(ref_door))
        s.set_anchor_for_state(dd.persistent_id, "open", feature=CANON_FEAT)
        s.set_state(dd.persistent_id, open=True)
        s._ma = motion_aware
        return s, dd

    def bb_of(a0, a1, b0, b1):
        return (b0 * vae_stride[2] / w, a0 * vae_stride[1] / h,
                b1 * vae_stride[2] / w, a1 * vae_stride[1] / h)

    summary = {}
    for tag, v, period in CONFIGS:
        print(f"\n[cr] === {tag}: v={v}, refresh every {period} ===", flush=True)
        F, inj, _ = run(y, Z_OPEN, v=v, period=period)
        s, dd = make_store(True)          # NEW motion-aware matcher only
        dinos, exs, poss, onscreen = [], [], [], 0
        ids, wids = [], 0
        trails, bgs, hist = [], [], []
        for cid in observe:
            a0, a1, b0, b1, sx = roi_at(cid, v)
            if not (0 <= b0 and b1 <= lat_w and 0 <= a0 and a1 <= lat_h):
                continue          # object has left the frame
            onscreen += 1
            bb = bb_of(a0, a1, b0, b1)
            fr = F[cid]
            sim, loc = best_match_loc(canon_door, fr, bb, dino_np)
            dinos.append(sim)
            exs.append(bool(sim > 0.6))
            exp = (int(bb[0] * fr.shape[1]), int(bb[1] * fr.shape[0]))
            poss.append(math.hypot(loc[0] - exp[0], loc[1] - exp[1]) if loc else 1e3)
            uv_c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            s.set_prediction(dd.persistent_id, uv=uv_c)
            dec = s.reacquire([dict(anchor=dino_np(patch(fr, bb)), uv=uv_c,
                                    world_transform=np.eye(4), bounds=[1, 2, 0.2])],
                              t=cid * 0.25, motion_aware=True)[0]
            ids.append(dec["id"] == dd.persistent_id)
            if dec["id"] not in (None, dd.persistent_id):
                wids += 1
            s.integrate(dd.persistent_id, 0.25)
            # ---- trail: a position the object centre has LEFT by >= 1 width ----
            past = None
            for c2 in observe:
                if c2 >= cid:
                    break
                px0, py0, px1, py1 = bb_of(*roi_at(c2, v)[:4])
                if abs(px0 - bb[0]) >= (DOOR_BB[2] - DOOR_BB[0]):
                    past = (px0, py0, px1, py1)
            if past:
                Hf, Wf = fr.shape[:2]
                x0_, y0_ = int(past[0] * Wf), int(past[1] * Hf)
                x1_, y1_ = int(past[2] * Wf), int(past[3] * Hf)
                if x1_ > x0_ and y1_ > y0_:
                    cur_s = float((dino_np(fr[y0_:y1_, x0_:x1_]) * CANON_FEAT).sum())
                    d0_s = float((dino_np(D0_frames[cid][y0_:y1_, x0_:x1_])
                                  * CANON_FEAT).sum())
                    trails.append(cur_s - d0_s)               # trail_excess
                    d = np.abs(fr[y0_:y1_, x0_:x1_].astype(float)
                               - D0_frames[cid][y0_:y1_, x0_:x1_].astype(float)).mean()
                    cx0, cy0 = int(0.05 * Wf), int(0.05 * Hf)
                    cw_, ch_ = x1_ - x0_, y1_ - y0_
                    c = np.abs(fr[cy0:cy0 + ch_, cx0:cx0 + cw_].astype(float)
                               - D0_frames[cid][cy0:cy0 + ch_,
                                                cx0:cx0 + cw_].astype(float)).mean()
                    bgs.append(float(d - c))                  # bg_excess
        summary[tag] = dict(
            speed=v, period=period, onscreen=onscreen,
            existence=float(np.mean(exs)), dino=float(np.mean(dinos)),
            pos_err=float(np.mean(poss)), identity=float(np.mean(ids)),
            wrong_id=wids,
            trail_excess=float(np.mean(trails)) if trails else None,
            bg_excess=float(np.mean(bgs)) if bgs else None,
            trail_n=len(trails),
            injections=int(sum(inj)))
        x = summary[tag]
        te = f"{x['trail_excess']:+.3f}" if x['trail_excess'] is not None else "n/a"
        be = f"{x['bg_excess']:+7.1f}" if x['bg_excess'] is not None else "n/a"
        print(f"    on {onscreen:2d} | exist {x['existence']*100:3.0f}% "
              f"id {x['identity']*100:3.0f}% wID {wids} pos {x['pos_err']:5.1f} "
              f"| trail_ex {te} bg_ex {be} (n={x['trail_n']}) inj {x['injections']}",
              flush=True)

    print("\n[cr] ===== §41B-2c cadence re-evaluation (NEW matcher) =====")
    print(f"  {'tag':>9s} {'onscr':>6s} {'exist':>6s} {'ident':>6s} {'wID':>4s} "
          f"{'pos':>6s} {'trail_ex':>9s} {'bg_ex':>8s} {'inj':>4s}")
    for tag, s in summary.items():
        te = f"{s['trail_excess']:+.3f}" if s['trail_excess'] is not None else "n/a"
        be = f"{s['bg_excess']:+8.1f}" if s['bg_excess'] is not None else "n/a"
        print(f"  {tag:>9s} {s['onscreen']:6d} {s['existence']*100:5.0f}% "
              f"{s['identity']*100:5.0f}% {s['wrong_id']:4d} {s['pos_err']:6.1f} "
              f"{te:>9s} {be:>8s} {s['injections']:4d}")

    # ---- lowest sufficient cadence per speed ----
    print("\n[cr] ===== lowest sufficient cadence per speed =====")
    print("  criterion: identity >= 0.95 AND existence >= 0.95 AND wrong-ID = 0")
    bysp = {}
    for tag, s in summary.items():
        bysp.setdefault(s["speed"], []).append(s)
    policy = {}
    for v in sorted(bysp):
        cands = sorted(bysp[v], key=lambda z: -z["period"])
        chosen = None
        for c in cands:
            if (c["identity"] >= 0.95 and c["existence"] >= 0.95
                    and c["wrong_id"] == 0):
                chosen = c
                break
        if chosen is None:
            chosen = min(bysp[v], key=lambda z: z["period"])
        policy[v] = chosen["period"]
        line = "  ".join(f"P{c['period']}:{c['identity']*100:.0f}%/{c['existence']*100:.0f}%"
                         for c in sorted(bysp[v], key=lambda z: z["period"]))
        print(f"  v={v:4.2f}  {line}   -> P{policy[v]}")
    bnd = [round((sorted(policy)[i - 1] + sorted(policy)[i]) / 2, 3)
           for i in range(1, len(policy))
           if policy[sorted(policy)[i]] != policy[sorted(policy)[i - 1]]]
    print(f"\n  empirical boundaries: {bnd}")
    print(f"  policy: {policy}")

    # ---- trail / background verdict ----
    print("\n[cr] ===== trail / background verdict =====")
    tes = [s["trail_excess"] for s in summary.values() if s["trail_excess"] is not None]
    if tes:
        print(f"  trail_excess over all configs: mean {np.mean(tes):+.3f} "
              f"max {np.max(tes):+.3f} min {np.min(tes):+.3f}")
        print(f"  (>0.10 would indicate visible residue; >0.25 serious)")
    bes = [s["bg_excess"] for s in summary.values() if s["bg_excess"] is not None]
    if bes:
        print(f"  bg_excess: mean {np.mean(bes):+.1f} max {np.max(bes):+.1f}")
    missing = [t for t, s in summary.items() if s["trail_excess"] is None]
    if missing:
        print(f"  coverage gap (object never left 1 object-width): {missing}")

    json.dump(dict(summary=summary, policy=policy, boundaries=bnd),
              open(f"{args.out_dir}/cadence_recheck.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()

