#!/usr/bin/env python
"""§41B: Dynamic Transform Authority -- can a MOVING object stay rendered?

§41A proved "it can be moved" (relocation). §41B asks "can it MOVE": while
ObjectState.transform(t) changes continuously, does LingBot keep rendering the
SAME object along the trajectory, without trails, duplicates, snapping back to
the old position, or identity loss?

Two rates are deliberately kept SEPARATE:
    transform   updated every chunk (authoritative simulation output)
    anchor      injected every 1 / 2 / 4 visible chunks (render binding)

Matrix: 3 speeds x 3 refresh cadences, constant-velocity TRANSLATION ONLY
(fixed orientation -- rotation is §41B-2, so a failure can be localised).

Speed is expressed in object-widths per 4 chunks (object width = 8 latent):
    slow   = 0.25 width / 4 chunks  ->  0.5 latent / chunk
    medium = 0.50                  ->  1.0
    fast   = 1.00                  ->  2.0

Seven primary metrics:
    1 identity retention      5 trail / duplicate score
    2 wrong-ID                6 background recovery
    3 trajectory / pos error  7 accumulated collateral
    4 existence
plus anchor injections / second.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python dyn_transform.py --scene 04 --seed 42 --out_chunks 24 --tail 14
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

SPEEDS = {"slow": 0.5, "med": 1.0, "fast": 2.0}    # latent units / chunk
PERIODS = {"P1": 1, "P2": 2, "P4": 4}


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
    ap.add_argument("--out_chunks", type=int, default=24)
    ap.add_argument("--tail", type=int, default=14)
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--speeds", default="slow,med,fast")
    ap.add_argument("--periods", default="P1,P2,P4")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/dyn")
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
    print("[dy] pipe + TAE built", flush=True)
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
    d = f"examples/dy_{scene}_O{args.out_chunks}_T{args.tail}"
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
    print(f"[dy] n_lat={n_lat} ref={ref_chunk} revisit={revisit} "
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
    print(f"[dy] setup done, free {vram_free_mb():.0f}MiB, "
          f"object {dw}x{dh} latent", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def roi_at(cid, v):
        """Authoritative transform(t) projected to a latent ROI (moves LEFT)."""
        if cid in revisit:
            t = 0.0
        elif cid in observe:
            t = float(cid - revisit[-1])          # 1..len(observe)
        else:
            t = 0.0
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
        outs, latents, inj_log = [], [], []
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
                if cid in revisit:
                    inject = True
                else:
                    inject = ((cid - revisit[-1] - 1) % period == 0)
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
            latents.append(x0.detach().float().cpu())
        del self_kv, cross_kv
        gc.collect(); torch.cuda.empty_cache()
        return outs, latents, inj_log

    # ---------- D0 + anchors ----------
    print("\n[dy] === D0 (no injection) ===", flush=True)
    D0_frames, D0_lat, _ = run(y, None)
    ref_frame = D0_frames[ref_chunk]
    ref_door = patch(ref_frame, DOOR_BB)
    z_closed = D0_lat[ref_chunk][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    print(f"\n[dy] === canonicalisation ({args.canon_chunks} chunks) ===", flush=True)
    canon_frames, canon_lat, _ = run(y_open, None, max_chunks=args.canon_chunks)
    ci = min(2, len(canon_lat) - 1)
    z_open = canon_lat[ci][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    canon_door = patch(canon_frames[ci], DOOR_BB)

    store = PersistentObjectStore()
    door = store.register("door", np.eye(4), [1, 2, 0.2], anchor=dino_np(ref_door),
                          t=0.0, gameplay_state=dict(open=False,
                                                     uv=((DOOR_BB[0] + DOOR_BB[2]) / 2,
                                                         (DOOR_BB[1] + DOOR_BB[3]) / 2)))
    store.register("trees", np.eye(4), [1, 1, 1],
                   anchor=dino_np(patch(ref_frame, TREES_BB)), t=0.0,
                   gameplay_state=dict(open=False,
                                       uv=((TREES_BB[0] + TREES_BB[2]) / 2,
                                           (TREES_BB[1] + TREES_BB[3]) / 2)))
    store.set_anchor_for_state(door.persistent_id, "closed",
                               latent=z_closed.cpu().numpy(), feature=dino_np(ref_door))
    store.set_anchor_for_state(door.persistent_id, "open",
                               latent=z_open.cpu().numpy(), feature=dino_np(canon_door))
    store.set_state(door.persistent_id, open=True)
    # §41B: authoritative motion. v latent/chunk -> world units/s
    #   1 chunk = 0.25 s; 1 latent unit = 8 px on screen (not world), so we
    #   express the motion in the projected domain and let integrate() advance
    #   the world transform with a nominal unit velocity.
    store.set_motion(door.persistent_id, velocity=np.array([1.0, 0.0, 0.0]))

    def bb_of(a0, a1, b0, b1):
        return (b0 * vae_stride[2] / w, a0 * vae_stride[1] / h,
                b1 * vae_stride[2] / w, a1 * vae_stride[1] / h)

    results = {"D0": D0_frames}
    summary = {}
    for sname in args.speeds.split(","):
        v = SPEEDS[sname]
        for pname in args.periods.split(","):
            period = PERIODS[pname]
            tag = f"{sname}_{pname}"
            print(f"\n[dy] === {tag}: v={v} latent/chunk, refresh every "
                  f"{period} chunk ===", flush=True)
            f, _, inj = run(y, z_open, v=v, period=period)
            results[tag] = f

            # ---- 7 metrics over the observe window ----
            dinos, poss, trails, bgs, exs, ids, wids = [], [], [], [], [], [], 0
            hist = []          # ROIs the object has already left
            for cid in observe:
                a0, a1, b0, b1, sx = roi_at(cid, v)
                bb = bb_of(a0, a1, b0, b1)
                fr_ = f[cid]
                sim, loc = best_match_loc(canon_door, fr_, bb, dino_np)
                dinos.append(sim)
                exp = (int(bb[0] * fr_.shape[1]), int(bb[1] * fr_.shape[0]))
                poss.append(math.hypot(loc[0] - exp[0], loc[1] - exp[1])
                            if loc else 1e3)
                # trail: how object-like are the PREVIOUS positions
                tr = 0.0
                for (ha0, ha1, hb0, hb1) in hist:
                    hbb = bb_of(ha0, ha1, hb0, hb1)
                    tr = max(tr, float((dino_np(patch(fr_, hbb)) *
                                        dino_np(canon_door)).sum()))
                trails.append(tr)
                hist.append((a0, a1, b0, b1))
                # background recovery: how close the vacated ROI is to D0
                if hist[:-1]:
                    ha0, ha1, hb0, hb1 = hist[-2]
                    hbb = bb_of(ha0, ha1, hb0, hb1)
                    Hf, Wf = fr_.shape[:2]
                    x0_, y0_ = int(hbb[0] * Wf), int(hbb[1] * Hf)
                    x1_, y1_ = int(hbb[2] * Wf), int(hbb[3] * Hf)
                    d0_ = D0_frames[cid][y0_:y1_, x0_:x1_].astype(float)
                    bgs.append(float(np.abs(fr_[y0_:y1_, x0_:x1_].astype(float)
                                            - d0_).mean()))
                else:
                    bgs.append(0.0)
                sb = surround_brightness_of(fr_, bb)
                exs.append(bool(sim > 0.6))
                cand = dict(anchor=dino_np(patch(fr_, bb)),
                            uv=((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2),
                            world_transform=np.eye(4), bounds=[1, 2, 0.2])
                dec = store.reacquire([cand], t=cid * 0.25)[0]
                ids.append(dec["id"] == door.persistent_id)
                if dec["id"] not in (None, door.persistent_id):
                    wids += 1
                store.integrate(door.persistent_id, 0.25)   # authoritative step
            # accumulated collateral vs D0 outside the CURRENT roi
            acc = []
            for cid in observe:
                a0, a1, b0, b1, sx = roi_at(cid, v)
                bb = bb_of(a0, a1, b0, b1)
                Hf, Wf = f[cid].shape[:2]
                diff = np.abs(f[cid].astype(float) - D0_frames[cid].astype(float)).mean(-1)
                m = np.ones((Hf, Wf), bool)
                m[int(bb[1] * Hf):int(bb[3] * Hf), int(bb[0] * Wf):int(bb[2] * Wf)] = False
                acc.append(float(diff[m].mean()))
            summary[tag] = dict(
                speed=v, period=period,
                identity=float(np.mean(ids)), wrong_id=wids,
                existence=float(np.mean(exs)),
                pos_err=float(np.mean(poss)), dino=float(np.mean(dinos)),
                trail=float(np.mean(trails)),
                bg_recovery=float(np.mean(bgs)),
                collateral=float(np.mean(acc)),
                injections=int(sum(inj)),
                inject_per_s=sum(inj) / (len(observe) * 0.25))
            s = summary[tag]
            print(f"    identity {s['identity']*100:3.0f}%  wrongID {wids}  "
                  f"exist {s['existence']*100:3.0f}%  pos_err {s['pos_err']:5.1f}  "
                  f"trail {s['trail']:.3f}  bg_rec {s['bg_recovery']:5.1f}  "
                  f"collat {s['collateral']:5.2f}  inj {s['injections']} "
                  f"({s['inject_per_s']:.1f}/s)", flush=True)

    print("\n[dy] ===== §41B Dynamic Transform =====")
    print(f"  {'tag':>9s} {'identity':>8s} {'wID':>4s} {'exist':>6s} "
          f"{'pos_err':>8s} {'trail':>7s} {'bg_rec':>7s} {'collat':>7s} "
          f"{'inj':>4s} {'inj/s':>6s}")
    for tag, s in summary.items():
        print(f"  {tag:>9s} {s['identity']*100:7.0f}% {s['wrong_id']:4d} "
              f"{s['existence']*100:5.0f}% {s['pos_err']:8.1f} {s['trail']:7.3f} "
              f"{s['bg_recovery']:7.1f} {s['collateral']:7.2f} "
              f"{s['injections']:4d} {s['inject_per_s']:6.1f}")

    print("\n[dy] ===== §41B GATE =====")
    print("  motion 3s: same persistent_id at the predicted position,")
    print("  wrong-ID 0, no trail/duplicate, background recovers,")
    print("  and a cadence BELOW every-chunk is usable")
    ok_any = []
    for tag, s in summary.items():
        ok = (s["identity"] >= 0.9 and s["wrong_id"] == 0
              and s["existence"] >= 0.8 and s["trail"] < 0.55
              and s["bg_recovery"] < 25.0)
        if ok and s["period"] > 1:
            ok_any.append((tag, s["injections"]))
        print(f"    {tag}: identity {s['identity']*100:.0f}% wrongID {s['wrong_id']} "
              f"exist {s['existence']*100:.0f}% trail {s['trail']:.3f} "
              f"bg_rec {s['bg_recovery']:.1f} inj {s['injections']} "
              f"-> {'PASS' if ok else 'FAIL'}")
    if ok_any:
        best = min(ok_any, key=lambda x: x[1])
        print(f"\n  §41B: PASS. Lowest sufficient cadence = {best[0]} "
              f"({best[1]} injections)")
    else:
        print("\n  §41B: FAIL (no sub-every-chunk cadence sustains motion)")

    json.dump(summary, open(f"{args.out_dir}/dyn.json", "w"), indent=1, default=float)
    for tag, frames in results.items():
        np.save(f"{args.out_dir}/{tag}_frames.npy", np.stack(frames))


if __name__ == "__main__":
    main()
