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

CONFIGS = [("s050_P1", 0.5, 1), ("s050_P2", 0.5, 2),
           ("s100_P1", 1.0, 1), ("s200_P1", 2.0, 1), ("s250_P1", 2.5, 1)]


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
    ap.add_argument("--out_dir", default="output/mprior")
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
        print(f"\n[mp] === {tag}: v={v}, refresh every {period} ===", flush=True)
        F, inj, _ = run(y, Z_OPEN, v=v, period=period)
        rows = []
        stores = {m: make_store(m) for m in (False, True)}
        dinos, exs, poss, onscreen = [], [], [], 0
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
            rec = {}
            for m, (s, dd) in stores.items():
                # authoritative prediction from transform(t)
                s.set_prediction(dd.persistent_id, uv=uv_c)
                dec = s.reacquire([dict(anchor=dino_np(patch(fr, bb)), uv=uv_c,
                                        world_transform=np.eye(4),
                                        bounds=[1, 2, 0.2])],
                                  t=cid * 0.25, motion_aware=m)[0]
                rec[m] = dec["id"] == dd.persistent_id
                s.integrate(dd.persistent_id, 0.25)
            rows.append(rec)
        old = float(np.mean([r[False] for r in rows]))
        new = float(np.mean([r[True] for r in rows]))
        summary[tag] = dict(speed=v, period=period, onscreen=onscreen,
                            existence=float(np.mean(exs)),
                            dino=float(np.mean(dinos)),
                            pos_err=float(np.mean(poss)),
                            identity_old=old, identity_new=new,
                            injections=int(sum(inj)))
        s_ = summary[tag]
        print(f"    on-screen {onscreen} chunks | existence {s_['existence']*100:3.0f}% "
              f"| identity OLD {old*100:3.0f}%  NEW {new*100:3.0f}%  "
              f"(delta {100*(new-old):+.0f}pp)", flush=True)

    print("\n[mp] ===== §41B-2b paired regression =====")
    print(f"  {'config':>9s} {'onscr':>6s} {'exist':>6s} {'id_OLD':>7s} "
          f"{'id_NEW':>7s} {'delta':>7s} {'inj':>4s}")
    for tag, s in summary.items():
        print(f"  {tag:>9s} {s['onscreen']:6d} {s['existence']*100:5.0f}% "
              f"{s['identity_old']*100:6.0f}% {s['identity_new']*100:6.0f}% "
              f"{100*(s['identity_new']-s['identity_old']):+6.0f}pp "
              f"{s['injections']:4d}")

    print("\n[mp] ===== GATE =====")
    print("  motion-aware prior must recover identity WITHOUT changing existence,")
    print("  and low-speed must not regress")
    ok = True
    for tag, s in summary.items():
        good = s["identity_new"] >= s["identity_old"] - 0.01
        if s["speed"] >= 1.0:
            good = good and s["identity_new"] >= 0.9
        ok = ok and good
        print(f"    {tag}: exist {s['existence']*100:.0f}% "
              f"id {s['identity_old']*100:.0f}%->{s['identity_new']*100:.0f}% "
              f"-> {'PASS' if good else 'FAIL'}")
    print(f"\n  §41B-2b: {'PASS' if ok else 'FAIL'}")

    json.dump(summary, open(f"{args.out_dir}/mprior.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
