#!/usr/bin/env python
"""§40D-D2: close the loop. Object first, then state.

Chain (locked):
    1. ObjectStore holds door_01  (ID + transform + visual_anchor + state=OPEN)
    2. at revisit, project the expected ROI
    3. identity anchor injection        -> door_01 reappears
    4. §39 reacquire                    -> must match the ORIGINAL persistent_id
    5. §40B display authority           -> the user immediately sees OPEN
    6. §40C-B KV-only commit            -> OPEN written into future generative state
    7. stop display overlay / commit
    8. keep generating                  -> does the model sustain OPEN on its own

Groups:
    D0  no anchor, no commit, no display
    D1  identity anchor only
    D2  identity anchor + OPEN authority (display) + KV commit

Four gates, kept separate so a failure localises itself:
    Existence   did the door come back
    Identity    is it door_01 (via the real §39 store, wrong-ID must be 0)
    Pose        is it in the right place
    State       is it OPEN (continuous state_response, discrete label secondary)

Trajectory is extended so there IS a post-revisit observation window:
    [start]*8 + out(T*4) + out[::-1](T*4) + [start]*(tail*4)
revisit = first 2 tail chunks, observe = the rest.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python d2_loop.py --scene 04 --seed 42 --out_chunks 40 --tail 10
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
TREES_BB = (0.86, 0.15, 1.00, 0.55)      # distractor object for the wrong-ID test


def vram_free_mb():
    free, _ = torch.cuda.mem_get_info()
    return free / 2**20


def build_traj_tail(scene, total_out, tail):
    """Same out-and-back as build_traj, but with `tail` chunks at the start
    pose at the END so there is a post-revisit observation window."""
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


def best_match_loc(ref_patch, img, bb, dino_np, search=0.10, steps=15):
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
    ap.add_argument("--out_chunks", type=int, default=40)
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--blend", type=float, default=1.0)
    ap.add_argument("--groups", default="D0,D1,D2")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/d2")
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
    print("[d2] pipe + TAE built", flush=True)
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
    d = f"examples/d2_{scene}_O{args.out_chunks}_T{args.tail}"
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
    print(f"[d2] n_lat={n_lat} ref={ref_chunk} revisit={revisit} "
          f"observe={observe[0]}..{observe[-1]} "
          f"({len(observe)} chunks = {len(observe)*0.25:.2f}s)", flush=True)

    base_free = vram_free_mb()
    with torch.no_grad():
        y = pipe.vae.encode([torch.concat([
            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                            mode='bicubic').transpose(0, 1),
            torch.zeros(3, frames_n - 1, h, w)], dim=1).to(dev)])[0]
    msk = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y]).detach()
    ref_img = np.array(img_pil.resize((w, h), Image.BICUBIC))
    open_img = make_open_ref(ref_img, DOOR_BB, SKY_BB)
    with torch.no_grad():
        z_ref = pipe.vae.encode([
            TF.to_tensor(Image.fromarray(ref_img)).sub_(0.5).div_(0.5)
            .unsqueeze(0).transpose(0, 1).to(dev)])[0]
        z_open = pipe.vae.encode([
            TF.to_tensor(Image.fromarray(open_img)).sub_(0.5).div_(0.5)
            .unsqueeze(0).transpose(0, 1).to(dev)])[0]
    dy0, dy1 = int(DOOR_BB[1] * lat_h), int(DOOR_BB[3] * lat_h)
    dx0, dx1 = int(DOOR_BB[0] * lat_w), int(DOOR_BB[2] * lat_w)
    delta = torch.zeros_like(z_open)
    delta[:, :, dy0:dy1, dx0:dx1] = (z_open - z_ref)[:, :, dy0:dy1, dx0:dx1]
    delta = delta.detach()
    del z_ref, z_open
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[d2] setup: VAE dropped, driver free {vram_free_mb():.0f}MiB", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(anchor, use_commit, use_display):
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, latents, disp = [], [], []
        for cid in range(n_lat):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            p = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                          c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [y.split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
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
            on = cid in revisit
            # 3. identity anchor injection
            if anchor is not None and args.blend > 0 and on:
                x0 = x0.clone()
                x0[:, :, dy0:dy1, dx0:dx1] = (
                    args.blend * anchor
                    + (1 - args.blend) * x0[:, :, dy0:dy1, dx0:dx1])
            # 6. KV-only OPEN commit (memory), decode stays untouched (path B)
            x0_kv = x0 + args.alpha * delta if (use_commit and on) else x0
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0_kv], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            with torch.no_grad():
                fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                      parallel=False, show_progress_bar=False)
            frame = (fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                     * 255.0).clip(0, 255).astype(np.uint8)
            raw = frame.copy()          # the MODEL's own output -- what we measure
            # 5. §40B display authority (only while `on`, then stopped)
            if use_display and on:
                import cv2
                Hf, Wf = frame.shape[:2]
                ax0, ay0 = int(DOOR_BB[0] * Wf), int(DOOR_BB[1] * Hf)
                ax1, ay1 = int(DOOR_BB[2] * Wf), int(DOOR_BB[3] * Hf)
                frame = frame.copy()
                frame[ay0:ay1, ax0:ax1] = cv2.resize(
                    open_anchor_np, (ax1 - ax0, ay1 - ay0))
            outs.append(raw)            # raw only; display is a separate product
            if use_display and on:
                disp.append(frame)
            latents.append(x0.detach().float().cpu())
        del self_kv, cross_kv
        gc.collect(); torch.cuda.empty_cache()
        return outs, latents, disp

    # ---- D0 first: gives us the registration patch, the anchor and the store ----
    print("\n[d2] === D0 (no anchor, no commit, no display) ===", flush=True)
    D0_frames, D0_lat, _ = run(None, False, False)
    ref_frame = D0_frames[ref_chunk]
    anchor = D0_lat[ref_chunk][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    ref_door = patch(ref_frame, DOOR_BB)
    open_anchor_np = make_open_ref(
        np.array(Image.fromarray(ref_frame).resize((w, h), Image.BICUBIC)),
        DOOR_BB, SKY_BB)[int(DOOR_BB[1] * h):int(DOOR_BB[3] * h),
                         int(DOOR_BB[0] * w):int(DOOR_BB[2] * w)]
    c_reg = abs(structural_features(
        ref_door, surround_brightness_of(ref_frame, DOOR_BB))["contrast"]) + 1e-9

    # ---- §39 store: door_01 + a distractor, so wrong-ID is a real test ----
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
    store.set_state(door.persistent_id, open=True)
    print(f"[d2] store: door_01 id={door.persistent_id} state=OPEN, "
          f"+ 1 distractor (wrong-ID test)", flush=True)

    results = {"D0": D0_frames}
    disp_results = {}
    if "D1" in args.groups:
        print("\n[d2] === D1 (identity anchor only) ===", flush=True)
        results["D1"], _, _ = run(anchor, False, False)
    if "D2" in args.groups:
        print("\n[d2] === D2 (anchor + OPEN display + KV commit) ===", flush=True)
        results["D2"], _, disp_results["D2"] = run(anchor, True, True)

    # ---------- gates ----------
    # IMPORTANT: the identity injection ITSELF raises the ROI contrast (an
    # injected latent decodes to a sharper door than the drifted one), and the
    # §40A-2 probe would read that as OPEN. So the state baseline is D1
    # (identity restored, state NOT touched): state_response is reported
    # BOTH absolutely and relative to D1.
    def mean_resp(frames, chunks):
        vals = []
        for c in chunks:
            sb = surround_brightness_of(frames[c], DOOR_BB)
            vals.append(abs(structural_features(
                patch(frames[c], DOOR_BB), sb)["contrast"]) / c_reg)
        return float(np.mean(vals))

    base_rv = mean_resp(results.get("D1", D0_frames), revisit)
    base_ob = mean_resp(results.get("D1", D0_frames), observe)
    print(f"\n[d2] state baseline from D1 (identity only, state untouched): "
          f"revisit resp={base_rv:.2f}  observe resp={base_ob:.2f}", flush=True)

    print("\n[d2] ===== §40D-D2 gates =====")
    print(f"  {'tag':>4s} {'chunk':>5s} {'phase':>8s} {'DINO':>6s} {'resp':>6s} "
          f"{'rel':>5s} {'pose':>6s} {'match_id':>9s} {'Ex':>3s} {'Id':>3s} "
          f"{'Po':>3s} {'St':>3s}")
    summary = {}
    for tag, frames in results.items():
        rows = []
        for cid in revisit + observe:
            f = frames[cid]
            sim, loc = best_match_loc(ref_door, f, DOOR_BB, dino_np)
            resp = abs(structural_features(
                patch(f, DOOR_BB), surround_brightness_of(f, DOOR_BB))["contrast"]) / c_reg
            exp = (int(DOOR_BB[0] * f.shape[1]), int(DOOR_BB[1] * f.shape[0]))
            pe = math.hypot(loc[0] - exp[0], loc[1] - exp[1]) if loc else float("nan")
            # §39 reacquire (real store match, spatial prior active)
            cand = dict(anchor=dino_np(patch(f, DOOR_BB)),
                        uv=((loc[0] + 0.5 * (DOOR_BB[2] - DOOR_BB[0]) * f.shape[1]) / f.shape[1],
                            (loc[1] + 0.5 * (DOOR_BB[3] - DOOR_BB[1]) * f.shape[0]) / f.shape[0]),
                        world_transform=np.eye(4), bounds=[1, 2, 0.2])
            dec = store.reacquire([cand], t=cid * 0.25)[0]
            mid = dec["id"]
            ex = bool(sim > 0.6 and resp > 0.35)
            ident = bool(mid == door.persistent_id)
            pose_ok = bool(pe < 20.0)
            rel = resp / max(base_rv if cid in revisit else base_ob, 1e-6)
            st = "OPEN" if rel > 1.5 else "CLOSED"
            rows.append(dict(chunk=cid, phase="revisit" if cid in revisit else "observe",
                             dino=sim, resp=resp, rel=rel, pose=pe, match_id=mid,
                             existence=ex, identity=ident, pose_ok=pose_ok,
                             state=st))
            print(f"  {tag:>4s} {cid:5d} {rows[-1]['phase']:>8s} {sim:6.3f} {resp:6.2f} "
                  f"{rel:5.2f} {pe:6.1f} {str(mid):>9s} "
                  f"{'Y' if ex else '.':>3s} {'Y' if ident else '.':>3s} "
                  f"{'Y' if pose_ok else '.':>3s} {st:>3s}")
        summary[tag] = rows

    print("\n[d2] ===== summary =====")
    print(f"  {'tag':>4s} {'Existence':>10s} {'Identity':>9s} {'Pose':>6s} "
          f"{'State OPEN':>11s} {'wrongID':>8s}")
    for tag, rows in summary.items():
        rv = [r for r in rows if r["phase"] == "revisit"]
        ob = [r for r in rows if r["phase"] == "observe"]
        print(f"  {tag:>4s} {np.mean([r['existence'] for r in rv])*100:9.0f}% "
              f"{np.mean([r['identity'] for r in rv])*100:8.0f}% "
              f"{np.mean([r['pose_ok'] for r in rv])*100:5.0f}% "
              f"{np.mean([r['state']=='OPEN' for r in ob])*100:10.0f}% "
              f"{sum(1 for r in rows if r['match_id'] not in (None, door.persistent_id)):8d}")

    json.dump(summary, open(f"{args.out_dir}/d2.json", "w"), indent=1, default=float)
    for tag, frames in results.items():
        np.save(f"{args.out_dir}/{tag}_frames.npy", np.stack(frames))


if __name__ == "__main__":
    main()
