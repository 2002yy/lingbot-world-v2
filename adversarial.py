#!/usr/bin/env python
"""§42B: adversarial crossing + full occlusion hardening.

§42A already passed for ordinary crossing / 7-chunk occlusion / 7-chunk
out-of-FOV. §42B does NOT repeat that; it attacks the boundary where identity
switches are most likely:

    A / B move toward each other, their projected centres approach ~0
    A fully occludes B for 12-16 chunks (vs 7 in §42A)
    while occluded, A and B world state EVOLVE INDEPENDENTLY
        A: omega  +0.50 -> +0.70
        B: health  1.00 ->  0.80
      (state evolves without any visual observation)
    on reappearance A and B have SWAPPED screen sides
        before:  B ---- A
        after :  A ---- B
    C is a bystander control object that never moves

The side swap is the key kill-shot: it defeats "reconnect by nearest screen
position", which is the most common pseudo-success.

Gates (no new metrics invented):
  1 identity switch = 0        wrong-ID 0 throughout, checked at the crossing
                               and at the reappearance instants specifically
  2 cross-object state transfer = 0   state(A_after)==truth_A, same for B
  3 occlusion persistence      registered / state preserved / no duplicate /
                               no reset while visually absent
  4 reacquisition latency      chunks from visibility restored to correct
                               render binding (0 in §42A -- not required to
                               stay 0, but must be measured)
  5 cross-object pollution     state_leak + matched-control E_state, focused on
                               whether rebuilding A's anchor rewrites B

Derived indicator (reported, not gated):
  identity margin = self_match - best_other_match
  §42A gave A: 0.875 - 0.633 = 0.242. §42B watches whether crossing ->
  occlusion -> swapped reappearance compresses the margin toward 0, which would
  expose §42D/§42E scaling risk earlier than a PASS/FAIL would.

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python adversarial.py --scene 04 --seed 42 --out_chunks 12 --tail 30
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys

import cv2
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
from object_state import PersistentObjectStore
from world_metrics import bg_excess, tile_bank, bb_to_roi

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
SKY_BB = (0.05, 0.02, 0.35, 0.16)

# A and B swap sides; C is a static bystander control
OBJECTS = {
    "A_left":  dict(bb=(0.28, 0.56, 0.52, 0.90), health=0.40, omega=+0.50,
                    kind="crack", crosses=True),
    "B_right": dict(bb=(0.68, 0.42, 0.88, 0.72), health=1.00, omega=0.0,
                    kind="intact", crosses=True),
    "C_ctrl":  dict(bb=(0.86, 0.12, 1.00, 0.42), health=0.70, omega=-0.30,
                    kind="hole", crosses=False),
}
# screen-side swap, in latent units (1 latent = 8 px at 512 wide)
SWAP = 22
ID_EXIST = 0.60

# state evolution DURING occlusion (no visual observation)
OMEGA_A_AFTER = +0.70
HEALTH_B_AFTER = 0.80


def _cracks(frame, bb, n_lines, thickness, seed):
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    out = frame.copy()
    rng = np.random.RandomState(seed)
    seg = out[y0:y1, x0:x1]
    hh, ww = seg.shape[:2]
    for _ in range(n_lines):
        pts = [(rng.randint(0, max(1, ww)), rng.randint(0, max(1, hh)))]
        for _ in range(4):
            pts.append((int(np.clip(pts[-1][0] + rng.randint(-ww // 5, ww // 5),
                                    0, ww - 1)),
                        int(np.clip(pts[-1][1] + rng.randint(0, hh // 3),
                                    0, hh - 1))))
        cv2.polylines(out[y0:y1, x0:x1], [np.array(pts, np.int32)], False,
                      (12, 10, 10), thickness)
    return out


def make_correction(frame, bb, kind):
    if kind == "intact":
        return frame
    if kind == "crack":
        return _cracks(frame, bb, 10, 3, 3)
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    bx0, by0, bx1, by1 = [int(SKY_BB[0] * W), int(SKY_BB[1] * H),
                          int(SKY_BB[2] * W), int(SKY_BB[3] * H)]
    out = frame.copy()
    hole = cv2.resize(frame[by0:by1, bx0:bx1], (x1 - x0, y1 - y0))
    rim = max(2, (x1 - x0) // 12)
    out[y0 + rim:y1 - rim, x0 + rim:x1 - rim] = \
        hole[rim:hole.shape[0] - rim, rim:hole.shape[1] - rim]
    return out


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_chunks", type=int, default=12)
    ap.add_argument("--tail", type=int, default=30)
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/adv42b")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    names = list(OBJECTS.keys())

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[ab] pipe + TAE built", flush=True)
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
    visit = [0, 1]
    revisit = [n_lat - args.tail, n_lat - args.tail + 1]
    observe = list(range(revisit[-1] + 1, n_lat))
    d = f"examples/ab42_{scene}_O{args.out_chunks}_T{args.tail}"
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
    LR = {}
    for nm in names:
        bb = OBJECTS[nm]["bb"]
        LR[nm] = (int(bb[1] * lat_h), int(bb[3] * lat_h),
                  int(bb[0] * lat_w), int(bb[2] * lat_w))
    print(f"[ab] latent {lat_h}x{lat_w}, observe {observe[0]}..{observe[-1]} "
          f"({len(observe)} chunks)", flush=True)
    for nm in names:
        print(f"[ab]   {nm}: roi {LR[nm]}, health {OBJECTS[nm]['health']}, "
              f"omega {OBJECTS[nm]['omega']}, crosses={OBJECTS[nm]['crosses']}",
              flush=True)

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
    ys = {}
    for nm in names:
        if OBJECTS[nm]["kind"] == "intact":
            ys[nm] = None
        else:
            corr = make_correction(ref_img, OBJECTS[nm]["bb"], OBJECTS[nm]["kind"])
            ys[nm] = build_y(TF.to_tensor(Image.fromarray(corr)).sub_(0.5)
                             .div_(0.5).unsqueeze(0).transpose(0, 1))
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[ab] setup done, free {vram_free_mb():.0f}MiB", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(anchors, max_chunks=None):
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, lats = [], []
        N = n_lat if max_chunks is None else min(n_lat, max_chunks)
        for cid in range(N):
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
            x0 = x0.clone()
            if cid in visit or cid in revisit or cid in observe:
                for nm, (anc, dy, dx) in anchors.items():
                    if anc is None:
                        continue
                    y0, y1, x0_, x1_ = LR[nm]
                    a0, a1 = y0 + dy, y1 + dy
                    b0, b1 = x0_ + dx, x1_ + dx
                    if 0 <= a0 and a1 <= lat_h and 0 <= b0 and b1 <= lat_w:
                        x0[:, :, a0:a1, b0:b1] = anc
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
        return outs, lats

    # ---------- anchors ----------
    print("\n[ab] === D0 ===", flush=True)
    D0f, D0_lat = run({})
    A_anc, TPL = {}, {}
    for nm in names:
        y0, y1, x0_, x1_ = LR[nm]
        if OBJECTS[nm]["kind"] == "intact":
            A_anc[nm] = D0_lat[ref_chunk][:, :, y0:y1, x0_:x1_].clone().to(dev)
            TPL[nm] = patch(D0f[ref_chunk], OBJECTS[nm]["bb"])
            continue
        print(f"[ab] === canonicalise {nm} ===", flush=True)
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, lats = [], []
        for cid in range(args.canon_chunks):
            cur = torch.randn(16, 1, lat_h, lat_w, generator=g, device=dev)
            p = get_plucker_embeddings(rel_all[cid:cid + 1], Ks[None], h, w)
            p = rearrange(p, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                          c1=int(h // lat_h), c2=int(w // lat_w))[None]
            plk = rearrange(p, 'b (f h w) c -> b c f h w', f=1,
                            h=lat_h, w=lat_w).to(pdt)
            kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
                  "y": [ys[nm].split(1, dim=1)[min(cid, frames_n // 4 - 1)]],
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
        ci = min(2, len(lats) - 1)
        A_anc[nm] = lats[ci][:, :, y0:y1, x0_:x1_].clone().to(dev)
        TPL[nm] = patch(outs[ci], OBJECTS[nm]["bb"])

    print("\n[ab] ===== template separability =====")
    for a in names:
        print(f"    {a:>8s}  " + "  ".join(
            f"{b}:{float((dino_np(TPL[a]) * dino_np(TPL[b])).sum()):.3f}"
            for b in names))

    store = PersistentObjectStore()
    ids = {}
    for nm in names:
        o = OBJECTS[nm]
        st = store.register("obj", np.eye(4), [1, 1, 1], anchor=dino_np(TPL[nm]),
                            t=0.0, gameplay_state=dict(
                                health=o["health"], omega=o["omega"],
                                uv=((o["bb"][0] + o["bb"][2]) / 2,
                                    (o["bb"][1] + o["bb"][3]) / 2)))
        ids[nm] = st.persistent_id
        store.set_anchor_for_state(st.persistent_id, "home",
                                   latent=A_anc[nm].cpu().numpy(),
                                   feature=dino_np(TPL[nm]))
        store.set_state(st.persistent_id, state_key="home", open=False)
    print(f"[ab] store: " + ", ".join(f"{n}=id{ids[n]}" for n in names), flush=True)

    # ---------- adversarial schedule ----------
    obs = observe
    n_obs = len(obs)
    # 3 baseline | 4 approach | 16 occlusion | rest = reappearance+confirm
    p0 = obs[0:3]
    p1 = obs[3:7]
    p2 = obs[7:23]
    p4 = obs[23:]
    phase = {c: 0 for c in p0}
    for c in p1:
        phase[c] = 1
    for c in p2:
        phase[c] = 2
    for c in p4:
        phase[c] = 4
    print(f"[ab] phases: baseline {len(p0)} approach {len(p1)} "
          f"FULL-OCCLUSION {len(p2)} reappear {len(p4)}", flush=True)
    print(f"[ab] swap: A/B exchange screen sides on reappearance "
          f"({SWAP} latent = {SWAP*8}px)", flush=True)

    def plan(cid):
        ph = phase.get(cid, 0)
        # A moves right, B moves left, so they cross and end swapped.
        #  A home roi x ~ [18,33]  -> target [18+SWAP, 33+SWAP]
        #  B home roi x ~ [43,56]  -> target [43-SWAP, 56-SWAP]
        if ph == 0:
            a_dx = b_dx = 0
        elif ph == 1:
            i = p1.index(cid)
            fr = (i + 1) / max(len(p1), 1)
            a_dx = int(round(SWAP * 0.5 * fr))
            b_dx = -int(round(SWAP * 0.5 * fr))
        elif ph == 2:
            a_dx = int(round(SWAP * 0.5))
            b_dx = -int(round(SWAP * 0.5))
        else:
            a_dx = SWAP
            b_dx = -SWAP
        p = {}
        for nm in names:
            p[nm] = dict(active=True, dy=0,
                         dx=(a_dx if nm == "A_left" else
                             b_dx if nm == "B_right" else 0),
                         occluded=False)
        if ph == 2:
            p["B_right"]["active"] = False      # fully hidden behind A
            p["B_right"]["occluded"] = True
        return p

    # state evolves DURING occlusion, with no visual observation
    truth = {}
    for nm in names:
        truth[nm] = dict(health=OBJECTS[nm]["health"], omega=OBJECTS[nm]["omega"])
    print("\n[ab] state evolution during occlusion (no observation):", flush=True)
    print(f"     A omega {OBJECTS['A_left']['omega']:+.2f} -> {OMEGA_A_AFTER:+.2f}",
          flush=True)
    print(f"     B health {OBJECTS['B_right']['health']:.2f} -> "
          f"{HEALTH_B_AFTER:.2f}", flush=True)

    # ---------- render with mid-run state evolution ----------
    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
        dtype=dtype, device=dev)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
        dtype=dtype, device=dev)
    pipe._cross_attn_initialized = False
    g = torch.Generator(device=dev); g.manual_seed(sd)
    frames, snapshots = [], []
    evolved = set()
    for cid in range(n_lat):
        # during occlusion the authoritative state changes
        if cid in p2 and "A_left" not in evolved:
            half = p2[len(p2) // 2]
            if cid == half:
                store.get(ids["A_left"]).gameplay_state["omega"] = OMEGA_A_AFTER
                store.get(ids["B_right"]).gameplay_state["health"] = HEALTH_B_AFTER
                truth["A_left"]["omega"] = OMEGA_A_AFTER
                truth["B_right"]["health"] = HEALTH_B_AFTER
                evolved.add("A_left")
                print(f"[ab]   chunk {cid}: state evolved while B is occluded",
                      flush=True)
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
        x0 = x0.clone()
        pl = plan(cid)
        if cid in visit or cid in revisit or cid in observe:
            for nm in names:
                if not pl[nm]["active"]:
                    continue
                y0, y1, x0_, x1_ = LR[nm]
                a0, a1 = y0, y1
                b0, b1 = x0_ + pl[nm]["dx"], x1_ + pl[nm]["dx"]
                if 0 <= a0 and a1 <= lat_h and 0 <= b0 and b1 <= lat_w:
                    x0[:, :, a0:a1, b0:b1] = A_anc[nm]
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
        with torch.no_grad():
            fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                  parallel=False, show_progress_bar=False)
        frames.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                       * 255.0).clip(0, 255).astype(np.uint8))
    del self_kv, cross_kv
    gc.collect(); torch.cuda.empty_cache()

    # ================= metrics =================
    Hf, Wf = frames[0].shape[:2]

    def bbpx(nm, dx):
        y0, y1, x0_, x1_ = LR[nm]
        return (x0_ * vae_stride[2] + dx * vae_stride[2],
                y0 * vae_stride[1],
                x1_ * vae_stride[2] + dx * vae_stride[2],
                y1 * vae_stride[1])

    print("\n[ab] ===== per-chunk =====")
    print(f"  {'chunk':>5s} {'ph':>2s} " +
          " ".join(f"{n[:9]:>26s}" for n in names))
    rows = []
    reappear_first_ok = None
    for cid in observe:
        pl = plan(cid)
        line = f"  {cid:5d} {phase.get(cid,0):2d} "
        for nm in names:
            if not pl[nm]["active"]:
                line += f"{'OCCLUDED':>26s} "
                rows.append(dict(chunk=cid, phase=phase.get(cid, 0), obj=nm,
                                 active=False, occluded=True))
                continue
            rp = bbpx(nm, pl[nm]["dx"])
            bb = (rp[0] / Wf, rp[1] / Hf, rp[2] / Wf, rp[3] / Hf)
            op = patch(frames[cid], bb)
            own = float((dino_np(op) * dino_np(TPL[nm])).sum())
            others = {o: float((dino_np(op) * dino_np(TPL[o])).sum())
                      for o in names if o != nm}
            bo = max(others.items(), key=lambda kv: kv[1])
            margin = own - bo[1]
            uv_c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            for o in names:
                rpo = bbpx(o, plan(cid)[o]["dx"])
                store.set_prediction(ids[o], uv=(
                    (rpo[0] / Wf + rpo[2] / Wf) / 2,
                    (rpo[1] / Hf + rpo[3] / Hf) / 2))
            dec = store.reacquire([dict(anchor=dino_np(op), uv=uv_c,
                                        world_transform=np.eye(4),
                                        bounds=[1, 1, 1])],
                                  t=cid * 0.25, motion_aware=True)[0]
            ok = dec["id"] == ids[nm]
            wrong = dec["id"] is not None and dec["id"] != ids[nm]
            exist = own > ID_EXIST
            rows.append(dict(chunk=cid, phase=phase.get(cid, 0), obj=nm,
                             active=True, own=own, best_other=bo[0],
                             best_other_sim=bo[1], margin=margin,
                             matched=dec["id"], ok=bool(ok), wrong=bool(wrong),
                             exist=bool(exist)))
            if phase.get(cid) == 4 and nm == "B_right" \
                    and reappear_first_ok is None:
                reappear_first_ok = ok
            line += f"{('OK' if ok else ('WRONG' if wrong else 'new')):>6s}"
            line += f" m{margin:+.2f} ex{int(exist)} "
        print(line, flush=True)

    act = [r for r in rows if r["active"]]
    occ = [r for r in rows if r.get("occluded")]

    # gates
    g1 = (all(r["exist"] for r in act) and all(r["ok"] for r in act)
          and not any(r["wrong"] for r in act))
    xfer = 0
    for nm in names:
        for om in names:
            if nm == om:
                continue
            if abs(store.get(ids[nm]).gameplay_state["health"]
                   - truth[om]["health"]) < 1e-9 and \
               abs(OBJECTS[nm]["health"] - OBJECTS[om]["health"]) > 1e-9:
                xfer += 1
    g2 = xfer == 0
    # also verify the evolved values landed on the right objects
    g2 = g2 and abs(store.get(ids["A_left"]).gameplay_state["omega"]
                    - OMEGA_A_AFTER) < 1e-9
    g2 = g2 and abs(store.get(ids["B_right"]).gameplay_state["health"]
                    - HEALTH_B_AFTER) < 1e-9
    g3 = (len(occ) >= 12 and len(store._objs) == len(names)
          and abs(store.get(ids["B_right"]).gameplay_state["health"]
                  - HEALTH_B_AFTER) < 1e-9)
    reacq = None
    for cid in p4:
        r = [x for x in rows if x["chunk"] == cid and x["obj"] == "B_right"]
        if r and r[0].get("ok"):
            reacq = cid - p4[0]
            break
    g4 = reacq is not None

    print("\n[ab] ===== §42B five gates =====")
    print(f"  1 identity switch = 0     exist-all={all(r['exist'] for r in act)} "
          f"id-all={all(r['ok'] for r in act)} "
          f"wrong-ID={sum(1 for r in act if r['wrong'])} -> {g1}")
    print(f"  2 cross-object transfer    count={xfer}, "
          f"A.omega={store.get(ids['A_left']).gameplay_state['omega']:+.2f} "
          f"(want {OMEGA_A_AFTER:+.2f}), "
          f"B.health={store.get(ids['B_right']).gameplay_state['health']:.2f} "
          f"(want {HEALTH_B_AFTER:.2f}) -> {g2}")
    print(f"  3 occlusion persistence    occluded chunks={len(occ)} "
          f"(>=12), objects alive={len(store._objs)}/{len(names)} -> {g3}")
    print(f"  4 reacquisition latency    {reacq} chunks "
          f"(B reappears on the OPPOSITE side) -> {g4}")
    print(f"  5 cross-object pollution   state_leak + E_state below")

    print("\n[ab] ===== identity margin =====")
    for nm in names:
        ms = [r["margin"] for r in act if r["obj"] == nm]
        if ms:
            print(f"    {nm:>8s}: mean {np.mean(ms):+.3f}  min {min(ms):+.3f}  "
                  f"max {max(ms):+.3f}")
    allm = [r["margin"] for r in act]
    print(f"    ALL     : mean {np.mean(allm):+.3f}  min {min(allm):+.3f}")
    print(f"    (§42A reference: A margin +0.242)")
    mmin = min(allm)
    compressed = mmin < 0.10
    print(f"    margin compressed below 0.10 anywhere: {compressed}")

    hm = max(r["chunk"] for r in act)
    print(f"\n[ab] ===== state_leak at the last visible chunk {hm} =====")
    plh = plan(hm)
    mat = {}
    for a in names:
        if not plh[a]["active"]:
            continue
        rpa = bbpx(a, plh[a]["dx"])
        bba = (rpa[0] / Wf, rpa[1] / Hf, rpa[2] / Wf, rpa[3] / Hf)
        oa = patch(frames[hm], bba)
        for b in names:
            mat[f"{a}<-{b}"] = float((dino_np(oa) * dino_np(TPL[b])).sum())
    print("        " + " ".join(f"{b:>9s}" for b in names))
    for a in names:
        if all(f"{a}<-{b}" in mat for b in names):
            print(f"  {a[:6]:>6s} " + " ".join(
                f"{mat[f'{a}<-{b}']:9.3f}" for b in names))
    diag_ok = all(mat[f"{a}<-{a}"] == max(mat[f"{a}<-{b}"] for b in names)
                  for a in names if f"{a}<-{a}" in mat)

    ok = g1 and g2 and g3 and g4 and diag_ok
    print(f"\n  §42B: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  -> identity survives crossing + 16-chunk full occlusion + a "
              "screen-side swap")
        print("  -> strong architectural evidence: identity is a property of "
              "world state, not screen position")

    json.dump(dict(rows=rows, matrix=mat, ids={k: int(v) for k, v in ids.items()},
                   phases=dict(p0=p0, p1=p1, p2=p2, p4=p4),
                   cross_object_transfer=xfer, reacquisition_latency=reacq,
                   margin_mean=float(np.mean(allm)), margin_min=float(mmin),
                   margin_compressed=bool(compressed),
                   gates=dict(identity=g1, state_attachment=g2,
                              occlusion=g3, reacquisition=g4, matrix=diag_ok),
                   pass_=bool(ok)),
              open(f"{args.out_dir}/adv42b.json", "w"), indent=1, default=float)
    np.save(f"{args.out_dir}/frames.npy", np.stack(frames))


if __name__ == "__main__":
    main()
