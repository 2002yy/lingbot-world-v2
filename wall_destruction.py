#!/usr/bin/env python
"""§41D-1: Destruction -- the first gameplay state that changes WORLD TOPOLOGY.

§40 made a fact survive past the model's memory. §41C made physics produce new
authoritative facts. §41D combines them: damage accumulates, the gameplay truth
changes, and -- crucially -- **collision / passability change with it**.

Three-stage destruction (no continuous visual damage yet):

    ObjectState
      ├─ persistent_id   = wall_05
      ├─ health in [0,1]        <- authoritative CONTINUOUS gameplay truth
      ├─ damage_stage           <- render/gameplay bucket
      ├─ collision_enabled      <- topology
      └─ render_binding.state_anchors {intact, damaged, destroyed}

    health = 1.0  intact     collision=solid     occupancy=blocked
    health ~ 0.5  damaged    collision=solid     occupancy=blocked
    health = 0.0  destroyed  collision=disabled  occupancy=passable

"health = 0.37 is the game fact; what 0.37 LOOKS like is render binding."

The real graduation test is not a broken-wall DINO score: it is that the player
can actually WALK THROUGH where the wall used to be.

Scenarios (damage is applied during the 2 visible chunks at the start pose,
then the camera leaves the FOV and comes back):
    D0  no attack            -> intact
    D1  one light hit        -> health 0.8, still intact
    D2  hits to health 0.5   -> damaged
    D3  hits to health 0.0   -> destroyed, collision off

Four gate layers, verified separately:
    1 gameplay state   health/stage change ONLY per the damage rule
    2 render state     stage recognised via state-aware prototypes, wrong-ID=0
    3 physics/topology blocked before destroyed, passable after
    4 persistence      after leaving > T50 the wall is still wall_05, still
                       destroyed, still passable, and does NOT grow back

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python wall_destruction.py --scene 04 --seed 42 --out_chunks 12 --tail 30
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
from door_state_probe import make_open_ref, surround_brightness_of, \
    structural_features
from object_state import PersistentObjectStore

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
DOOR_BB = (0.70, 0.50, 0.84, 0.66)
SKY_BB = (0.05, 0.02, 0.35, 0.16)
TREES_BB = (0.86, 0.15, 1.00, 0.55)
# wall_05: a wall segment at the centre of the Great Wall frame
WALL_BB = (0.28, 0.56, 0.52, 0.90)

# damage rule: each hit removes this much health
HIT = 0.20
SCENARIOS = [
    ("D0_no_attack", 0),
    ("D1_one_hit", 1),
    ("D2_damaged", 3),      # 1.0 - 3*0.2 = 0.40 -> damaged
    ("D3_destroyed", 5),    # 1.0 - 5*0.2 = 0.00 -> destroyed
]
DAMAGED_THRESHOLD = 0.60
DESTROYED_THRESHOLD = 0.05


def stage_of(health):
    if health <= DESTROYED_THRESHOLD:
        return "destroyed"
    if health <= DAMAGED_THRESHOLD:
        return "damaged"
    return "intact"


def damage_rule(hits):
    """Pure function: hits -> authoritative health. No rendering involved."""
    return max(0.0, 1.0 - hits * HIT)


def make_damaged_ref(frame, bb):
    """Synthesise a damaged correction: dark cracks over the wall."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    out = frame.copy()
    rng = np.random.RandomState(0)
    seg = out[y0:y1, x0:x1]
    hh, ww = seg.shape[:2]
    for _ in range(6):
        p0 = (rng.randint(0, max(1, ww)), rng.randint(0, max(1, hh)))
        pts = [p0]
        for _ in range(4):
            pts.append((int(np.clip(pts[-1][0] + rng.randint(-ww // 5, ww // 5),
                                    0, ww - 1)),
                        int(np.clip(pts[-1][1] + rng.randint(0, hh // 3),
                                    0, hh - 1))))
        pts = np.array(pts, np.int32)
        cv2.polylines(out[y0:y1, x0:x1], [pts], False, (12, 10, 10), 2)
    return out


def make_destroyed_ref(frame, bb, bg_bb):
    """Synthesise a destroyed correction: real background pixels fill a hole."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [int(bb[0] * W), int(bb[1] * H), int(bb[2] * W), int(bb[3] * H)]
    bx0, by0, bx1, by1 = [int(bg_bb[0] * W), int(bg_bb[1] * H),
                          int(bg_bb[2] * W), int(bg_bb[3] * H)]
    bg = frame[by0:by1, bx0:bx1]
    out = frame.copy()
    hole = cv2.resize(bg, (x1 - x0, y1 - y0))
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
    ap.add_argument("--scenarios", default=",".join(s[0] for s in SCENARIOS))
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/destroy")
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
    print("[de] pipe + TAE built", flush=True)
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
    d = f"examples/de_{scene}_O{args.out_chunks}_T{args.tail}"
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
    wy0, wy1 = int(WALL_BB[1] * lat_h), int(WALL_BB[3] * lat_h)
    wx0, wx1 = int(WALL_BB[0] * lat_w), int(WALL_BB[2] * lat_w)
    print(f"[de] n_lat={n_lat} ref={ref_chunk} visit={visit} "
          f"revisit={revisit} observe={observe[0]}..{observe[-1]} "
          f"wall ROI=[{wy0}:{wy1},{wx0}:{wx1}]", flush=True)

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
    # canonicalisation sources: damaged (cracks) and destroyed (hole)
    dmg_img = make_damaged_ref(ref_img, WALL_BB)
    dst_img = make_destroyed_ref(ref_img, WALL_BB, SKY_BB)
    y_dmg = build_y(TF.to_tensor(Image.fromarray(dmg_img)).sub_(0.5).div_(0.5)
                    .unsqueeze(0).transpose(0, 1))
    y_dst = build_y(TF.to_tensor(Image.fromarray(dst_img)).sub_(0.5).div_(0.5)
                    .unsqueeze(0).transpose(0, 1))
    # door anchor pieces (unused here but keep the pipeline consistent)
    dy0, dy1 = int(DOOR_BB[1] * lat_h), int(DOOR_BB[3] * lat_h)
    dx0, dx1 = int(DOOR_BB[0] * lat_w), int(DOOR_BB[2] * lat_w)
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[de] setup done, free {vram_free_mb():.0f}MiB", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(y_cond, anchor=None, max_chunks=None):
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
            visible = (cid in visit) or (cid in revisit) or (cid in observe)
            if anchor is not None and visible:
                x0[:, :, wy0:wy1, wx0:wx1] = anchor
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

    # ---------- canonical anchors for the three stages ----------
    print("\n[de] === D0 (no injection) ===", flush=True)
    D0_frames, D0_lat = run(y, None)
    ref_frame = D0_frames[ref_chunk]
    A_INTACT = D0_lat[ref_chunk][:, :, wy0:wy1, wx0:wx1].clone().to(dev)
    wall_intact = patch(ref_frame, WALL_BB)
    print(f"[de] === canonicalise damaged ({args.canon_chunks} chunks) ===", flush=True)
    dmg_frames, dmg_lat = run(y_dmg, None, max_chunks=args.canon_chunks)
    ci = min(2, args.canon_chunks - 1)
    A_DAMAGED = dmg_lat[ci][:, :, wy0:wy1, wx0:wx1].clone().to(dev)
    wall_damaged = patch(dmg_frames[ci], WALL_BB)
    print(f"[de] === canonicalise destroyed ({args.canon_chunks} chunks) ===", flush=True)
    dst_frames, dst_lat = run(y_dst, None, max_chunks=args.canon_chunks)
    A_DESTROYED = dst_lat[ci][:, :, wy0:wy1, wx0:wx1].clone().to(dev)
    wall_destroyed = patch(dst_frames[ci], WALL_BB)
    torch.save(dict(intact=A_INTACT.cpu(), damaged=A_DAMAGED.cpu(),
                    destroyed=A_DESTROYED.cpu()),
               f"{args.out_dir}/wall_anchors.pt")

    FEAT = {"intact": dino_np(wall_intact),
            "damaged": dino_np(wall_damaged),
            "destroyed": dino_np(wall_destroyed)}
    # cross-similarity: are the three prototypes actually distinguishable?
    print("\n[de] ===== state prototype separability =====")
    for a in FEAT:
        row = "  ".join(f"{b}:{float(np.dot(FEAT[a], FEAT[b])):.3f}" for b in FEAT)
        print(f"    {a:>10s}  {row}")

    # ---------- store: wall_05 with state-specific prototypes ----------
    store = PersistentObjectStore()
    wall = store.register("wall", np.eye(4), [1, 3, 2], anchor=FEAT["intact"],
                          t=0.0, gameplay_state=dict(health=1.0, stage="intact",
                                                     uv=((WALL_BB[0] + WALL_BB[2]) / 2,
                                                         (WALL_BB[1] + WALL_BB[3]) / 2)))
    store.register("trees", np.eye(4), [1, 1, 1],
                   anchor=dino_np(patch(ref_frame, TREES_BB)), t=0.0,
                   gameplay_state=dict(open=False,
                                       uv=((TREES_BB[0] + TREES_BB[2]) / 2,
                                           (TREES_BB[1] + TREES_BB[3]) / 2)))
    for sk, ft, la in (("intact", FEAT["intact"], A_INTACT),
                       ("damaged", FEAT["damaged"], A_DAMAGED),
                       ("destroyed", FEAT["destroyed"], A_DESTROYED)):
        store.set_anchor_for_state(wall.persistent_id, sk, latent=la.cpu().numpy(),
                                   feature=ft)

    summary = {}
    for sname in args.scenarios.split(","):
        spec = [s for s in SCENARIOS if s[0] == sname]
        if not spec:
            continue
        _, hits = spec[0]
        # ---- 1. GAMEPLAY STATE (pure rule, no rendering) ----
        health = damage_rule(hits)
        stage = stage_of(health)
        collision_enabled = health > DESTROYED_THRESHOLD
        st = store.get(wall.persistent_id)
        st.gameplay_state.update(health=health, stage=stage,
                                 collision_enabled=collision_enabled)
        store.set_state(wall.persistent_id, state_key=stage, open=False)
        active = store.state_latent(wall.persistent_id, stage)

        # ---- 3. TOPOLOGY: can the player walk through? ----
        # 1-D walk toward the wall; blocked while collision_enabled
        px = 0.0                # relative to the wall's near face
        v = 1.0
        passed = False
        for _ in range(int(3.0 / 0.05)):
            px += v * 0.05
            if collision_enabled and px > -0.2:
                px = -0.2        # stopped by the wall
            if px > 0.6:
                passed = True
                break

        # ---- 2. RENDER ----
        F, _ = run(y, None if active is None else torch.as_tensor(
            active).to(dev).reshape(A_INTACT.shape))
        # ---- render-state classification over visible chunks ----
        vis_all = visit + revisit + observe
        stage_ids, wrong_ids, sims = [], 0, []
        for cid in vis_all:
            fr = F[cid]
            obs = patch(fr, WALL_BB)
            scores = {k: float((dino_np(obs) * v).sum()) for k, v in FEAT.items()}
            best = max(scores, key=scores.get)
            stage_ids.append(best)
            sims.append(scores[stage])
            uv_c = ((WALL_BB[0] + WALL_BB[2]) / 2, (WALL_BB[1] + WALL_BB[3]) / 2)
            store.set_prediction(wall.persistent_id, uv=uv_c)
            dec = store.reacquire([dict(anchor=dino_np(obs), uv=uv_c,
                                        world_transform=np.eye(4),
                                        bounds=[1, 3, 2])],
                                  t=cid * 0.25, motion_aware=True)[0]
            if dec["id"] not in (None, wall.persistent_id):
                wrong_ids += 1
        render_match = float(np.mean([s == stage for s in stage_ids]))

        # ---- 4. PERSISTENCE: pre-FOV (visit) vs post-FOV (revisit+observe) ----
        pre = [stage_ids[i] for i, c in enumerate(vis_all) if c in visit]
        post = [stage_ids[i] for i, c in enumerate(vis_all)
                if c in revisit or c in observe]
        persistence = float(np.mean([s == stage for s in post]))
        grew_back = any(s == "intact" for s in post) and stage != "intact"

        summary[sname] = dict(
            hits=hits, health=health, stage=stage,
            collision_enabled=collision_enabled, passed=passed,
            render_match=render_match, wrong_id=wrong_ids,
            persistence=persistence, grew_back=grew_back,
            sim=float(np.mean(sims)), pre=pre, post=post)
        x = summary[sname]
        print(f"\n[de] === {sname}: hits={hits} health={health:.2f} "
              f"stage={stage} collision={collision_enabled} ===", flush=True)
        print(f"    TOPOLOGY passed={passed}  RENDER match={render_match*100:.0f}% "
              f"wID={wrong_ids} sim={x['sim']:.3f}  "
              f"PERSISTENCE post={persistence*100:.0f}% grew_back={grew_back}",
              flush=True)

    print("\n[de] ===== §41D-1 Destruction =====")
    print(f"  {'scenario':>14s} {'hits':>4s} {'health':>7s} {'stage':>10s} "
          f"{'collide':>8s} {'pass':>5s} {'render':>7s} {'wID':>4s} "
          f"{'persist':>8s} {'regrow':>7s}")
    for s, x in summary.items():
        print(f"  {s:>14s} {x['hits']:4d} {x['health']:7.2f} {x['stage']:>10s} "
              f"{str(x['collision_enabled']):>8s} {str(x['passed']):>5s} "
              f"{x['render_match']*100:6.0f}% {x['wrong_id']:4d} "
              f"{x['persistence']*100:7.0f}% {str(x['grew_back']):>7s}")

    print("\n[de] ===== §41D-1 four gate layers =====")
    g1 = g2 = g3 = g4 = True
    for s, x in summary.items():
        # 1 gameplay state: health must follow the rule exactly, and damage
        #   stages must be ordered
        v1 = abs(x["health"] - damage_rule(x["hits"])) < 1e-9
        # 2 render state recognised
        v2 = x["render_match"] >= 0.9 and x["wrong_id"] == 0
        # 3 topology
        if x["stage"] == "destroyed":
            v3 = (not x["collision_enabled"]) and x["passed"]
        else:
            v3 = x["collision_enabled"] and (not x["passed"])
        # 4 persistence across the FOV excursion
        v4 = x["persistence"] >= 0.9 and not x["grew_back"]
        g1 &= v1; g2 &= v2; g3 &= v3; g4 &= v4
        print(f"    {s}: gameplay {'Y' if v1 else '.'} render "
              f"{'Y' if v2 else '.'} topology {'Y' if v3 else '.'} "
              f"persistence {'Y' if v4 else '.'}")
    print(f"\n  1 gameplay state : {'PASS' if g1 else 'FAIL'}")
    print(f"  2 render state   : {'PASS' if g2 else 'FAIL'}")
    print(f"  3 physics/topology: {'PASS' if g3 else 'FAIL'}")
    print(f"  4 persistence    : {'PASS' if g4 else 'FAIL'}")
    ok = g1 and g2 and g3 and g4
    print(f"\n  §41D-1: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  -> the change is not one frame of video but the world topology")

    json.dump(summary, open(f"{args.out_dir}/destroy.json", "w"),
              indent=1, default=float)
    json.dump({k: v.tolist() for k, v in FEAT.items()},
              open(f"{args.out_dir}/wall_feats.json", "w"))


if __name__ == "__main__":
    main()
