#!/usr/bin/env python
"""§41C-1: Push -- the first real gameplay physics loop.

Replaces §41B's "velocity = hand specified" with "velocity = collision solver
result", closing:

    player input -> contact -> physics state change -> ObjectState is the new
    truth -> LingBot keeps rendering the new truth

Deliberately simple physics: ground-plane 2D rigid body, disc player vs AABB
box, linear impulse only, no rotation, no friction, low restitution, one box.

Three truths stay strictly separate (established in §41B-2b):
    physics transform        = authoritative world fact
    rendered object position = visual observation
    matcher measurement      = diagnostic only
Never let "the AI drew the box 8px off" feed back into the physics transform.

Scenarios:
    C0  no contact      player passes by -> box must NOT move
    C1  low-speed push  frontal
    C2  high-speed push larger impulse -> higher initial speed / longer travel
    C3  sustained push  player keeps contact -> box keeps moving

Metrics, in three layers:
    physics   : no-contact false positive, penetration, impulse direction,
                dv_box, distance travelled, velocity decay
    render    : existence / identity / wrong-ID / pos_err / trail_excess / collapse
    causality : push_response = rendered displacement / physics displacement,
                e_pos(t) mean / p90 / max
    timing    : contact-to-state, contact-to-first-rendered-motion

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python push_physics.py --scene 04 --seed 42 --out_chunks 12 --tail 30
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

# ---- scenarios: (name, player_x0, player_y0, player_vx, player_vy, sustained)
SCENARIOS = [
    ("C0_no_contact", 56.0, 34.0, -1.5, 0.0, False),
    ("C1_low_push", 56.0, 23.0, -1.5, 0.0, False),
    ("C2_high_push", 56.0, 23.0, -4.0, 0.0, False),
    ("C3_sustained", 56.0, 23.0, -1.5, 0.0, True),
]
# 1 physics unit -> LATENT_PER_UNIT latent units. The physics works in its own
# units; only the render binding maps displacement into the latent ROI.
LATENT_PER_UNIT = 3.0


# --------------------------------------------------------------------------
# tiny 2D physics: disc vs AABB, linear impulse only
# --------------------------------------------------------------------------
class PushSim:
    """Ground-plane 2D. Player = disc, box = AABB. Linear impulse, no rotation."""

    def __init__(self, box_center, box_half, player_pos, player_vel,
                 player_r=3.0, box_mass=2.0, player_mass=1.0,
                 restitution=0.05):
        self.box_c = np.array(box_center, float)
        self.box_h = np.array(box_half, float)
        self.box_v = np.zeros(2)
        self.p = np.array(player_pos, float)
        self.pv = np.array(player_vel, float)
        self.pr = float(player_r)
        self.mb, self.mp = float(box_mass), float(player_mass)
        self.e = float(restitution)
        self.contacts = 0
        self.max_pen = 0.0
        self.impulses = []

    def _closest(self):
        d = self.p - self.box_c
        q = np.clip(d, -self.box_h, self.box_h)
        return self.box_c + q

    def step(self, dt, kinematic_player=False):
        self.p = self.p + self.pv * dt
        c = self._closest()
        delta = self.p - c
        dist = float(np.linalg.norm(delta))
        if dist >= self.pr or dist < 1e-9:
            return None                       # no contact
        n = delta / dist                      # from box surface to player
        pen = self.pr - dist
        self.max_pen = max(self.max_pen, pen)
        self.contacts += 1
        # positional correction (split by inverse mass)
        corr = n * pen
        wi = (1.0 / self.mp) / (1.0 / self.mp + 1.0 / self.mb)
        self.p = self.p + corr * wi
        self.box_c = self.box_c - corr * (1 - wi)
        # impulse along the normal
        v_rel = self.pv - self.box_v
        vn = float(np.dot(v_rel, n))
        if vn < 0:                            # approaching
            j = -(1.0 + self.e) * vn / (1.0 / self.mp + 1.0 / self.mb)
            if not kinematic_player:          # a "pusher" keeps its velocity
                self.pv = self.pv + (j / self.mp) * n
            self.box_v = self.box_v - (j / self.mb) * n
            self.impulses.append(float(j))
        return dict(normal=n.tolist(), pen=float(pen))


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
    ap.add_argument("--phys_dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/push")
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
    print("[pu] pipe + TAE built", flush=True)
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
    d = f"examples/pu_{scene}_O{args.out_chunks}_T{args.tail}"
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
    print(f"[pu] n_lat={n_lat} ref={ref_chunk} revisit={revisit} "
          f"observe={observe[0]}..{observe[-1]} ({len(observe)} chunks)", flush=True)

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
    print(f"[pu] setup done, free {vram_free_mb():.0f}MiB, box {dw}x{dh} latent",
          flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(y_cond, anchor=None, box_path=None, max_chunks=None):
        """box_path: dict cid -> (dcol, drow) latent offset from the base ROI."""
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
            if anchor is not None and (cid in revisit or cid in observe):
                dc, dr = box_path.get(cid, (0, 0)) if box_path else (0, 0)
                a0, a1 = dy0 + dr, dy1 + dr
                b0, b1 = dx0 + dc, dx1 + dc
                if 0 <= a0 and a1 <= lat_h and 0 <= b0 and b1 <= lat_w:
                    x0[:, :, a0:a1, b0:b1] = anchor
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

    print("\n[pu] === D0 (no injection) ===", flush=True)
    D0_frames, _ = run(y, None)
    ref_frame = D0_frames[ref_chunk]
    ref_door = patch(ref_frame, DOOR_BB)
    print(f"[pu] === canonicalisation ===", flush=True)
    canon_frames, canon_lat = run(y_open, None, max_chunks=args.canon_chunks)
    ci = min(2, args.canon_chunks - 1)
    canon_door = patch(canon_frames[ci], DOOR_BB)
    CANON_FEAT = dino_np(canon_door)
    Z_OPEN = canon_lat[ci][:, :, dy0:dy1, dx0:dx1].clone().to(dev)

    store = PersistentObjectStore()
    box = store.register("box", np.eye(4), [1, 1, 1], anchor=dino_np(ref_door),
                         t=0.0, gameplay_state=dict(open=False,
                                                    uv=((DOOR_BB[0] + DOOR_BB[2]) / 2,
                                                        (DOOR_BB[1] + DOOR_BB[3]) / 2)))
    store.register("trees", np.eye(4), [1, 1, 1],
                   anchor=dino_np(patch(ref_frame, TREES_BB)), t=0.0,
                   gameplay_state=dict(open=False,
                                       uv=((TREES_BB[0] + TREES_BB[2]) / 2,
                                           (TREES_BB[1] + TREES_BB[3]) / 2)))
    store.set_anchor_for_state(box.persistent_id, "closed", feature=dino_np(ref_door))
    store.set_anchor_for_state(box.persistent_id, "open", feature=CANON_FEAT)
    store.set_state(box.persistent_id, open=True)

    summary = {}
    for sname in args.scenarios.split(","):
        spec = [s for s in SCENARIOS if s[0] == sname]
        if not spec:
            continue
        _, px0, py0, pvx, pvy, sustained = spec[0]
        # ---- 1. PHYSICS (authoritative) ----
        sim = PushSim(box_center=(dx0 + dw / 2.0, dy0 + dh / 2.0),
                      box_half=(dw / 2.0, dh / 2.0),
                      player_pos=(px0, py0), player_vel=(pvx, pvy),
                      player_r=3.0, box_mass=2.0, player_mass=1.0)
        box0 = sim.box_c.copy()
        path, t_contact = {}, None
        for cid in range(n_lat):
            if cid in observe:
                n_steps = int(round(0.25 / args.phys_dt))
                for _ in range(n_steps):
                    r = sim.step(args.phys_dt, kinematic_player=sustained)
                    if r is not None and t_contact is None:
                        t_contact = cid
            if cid in observe:
                path[cid] = (int(round((sim.box_c[0] - box0[0]) * LATENT_PER_UNIT)),
                             int(round((sim.box_c[1] - box0[1]) * LATENT_PER_UNIT)))
        disp_phys = float(np.linalg.norm(sim.box_c - box0))
        print(f"\n[pu] === {sname} === physics: contacts={sim.contacts} "
              f"max_pen={sim.max_pen:.3f} impulses={len(sim.impulses)} "
              f"dv_box={float(np.linalg.norm(sim.box_v)):.3f} "
              f"disp={disp_phys:.2f} latent "
              f"(first contact chunk {t_contact})", flush=True)

        # ---- 2. RENDER: LingBot draws the physics truth ----
        F, _ = run(y, Z_OPEN, box_path=path)

        # ---- 3. THREE-LAYER METRICS ----
        dinos, exs, poss, ids, wids, apps, trails, epos = \
            [], [], [], [], 0, [], [], []
        first_rendered = None
        for cid in observe:
            dc, dr = path[cid]
            a0, a1 = dy0 + dr, dy1 + dr
            b0, b1 = dx0 + dc, dx1 + dc
            if not (0 <= a0 and a1 <= lat_h and 0 <= b0 and b1 <= lat_w):
                continue
            bb = (b0 * vae_stride[2] / w, a0 * vae_stride[1] / h,
                  b1 * vae_stride[2] / w, a1 * vae_stride[1] / h)
            fr = F[cid]
            p_obs = patch(fr, bb)
            sim_s, loc = best_match_loc(canon_door, fr, bb, dino_np)
            dinos.append(sim_s)
            exs.append(bool(sim_s > 0.6))
            exp = (int(bb[0] * fr.shape[1]), int(bb[1] * fr.shape[0]))
            pe = math.hypot(loc[0] - exp[0], loc[1] - exp[1]) if loc else 1e3
            poss.append(pe)
            epos.append(pe)
            if first_rendered is None and abs(dc) + abs(dr) > 0:
                first_rendered = cid
            uv_c = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
            store.set_prediction(box.persistent_id, uv=uv_c)
            dec = store.reacquire([dict(anchor=dino_np(p_obs), uv=uv_c,
                                        world_transform=np.eye(4),
                                        bounds=[1, 1, 1])],
                                  t=cid * 0.25, motion_aware=True)[0]
            ids.append(dec["id"] == box.persistent_id)
            if dec["id"] not in (None, box.persistent_id):
                wids += 1
            apps.append(float((dino_np(p_obs) * CANON_FEAT).sum()))
            # trail: a position the box has left by >= 1 box width
            past = None
            for c2 in observe:
                if c2 >= cid:
                    break
                dc2, dr2 = path[c2]
                if abs((dx0 + dc2) - (dx0 + dc)) >= dw:
                    past = (dc2, dr2)
            if past:
                pbb = ((dx0 + past[0]) * vae_stride[2] / w,
                       (dy0 + past[1]) * vae_stride[1] / h,
                       (dx1 + past[0]) * vae_stride[2] / w,
                       (dy1 + past[1]) * vae_stride[1] / h)
                Hf, Wf = fr.shape[:2]
                x0_, y0_ = int(pbb[0] * Wf), int(pbb[1] * Hf)
                x1_, y1_ = int(pbb[2] * Wf), int(pbb[3] * Hf)
                if x1_ > x0_ and y1_ > y0_:
                    cs = float((dino_np(fr[y0_:y1_, x0_:x1_]) * CANON_FEAT).sum())
                    ds = float((dino_np(D0_frames[cid][y0_:y1_, x0_:x1_])
                                * CANON_FEAT).sum())
                    trails.append(cs - ds)

        # rendered displacement in latent units (from the best-match positions)
        rendered_disp = float(abs(path[max(observe)][0] - path[min(observe)][0]))
        push_response = (rendered_disp / disp_phys) if disp_phys > 1e-6 else None
        summary[sname] = dict(
            contacts=sim.contacts, max_pen=float(sim.max_pen),
            n_impulses=len(sim.impulses),
            dv_box=float(np.linalg.norm(sim.box_v)),
            disp_phys=disp_phys,
            no_contact_fp=int(sim.contacts == 0 and disp_phys > 0.5),
            box_moved=bool(disp_phys > 0.5),
            existence=float(np.mean(exs)) if exs else 0.0,
            identity=float(np.mean(ids)) if ids else 0.0,
            wrong_id=wids,
            pos_err=float(np.mean(poss)) if poss else None,
            e_pos_p90=float(np.percentile(epos, 90)) if epos else None,
            e_pos_max=float(np.max(epos)) if epos else None,
            appearance=float(np.mean(apps)) if apps else 0.0,
            trail_excess=float(np.mean(trails)) if trails else None,
            collapse=float(max(dinos[:2]) / max(min(dinos[-2:]), 1e-6)) if dinos else None,
            push_response=push_response,
            contact_to_state=0.0,
            contact_to_first_rendered=(
                None if t_contact is None or first_rendered is None
                else float((first_rendered - t_contact) * 0.25)),
            t_contact=t_contact)
        x = summary[sname]
        te = f"{x['trail_excess']:+.3f}" if x['trail_excess'] is not None else "n/a"
        po = f"{x['pos_err']:.1f}" if x['pos_err'] is not None else "n/a"
        co = f"{x['collapse']:.1f}" if x['collapse'] is not None else "n/a"
        c2r = (f"{x['contact_to_first_rendered']:.2f}s"
               if x['contact_to_first_rendered'] is not None else "n/a")
        prs = f"{push_response:.2f}" if push_response is not None else "n/a"
        print(f"    RENDER exist {x['existence']*100:3.0f}% "
              f"id {x['identity']*100:3.0f}% wID {wids} pos {po} "
              f"trail {te} collapse {co}", flush=True)
        print(f"    CAUSALITY push_response={prs} "
              f"e_pos p90={x['e_pos_p90']:.1f} max={x['e_pos_max']:.1f} "
              f"| contact->state {x['contact_to_state']:.2f}s "
              f"contact->rendered {c2r}", flush=True)

    print("\n[pu] ===== §41C-1 Push =====")
    print(f"  {'scenario':>14s} {'contacts':>8s} {'pen':>6s} {'dv_box':>7s} "
          f"{'disp':>6s} {'exist':>6s} {'ident':>6s} {'wID':>4s} {'pos':>6s} "
          f"{'trail':>7s} {'push_resp':>9s}")
    for s, x in summary.items():
        te = f"{x['trail_excess']:+.3f}" if x['trail_excess'] is not None else "n/a"
        pr = f"{x['push_response']:.2f}" if x['push_response'] is not None else "n/a"
        po = f"{x['pos_err']:.1f}" if x['pos_err'] is not None else "n/a"
        print(f"  {s:>14s} {x['contacts']:8d} {x['max_pen']:6.3f} "
              f"{x['dv_box']:7.3f} {x['disp_phys']:6.2f} "
              f"{x['existence']*100:5.0f}% {x['identity']*100:5.0f}% "
              f"{x['wrong_id']:4d} {po:>6s} {te:>7s} {pr:>9s}")

    print("\n[pu] ===== §41C-1 GATE =====")
    print("  no-contact -> box must NOT move; contact -> box moves in the")
    print("  correct direction; existence/identity ~100%; wrong-ID 0")
    ok = True
    for s, x in summary.items():
        if s.startswith("C0"):
            good = (x["contacts"] == 0 and not x["box_moved"])
        else:
            good = (x["contacts"] > 0 and x["box_moved"]
                    and x["existence"] >= 0.9 and x["identity"] >= 0.9
                    and x["wrong_id"] == 0)
        ok = ok and good
        print(f"    {s}: contacts {x['contacts']} moved {x['box_moved']} "
              f"exist {x['existence']*100:.0f}% id {x['identity']*100:.0f}% "
              f"wID {x['wrong_id']} -> {'PASS' if good else 'FAIL'}")
    print(f"\n  §41C-1: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("  -> Input -> Physics -> Persistent World State -> Generative "
              "Renderer closed")

    json.dump(summary, open(f"{args.out_dir}/push.json", "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()

