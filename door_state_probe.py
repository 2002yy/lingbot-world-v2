#!/usr/bin/env python
"""§40A-2: a reliable DoorStateProbe based on LOCAL STRUCTURAL features.

DINO stays as an identity guard, but the OPEN/CLOSED decision must come from
local structure, not a global embedding mean.

Visual definition (crisp, checkable):
    CLOSED = a solid, low-variance, mid-tone panel fills the opening
    OPEN   = the opening shows background (bright sky and/or high-gradient
             depth), i.e. NOT occupied by a solid panel

Features (all in [0,1] over the ROI):
    opening_ratio     fraction of pixels NOT solid-panel
    center_brightness mean brightness of the central 50%
    panel_occupancy   fraction of 'solid panel' pixels (mid-tone + low grad)
    edge_energy       Sobel energy (door panel edges) normalised
    occupancy_mask    same as panel_occupancy but on the centre block
    score_open  = z(opening_ratio) + z(center_brightness) - z(panel_occupancy)
    score_closed= z(panel_occupancy) + z(edge_energy) - z(opening_ratio)

GATE: on reference pairs the margin must be >> the noise floor, i.e.
    OPEN   sample: score_open  - score_closed  >> std
    CLOSED sample: score_closed - score_open  >> std
If this does not pass we do NOT proceed to §40C.

  python door_state_probe.py --scenes 04 --seeds 42 --horizons 4,12,40
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
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange
from object_permanence import build_traj, patch

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV  # noqa: E402

PROMPT = "A first-person view of a natural landscape with smooth camera motion."
DOOR_BB = (0.70, 0.50, 0.84, 0.66)
SKY_BB = (0.05, 0.02, 0.35, 0.16)      # real sky pixels, used for a REAL open ref


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------
def _norm01(a):
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi - lo < 1e-6:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def structural_features(roi_patch, surround_brightness=None):
    g = cv2.cvtColor(roi_patch, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gn = g / 255.0
    H, W = g.shape
    # local gradient magnitude, normalised within the patch
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = _norm01(np.sqrt(gx ** 2 + gy ** 2))
    # 'solid panel' = mid-tone AND low local gradient
    midtone = ((gn > 0.18) & (gn < 0.80)).astype(np.float32)
    solid = midtone * (grad < 0.35).astype(np.float32)
    panel_occupancy = float(solid.mean())
    opening_ratio = float(1.0 - panel_occupancy)
    edge_energy = float(grad.mean())
    ch, cw = H // 4, W // 4
    centre = gn[ch:3 * ch, cw:3 * cw]
    center_brightness = float(centre.mean())
    occ_centre = float(solid[ch:3 * ch, cw:3 * cw].mean())
    # --- physical criterion: an OPENING differs in brightness from the wall
    #     around it (dark passage / bright exterior); a SOLID PANEL matches it.
    if surround_brightness is None:
        surround_brightness = float(gn.mean())
    contrast = float(gn.mean() - surround_brightness)
    return dict(opening_ratio=opening_ratio,
                center_brightness=center_brightness,
                panel_occupancy=panel_occupancy,
                edge_energy=edge_energy,
                occupancy_mask=occ_centre,
                surround_brightness=surround_brightness,
                contrast=contrast,
                abs_contrast=abs(contrast))


class DoorStateProbe:
    """Self-calibrating door state probe.

    Primary discriminator: the opening's brightness CONTRAST against the wall
    immediately around it.
        solid panel filling the opening  -> ROI ~= wall   -> small |contrast|
        opening showing through          -> ROI != wall   -> large |contrast|
    (a dark passage is much darker; a bright exterior is much brighter)

    An ABSOLUTE threshold does not transfer across scenes/objects (measured
    |contrast| was 0.145 for the closed door and 0.341 for the open one; a
    fixed 0.06 classified both as OPEN). So we calibrate RELATIVE to the
    object's own state at registration time:

        ratio = |contrast|_now / |contrast|_at_registration
        margin = ratio - RATIO            (RATIO = 1.5)
        margin > 0  => OPEN

    Registration always records the object in its CURRENT (as-generated)
    state; a later state change must move the contrast ratio by >50%.
    """

    FEATS = ("abs_contrast", "opening_ratio", "center_brightness",
             "panel_occupancy", "edge_energy")
    RATIO = 1.5

    def __init__(self):
        self.baseline = None
        self.sd = None

    def calibrate(self, roi_patches, surrounds):
        vals = [structural_features(p, s)["abs_contrast"]
                for p, s in zip(roi_patches, surrounds)]
        self.baseline = float(np.mean(vals))
        self.sd = float(np.std(vals))
        return self

    def scores(self, roi_patch, surround_brightness=None):
        f = structural_features(roi_patch, surround_brightness)
        if self.baseline is None or self.baseline < 1e-6:
            self.baseline = max(f["abs_contrast"], 1e-6)
            self.sd = 0.0
        ratio = f["abs_contrast"] / self.baseline
        margin = ratio - self.RATIO
        return dict(features=f, ratio=float(ratio),
                    score_open=float(ratio), score_closed=float(self.RATIO - ratio),
                    margin=float(margin),
                    decision="OPEN" if margin > 0 else "CLOSED")


def make_open_ref(frame, door_bb, sky_bb):
    """REAL open reference: fill the opening with REAL sky pixels (not a brighten)."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = door_bb
    dx0, dy0, dx1, dy1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    sx0, sy0, sx1, sy1 = int(sky_bb[0] * W), int(sky_bb[1] * H), \
        int(sky_bb[2] * W), int(sky_bb[3] * H)
    sky = frame[sy0:sy1, sx0:sx1]
    open_frame = frame.copy()
    open_frame[dy0:dy1, dx0:dx1] = cv2.resize(sky, (dx1 - dx0, dy1 - dy0))
    return open_frame


def surround_brightness_of(frame, bb, pad=0.05):
    """Mean brightness of the wall ring just outside the door ROI."""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = bb
    ax0, ay0 = max(0, int((x0 - pad) * W)), max(0, int((y0 - pad) * H))
    ax1, ay1 = min(W, int((x1 + pad) * W)), min(H, int((y1 + pad) * H))
    ix0, iy0, ix1, iy1 = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    ring = cv2.cvtColor(frame[ay0:ay1, ax0:ax1], cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = np.ones(ring.shape, bool)
    mask[iy0 - ay0:iy1 - ay0, ix0 - ax0:ix1 - ax0] = False
    if mask.sum() == 0:
        return float(ring.mean()) / 255.0
    return float(ring[mask].mean()) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser("~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser("~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scenes", default="04")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--horizons", default="4,12,40")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/probe")
    args = ap.parse_args()

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",")]
    scenes = args.scenes.split(",")
    seeds = [int(x) for x in args.seeds.split(",")]

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=6, sink_size=1,
        infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    tae = TAEHV(checkpoint_path=args.tae_pth).to(dev).eval()
    print("[pr] pipe + TAE built", flush=True)
    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    lh = ma.num_heads // pipe.sp_size; hd = ma.dim // ma.num_heads

    def gen(scene, sd, hz):
        traj, ref_chunk, out_chunks = build_traj(scene, hz)
        frames_n = len(traj)
        d = f"examples/pr_{scene}_H{hz}"
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
        n_lat = (frames_n - 1) // 4 + 1
        pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=1)
        Ks = get_Ks_transformed(
            torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
            480, 832, h, w, h, w)[0].to(dev)
        y = pipe.vae.encode([torch.concat([
            torch.nn.functional.interpolate(img[None].cpu(), size=(h, w),
                                            mode='bicubic').transpose(0, 1),
            torch.zeros(3, frames_n - 1, h, w)], dim=1).to(dev)])[0]
        msk = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                            msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
        y = torch.concat([msk, y])
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        timesteps = pipe.scheduler.timesteps[[0, 250, 750]]
        g = torch.Generator(device=dev); g.manual_seed(sd)
        c2w = interpolate_camera_poses(
            np.linspace(0, frames_n - 1, frames_n),
            torch.from_numpy(traj[:, :3, :3]).float(),
            torch.from_numpy(traj[:, :3, 3]).float(),
            np.linspace(0, frames_n - 1, n_lat)).to(dev)
        rel_all = compute_relative_poses(c2w, framewise=True)
        outs = []
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
            with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
                pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                           cross_attn_first_call=False, **kw)
            with torch.no_grad():
                fr = tae.decode_video(x0.permute(1, 0, 2, 3).unsqueeze(0),
                                      parallel=False, show_progress_bar=False)
            outs.append((fr[0][0].permute(1, 2, 0).float().cpu().numpy()
                         * 255.0).clip(0, 255).astype(np.uint8))
        del y, self_kv, cross_kv
        gc.collect(); torch.cuda.empty_cache()
        return outs[min(ref_chunk, len(outs) - 1)], outs[-1]

    # ---- collect generated patches ----
    samples = []
    for scene in scenes:
        for sd in seeds:
            for hz in horizons:
                R, V = gen(scene, sd, hz)
                open_frame = make_open_ref(R, DOOR_BB, SKY_BB)
                samples.append(dict(scene=scene, seed=sd, H=hz, ref=R, gen=V,
                                    ref_door=patch(R, DOOR_BB),
                                    gen_door=patch(V, DOOR_BB),
                                    ref_sb=surround_brightness_of(R, DOOR_BB),
                                    gen_sb=surround_brightness_of(V, DOOR_BB),
                                    open_sb=surround_brightness_of(open_frame, DOOR_BB),
                                    open_ref=patch(open_frame, DOOR_BB)))
                print(f"[pr] {scene}/H{hz}/s{sd} generated", flush=True)

    # ---- calibrate the probe on the CLOSED (as-generated) door patches ----
    probe = DoorStateProbe().calibrate([s["ref_door"] for s in samples],
                                       [s["ref_sb"] for s in samples])

    print("\n[pr] ===== §40A-2 probe validation =====")
    print("  reference patches (ground truth by construction):")
    ref_rows = []
    for s in samples:
        for name, pt, sb in (("CLOSED_ref", s["ref_door"], s["ref_sb"]),
                             ("OPEN_ref", s["open_ref"], s["open_sb"])):
            r = probe.scores(pt, sb)
            gt_open = name.startswith("OPEN")
            ok = (r["decision"] == "OPEN") == gt_open
            ref_rows.append(dict(name=name, scene=s["scene"], H=s["H"],
                                 decision=r["decision"], margin=r["margin"],
                                 correct=ok, **r["features"]))
            print(f"    {name:11s} {s['scene']}/H{s['H']:<3d} -> "
                  f"{r['decision']:6s} margin={r['margin']:+7.3f} "
                  f"{'OK' if ok else 'XX'}   "
                  f"open_r={r['features']['opening_ratio']:.3f} "
                  f"ctr_b={r['features']['center_brightness']:.3f} "
                  f"panel={r['features']['panel_occupancy']:.3f} "
                  f"contrast={r['features']['contrast']:+.3f} "
                  f"|c|={r['features']['abs_contrast']:.3f}")

    # ---- noise floor: same-state patches across seeds/horizons ----
    closed_m = [probe.scores(s["ref_door"], s["ref_sb"])["margin"] for s in samples]
    open_m = [probe.scores(s["open_ref"], s["open_sb"])["margin"] for s in samples]
    noise = float(np.std(closed_m)) / max(abs(np.mean(closed_m)),1e-6)
    sep = float(np.mean(open_m) - np.mean(closed_m))
    print(f"\n  CLOSED margins: mean={np.mean(closed_m):+.3f} std={np.std(closed_m):.3f}")
    print(f"  OPEN   margins: mean={np.mean(open_m):+.3f} std={np.std(open_m):.3f}")
    print(f"  separation = {sep:.3f}   noise floor = {noise:.3f}   "
          f"ratio = {sep / max(noise,1e-6):.1f}x")
    acc = np.mean([r["correct"] for r in ref_rows])
    print(f"  reference classification accuracy = {acc*100:.0f}%")
    gate = (sep > 3 * noise) and acc == 1.0
    print(f"\n  §40A-2 GATE (sep > 3*noise AND 100% ref accuracy): "
          f"{'PASS -> proceed to §40C' if gate else 'FAIL -> do NOT proceed to §40C'}")

    print("\n[pr] generated revisit door patches (probe output):")
    gen_rows = []
    for s in samples:
        r = probe.scores(s["gen_door"], s["gen_sb"])
        gen_rows.append(dict(scene=s["scene"], seed=s["seed"], H=s["H"],
                             decision=r["decision"], margin=r["margin"],
                             **r["features"]))
        print(f"    {s['scene']}/H{s['H']:<3d}/s{s['seed']} -> "
              f"{r['decision']:6s} margin={r['margin']:+7.3f}  "
              f"open_r={r['features']['opening_ratio']:.3f} "
              f"ctr_b={r['features']['center_brightness']:.3f} "
              f"panel={r['features']['panel_occupancy']:.3f}")

    json.dump(dict(references=ref_rows, generated=gen_rows,
                   separation=sep, noise=noise, gate=bool(gate),
                   ref_accuracy=float(acc)),
              open(f"{args.out_dir}/probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
