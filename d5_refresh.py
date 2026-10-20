#!/usr/bin/env python
"""§40D-D5: architecture convergence.

D4 settled the direction: a state-specific anchor is a better control surface
than an additive state delta (canonicalised OPEN anchor pulled existence from
50% back to 100%), but the base model will NOT carry the gameplay state for
you (rel 2.78 -> 0.14 after injection stops). So stop trying to make the model
absorb the state. The production shape is:

    Persistent ObjectState   <- the durable truth
            |
            +- ID / Transform / gameplay_state
            +- state_anchors[state]
                       |
                       v
              World Model Renderer   <- redraws the authoritative fact
                                        whenever the object is in view

Two things are tested here.

1. MULTI-STATE IDENTITY MATCHING (§39 fix).
   Every state prototype carries {latent, feature}; identity takes the BEST
   score over ALL state prototypes, so a door drawn OPEN or CLOSED is both
   recognised as door_01. State is judged separately against the
   authoritative gameplay_state. (Matching only the active state would
   re-break identity whenever the model renders the wrong state.)

2. ANCHOR REFRESH POLICY.
       R0  inject once at the revisit
       R1  refresh every 4 chunks
       R2  refresh every 2 chunks
       R3  refresh every visible chunk
   The goal is the LOWEST refresh rate that sustains the correct state, not
   "the model holds it forever".

  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python d5_refresh.py --scene 04 --seed 42 --out_chunks 40 --tail 10
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

POLICIES = {"R0": 0, "R1": 4, "R2": 2, "R3": 1}   # refresh period in chunks


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
    ap.add_argument("--canon_chunks", type=int, default=4)
    ap.add_argument("--policies", default="R0,R1,R2,R3")
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--tae_pth", default=os.path.expanduser("~/ai/taehv/taew2_1.pth"))
    ap.add_argument("--out_dir", default="output/d5")
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
    print("[d5] pipe + TAE built", flush=True)
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
    d = f"examples/d5_{scene}_O{args.out_chunks}_T{args.tail}"
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
    print(f"[d5] n_lat={n_lat} ref={ref_chunk} revisit={revisit} "
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
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()
    print(f"[d5] setup done, driver free {vram_free_mb():.0f}MiB", flush=True)

    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, n_lat)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    def run(y_cond, anchor_patch, period, max_chunks=None):
        """period=0 -> inject only at the revisit; period=k -> also refresh on
        every k-th observe chunk (k=1 -> every visible chunk)."""
        self_kv = pipe._initialize_self_kv_cache(
            num_layers=ma.num_layers, shape=[1, kv_size, lh_, hd],
            dtype=dtype, device=dev)
        cross_kv = pipe._initialize_crossattn_cache(
            num_layers=ma.num_layers, shape=[1, 512, ma.num_heads, hd],
            dtype=dtype, device=dev)
        pipe._cross_attn_initialized = False
        g = torch.Generator(device=dev); g.manual_seed(sd)
        outs, latents, injections = [], [], []
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
            inject = False
            if anchor_patch is not None:
                if cid in revisit:
                    inject = True
                elif period > 0 and cid in observe:
                    inject = ((cid - revisit[-1] - 1) % period == 0)
            if inject:
                x0[:, :, dy0:dy1, dx0:dx1] = anchor_patch
            injections.append(inject)
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
        return outs, latents, injections

    # ---------- D0: reference, CLOSED anchor, canonicalised OPEN anchor ----------
    print("\n[d5] === D0 (no anchor) ===", flush=True)
    D0_frames, D0_lat, _ = run(y, None, 0)
    ref_frame = D0_frames[ref_chunk]
    z_closed = D0_lat[ref_chunk][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    ref_door = patch(ref_frame, DOOR_BB)
    c_reg = abs(structural_features(
        ref_door, surround_brightness_of(ref_frame, DOOR_BB))["contrast"]) + 1e-9

    print(f"\n[d5] === canonicalisation ({args.canon_chunks} chunks on R_open) ===",
          flush=True)
    canon_frames, canon_lat, _ = run(y_open, None, 0, max_chunks=args.canon_chunks)
    ci = min(2, len(canon_lat) - 1)
    z_open = canon_lat[ci][:, :, dy0:dy1, dx0:dx1].clone().to(dev)
    canon_door = patch(canon_frames[ci], DOOR_BB)
    print(f"[d5] canonicalised OPEN anchor resp="
          f"{abs(structural_features(canon_door, surround_brightness_of(canon_frames[ci], DOOR_BB))['contrast'])/c_reg:.2f}",
          flush=True)

    # ---------- §40D-D5 store: multi-state prototypes ----------
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
                               latent=z_closed.cpu().numpy(),
                               feature=dino_np(ref_door))
    store.set_anchor_for_state(door.persistent_id, "open",
                               latent=z_open.cpu().numpy(),
                               feature=dino_np(canon_door))
    store.set_state(door.persistent_id, open=True)
    print(f"[d5] store: door_01 multi-state prototypes "
          f"{list(store.get(door.persistent_id).state_anchors.keys())}, "
          f"active={store.get(door.persistent_id).active_anchor_key}", flush=True)

    results, inject_log = {"D0": D0_frames}, {}
    for pname in args.policies.split(","):
        if pname not in POLICIES:
            continue
        print(f"\n[d5] === {pname} (refresh period {POLICIES[pname]} chunks) ===",
              flush=True)
        f, _, inj = run(y, z_open, POLICIES[pname])
        results[pname] = f
        inject_log[pname] = inj

    # ---------- gates ----------
    def resp_of(f):
        sb = surround_brightness_of(f, DOOR_BB)
        return abs(structural_features(patch(f, DOOR_BB), sb)["contrast"]) / c_reg

    base_ob = float(np.mean([resp_of(results["D0"][c]) for c in observe]))
    print(f"\n[d5] D0 observe baseline resp = {base_ob:.2f}", flush=True)

    print("\n[d5] ===== §40D-D5 gates =====")
    print(f"  {'policy':>6s} {'chunk':>5s} {'phase':>8s} {'inj':>3s} {'DINO':>6s} "
          f"{'resp':>6s} {'rel':>5s} {'pose':>6s} {'id':>4s} {'proto':>5s} "
          f"{'Ex':>3s} {'Id':>3s} {'St':>3s}")
    summary = {}
    for gname, frames in results.items():
        rows = []
        inj = inject_log.get(gname)
        for cid in revisit + observe:
            f = frames[cid]
            sim, loc = best_match_loc(ref_door, f, DOOR_BB, dino_np)
            resp = resp_of(f)
            exp = (int(DOOR_BB[0] * f.shape[1]), int(DOOR_BB[1] * f.shape[0]))
            pe = math.hypot(loc[0] - exp[0], loc[1] - exp[1]) if loc else 1e3
            cand = dict(anchor=dino_np(patch(f, DOOR_BB)),
                        uv=((loc[0] + 0.5 * (DOOR_BB[2] - DOOR_BB[0]) * f.shape[1]) / f.shape[1],
                            (loc[1] + 0.5 * (DOOR_BB[3] - DOOR_BB[1]) * f.shape[0]) / f.shape[0]),
                        world_transform=np.eye(4), bounds=[1, 2, 0.2])
            dec = store.reacquire([cand], t=cid * 0.25)[0]
            rel = resp / max(base_ob, 1e-6)
            ex = bool(sim > 0.6 and resp > 0.35)
            ident = bool(dec["id"] == door.persistent_id)
            rows.append(dict(chunk=cid, phase="revisit" if cid in revisit else "observe",
                             injected=bool(inj[cid]) if inj else False,
                             dino=sim, resp=resp, rel=rel, pose=pe,
                             match_id=dec["id"], proto=dec.get("matched_proto"),
                             existence=ex, identity=ident,
                             state="OPEN" if rel > 1.5 else "CLOSED"))
            print(f"  {gname:>6s} {cid:5d} {rows[-1]['phase']:>8s} "
                  f"{'Y' if rows[-1]['injected'] else '.':>3s} {sim:6.3f} {resp:6.2f} "
                  f"{rel:5.2f} {pe:6.1f} {str(dec['id']):>4s} "
                  f"{str(dec.get('matched_proto')):>5s} "
                  f"{'Y' if ex else '.':>3s} {'Y' if ident else '.':>3s} "
                  f"{rows[-1]['state']:>3s}")
        summary[gname] = rows

    print("\n[d5] ===== summary =====")
    print(f"  {'policy':>6s} {'Existence':>10s} {'Identity':>9s} {'DINO':>6s} "
          f"{'rel_ob':>7s} {'OPEN':>6s} {'wrongID':>8s} {'collapse':>9s} "
          f"{'injects':>8s}")
    pareto = {}
    for gname, rows in summary.items():
        ob = [r for r in rows if r["phase"] == "observe"]
        rv = [r for r in rows if r["phase"] == "revisit"]
        resps = [r["resp"] for r in ob]
        collapse = float(max(resps[:2]) / max(min(resps[-2:]), 1e-6))
        wids = sum(1 for r in rows if r["match_id"] not in (None, door.persistent_id))
        ninj = sum(1 for r in rows if r["injected"])
        pareto[gname] = dict(
            existence=float(np.mean([r["existence"] for r in rv])),
            identity=float(np.mean([r["identity"] for r in rv])),
            dino=float(np.mean([r["dino"] for r in ob])),
            rel=float(np.mean([r["rel"] for r in ob])),
            state_open=float(np.mean([r["state"] == "OPEN" for r in ob])),
            wrong_id=wids, collapse=collapse, injections=ninj)
        p = pareto[gname]
        print(f"  {gname:>6s} {p['existence']*100:9.0f}% {p['identity']*100:8.0f}% "
              f"{p['dino']:6.3f} {p['rel']:7.2f} {p['state_open']*100:5.0f}% "
              f"{wids:8d} {collapse:9.1f} {ninj:8d}")

    print("\n[d5] ===== §40D-D5 GATE =====")
    print("  cross-T50: identity/transform/state recovered from ObjectState,")
    print("  sustained by sparse local anchor refresh, no full-frame replay,")
    print("  no in-loop VAE encode")
    best = None
    for gname, p in pareto.items():
        if gname == "D0":
            continue
        ok = (p["existence"] >= 0.9 and p["identity"] >= 0.9
              and p["state_open"] >= 0.75 and p["wrong_id"] == 0
              and p["collapse"] <= 5.0)
        print(f"    {gname}: existence {p['existence']*100:.0f}% "
              f"identity {p['identity']*100:.0f}% OPEN {p['state_open']*100:.0f}% "
              f"wrongID {p['wrong_id']} collapse {p['collapse']:.1f} "
              f"injects {p['injections']} -> {'PASS' if ok else 'FAIL'}")
        if ok and (best is None or p["injections"] < best[1]):
            best = (gname, p["injections"])
    if best:
        print(f"\n  §40D-D5: PASS. Lowest sufficient refresh = {best[0]} "
              f"({best[1]} injections over {len(revisit)+len(observe)} chunks)")
    else:
        print("\n  §40D-D5: FAIL (no policy sustains identity AND state)")

    json.dump(pareto, open(f"{args.out_dir}/d5.json", "w"), indent=1, default=float)
    for gname, frames in results.items():
        np.save(f"{args.out_dir}/{gname}_frames.npy", np.stack(frames))


if __name__ == "__main__":
    main()
