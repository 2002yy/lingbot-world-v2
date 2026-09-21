#!/usr/bin/env python
"""P2d-1a2-r3: elementwise exactness matrix, INLINE comparison (no corpus stored).

WHY NOT r2's APPROACH
---------------------
r2 stored the corpus on CPU, but every sample is THREE full-size tensors (a/b/c
are all [1,1881,1536], ~5.8 MB each), so 40 samples x 5 chains is ~3.5 GB
resident plus roughly as much again during capture. That OOM-killed the WSL VM.

Here there is no corpus at all: reference and candidate are computed at the
capture point and only counters accumulate (equal count, max|diff|, worst meta).
Memory stays flat.

FIXES CARRIED OVER FROM THE FIRST RUN
  * cam candidate is algebraically correct: addcmul(cam_shift, x, 1+cam_scale).
    (torch.addcmul(input,t1,t2) = input + t1*t2, so the earlier
     addcmul(x, 1+cs, cam_shift) computed x + (1+cs)*cam_shift -- a different
     expression, which is why `compile` and `addcmul` showed identical error.)
  * `1 + scale` is computed once and shared, so only the fusion varies.
  * stratified sampling over chunk x forward x depth:
        chunk 0: forwards 0-3 x blocks [0,4,8,12,17,21,25,29]
        chunk 1: forwards 0-3 x blocks [7,23]
  * capture armed only AFTER prewarm (prewarm's dummy pass has a different shape
    and poisoned the first run's corpus).
  * every dtype reported, so a cam failure can legitimately be discussed as a
    promotion/broadcast difference.
"""
import argparse
import collections
import gc
import hashlib
import json
import os
import shutil

import numpy as np
import torch
from PIL import Image

import wan
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

BLOCKS_CH0 = [0, 4, 8, 12, 17, 21, 25, 29]
BLOCKS_CH1 = [7, 23]
CHAINS = ("mod1", "mod2", "resA", "resB", "cam")
NEG = 6

STAT = collections.defaultdict(
    lambda: dict(n=0, eq=0, maxd=0.0, worst=None, dt={}, shape=None,
                 ref_dt=None, cand_dt=None, cov=set()))
NEGSTAT = collections.defaultdict(lambda: dict(n=0, eq=0, maxd=0.0))
HOLD = {}
CAP = {"on": False, "cid": 0, "fwd": 0}
FUSED = {"fn": None}


def want(cid, bi):
    return (cid == 0 and bi in BLOCKS_CH0) or (cid == 1 and bi in BLOCKS_CH1)


def check(chain, a, b, c, meta):
    if not CAP["on"]:
        return
    s = STAT[chain]
    with torch.no_grad():
        r = a + b * c
        cand = torch.addcmul(a, b, c)
    s["n"] += 1
    if torch.equal(r, cand):
        s["eq"] += 1
    d = (r.float() - cand.float()).abs().max().item()
    if d > s["maxd"]:
        s["maxd"] = d
        s["worst"] = f"sample {s['n']-1} {meta} max|d|={d:.3e}"
    if not s["dt"]:
        s["dt"] = {k: str(t.dtype).replace("torch.", "")
                   for k, t in zip(("a", "b", "c"), (a, b, c))}
        s["shape"] = list(a.shape)
        s["ref_dt"] = str(r.dtype).replace("torch.", "")
        s["cand_dt"] = str(cand.dtype).replace("torch.", "")
    s["cov"].add((meta["chunk"], meta["fwd"], meta["block"]))
    if FUSED["fn"] is not None and NEGSTAT[chain]["n"] < NEG:
        try:
            with torch.no_grad():
                cd = FUSED["fn"](a, b, c)
            NEGSTAT[chain]["n"] += 1
            if torch.equal(r, cd):
                NEGSTAT[chain]["eq"] += 1
            NEGSTAT[chain]["maxd"] = max(
                NEGSTAT[chain]["maxd"],
                (r.float() - cd.float()).abs().max().item())
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--mode", default="default")
    ap.add_argument("--out_dir", default="output/p2d1a2r3")
    args = ap.parse_args()

    os.environ["LINGBOT_FP8"] = "1"
    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd = args.scene, args.seed
    CS = args.chunk_size

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype
    print(f"[r3] layers={pipe.model.config.num_layers} dtype={pdt}", flush=True)

    def sh(tag):
        def hook(mod, ai, out):
            HOLD[tag] = out.detach() if torch.is_tensor(out) else out[0].detach()
        return hook

    def ph(tag):
        def hook(mod, ai):
            if ai and torch.is_tensor(ai[0]):
                HOLD[tag] = ai[0].detach()
        return hook

    for i, blk in enumerate(pipe.model.blocks):
        blk.norm1.register_forward_hook(sh(f"{i}:n1"))
        blk.norm2.register_forward_hook(sh(f"{i}:n2"))
        blk.self_attn.register_forward_hook(sh(f"{i}:yattn"))
        blk.cross_attn.register_forward_hook(sh(f"{i}:ycross"))
        blk.ffn.register_forward_hook(sh(f"{i}:yffn"))
        blk.register_forward_pre_hook(ph(f"{i}:xin"))

        def mk(i_):
            def h1(mod, a, out):
                HOLD[f"{i_}:cs"] = out.detach()

            def h2(mod, a, out):
                HOLD[f"{i_}:ct"] = out.detach()
            return h1, h2
        h1, h2 = mk(i)
        blk.cam_scale_layer.register_forward_hook(h1)
        blk.cam_shift_layer.register_forward_hook(h2)

    orig_cb = mf.CausalWanAttentionBlock.forward

    def make_patch(bi, blk):
        # instance attribute, so it must not take `self`
        def patched(x, e, *a, **kw):
            out = orig_cb(blk, x, e, *a, **kw)
            if CAP["on"] and want(CAP["cid"], bi):
                meta = dict(chunk=CAP["cid"], fwd=CAP["fwd"], block=bi)
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    ec = (blk.modulation.unsqueeze(0) + e).chunk(6, dim=2)
                n1 = HOLD.get(f"{bi}:n1")
                n2 = HOLD.get(f"{bi}:n2")
                xin = HOLD.get(f"{bi}:xin")
                yattn = HOLD.get(f"{bi}:yattn")
                ycross = HOLD.get(f"{bi}:ycross")
                yffn = HOLD.get(f"{bi}:yffn")
                cs = HOLD.get(f"{bi}:cs")
                ct = HOLD.get(f"{bi}:ct")
                if n1 is not None:
                    check("mod1", n1, 1 + ec[1].squeeze(2), ec[0].squeeze(2), meta)
                if n2 is not None:
                    check("mod2", n2, 1 + ec[4].squeeze(2), ec[3].squeeze(2), meta)
                if xin is not None and yattn is not None:
                    check("resA", xin, yattn, ec[2].squeeze(2), meta)
                    if cs is not None:
                        xA = xin + yattn * ec[2].squeeze(2)
                        check("cam", xA, 1.0 + cs, ct, meta)
                        if ycross is not None and yffn is not None:
                            xCross = (1.0 + cs) * xA + ct + ycross
                            check("resB", xCross, yffn, ec[5].squeeze(2), meta)
                del n1, n2, xin, yattn, ycross, yffn, cs, ct, ec
            return out
        return patched

    for i, blk in enumerate(pipe.model.blocks):
        blk.forward = make_patch(i, blk)

    def ref_form(a, b, c):
        return a + b * c
    try:
        FUSED["fn"] = torch.compile(ref_form, mode=args.mode, fullgraph=True)
        print("[r3] compile negative control built", flush=True)
    except Exception as e:
        print(f"[r3] compile unavailable: {e}", flush=True)

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()
    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    frames_n = (args.frames - 1) // 4 * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))
    n_test = min(args.chunks, lat_f // CS)

    d = f"examples/p2d1a2r3_{scene}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"examples/{scene}/intrinsics.npy", f"{d}/intrinsics.npy")
    shutil.copy(f"examples/{scene}/image.jpg", f"{d}/image.jpg")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    th = int(np.sqrt(W * H * (480 / 832)) // 8 * 8)
    tw = int(np.sqrt(W * H / (480 / 832)) // 8 * 8)
    img = (torch.nn.functional.interpolate(
        torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None].float(),
        size=(th, tw), mode='bicubic').squeeze(0) / 255.0 - 0.5) / 0.5
    h, w = img.shape[1:]
    lat_h, lat_w = h // vae_stride[1], w // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = CS * fsl
    kv_size = fsl * args.local_attn_size
    print(f"[r3] expected M = {CS}*{fsl} = {max_seq_len}", flush=True)

    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=CS)
    from wan.modules.model_fast import bump_cam_epoch
    bump_cam_epoch()
    CAP["on"] = True
    print("[r3] capture armed (after prewarm)", flush=True)

    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, h, w, h, w)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        img[None].transpose(0, 1).to(dev),
        torch.zeros(3, frames_n - 1, h, w, device=dev)], dim=1)])[0]
    msk = torch.ones(1, frames_n, lat_h, lat_w, device=dev)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                        msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]
    y = torch.concat([msk, y]).detach()
    del img
    pipe.vae = None
    gc.collect(); torch.cuda.empty_cache()

    p = np.load(f"examples/{scene}/poses.npy")
    traj = np.tile(p, (frames_n // len(p) + 1, 1, 1))[:frames_n]
    c2w = interpolate_camera_poses(
        np.linspace(0, frames_n - 1, frames_n),
        torch.from_numpy(traj[:, :3, :3]).float(),
        torch.from_numpy(traj[:, :3, 3]).float(),
        np.linspace(0, frames_n - 1, lat_f)).to(dev)
    rel_all = compute_relative_poses(c2w, framewise=True)
    timesteps = pipe.scheduler.timesteps[[0, 250, 750]]

    self_kv = pipe._initialize_self_kv_cache(
        num_layers=ma.num_layers,
        shape=[1, kv_size, ma.num_heads // pipe.sp_size, ma.dim // ma.num_heads],
        dtype=dtype, device=dev, python_metadata=pipe._py_cache_meta)
    cross_kv = pipe._initialize_crossattn_cache(
        num_layers=ma.num_layers,
        shape=[1, 512, ma.num_heads, ma.dim // ma.num_heads],
        dtype=dtype, device=dev, python_metadata=pipe._py_cache_meta)

    def reset():
        for c in self_kv:
            c["global_end_index"] = 0; c["local_end_index"] = 0
            c["k"].zero_(); c["v"].zero_()
        for c in cross_kv:
            c["is_init"] = False
            c["k"].zero_(); c["v"].zero_()

    gg = torch.Generator(device=dev); gg.manual_seed(sd)
    reset(); pipe._cross_attn_initialized = False
    for cid in range(n_test):
        CAP["cid"] = cid
        c0 = cid * CS
        cur = torch.randn(16, CS, lat_h, lat_w, generator=gg, device=dev)
        pp = get_plucker_embeddings(rel_all[c0:c0 + CS], Ks[None], h, w)
        pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                       c1=int(h // lat_h), c2=int(w // lat_w))[None]
        plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=CS,
                        h=lat_h, w=lat_w).to(pdt)
        kw = {"context": [pipe._t5_cache[key][0]], "seq_len": max_seq_len,
              "y": [y.split(CS, dim=1)[cid]],
              "dit_cond_dict": {"c2ws_plucker_emb": plk.chunk(1, dim=0)},
              "kv_cache": self_kv, "crossattn_cache": cross_kv,
              "current_start": cid * CS * fsl,
              "max_attention_size": kv_size, "frame_seqlen": fsl}
        for ti in range(len(timesteps)):
            CAP["fwd"] = ti
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
                        x0, torch.randn(x0.shape, generator=gg,
                                        device=dev, dtype=x0.dtype),
                        timesteps[ti + 1])
        CAP["fwd"] = 3
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
    CAP["on"] = False
    for i, blk in enumerate(pipe.model.blocks):
        blk.forward = orig_cb.__get__(blk, type(blk))

    print(f"\n[r3] ===== elementwise exactness matrix (inline) =====")
    print(f"   {'chain':<7} {'exact':>9} {'max|diff|':>12}  "
          f"{'a/b/c dtype':<22} {'ref':<9} {'cand':<9} shape")
    report = {}
    for ch in CHAINS:
        s = STAT[ch]
        if not s["n"]:
            print(f"   {ch:<7} {'no data':>9}")
            continue
        dts = "/".join(s["dt"].get(k, "?") for k in ("a", "b", "c"))
        print(f"   {ch:<7} {s['eq']:>4}/{s['n']:<4} {s['maxd']:>12.3e}  "
              f"{dts:<22} {s['ref_dt']:<9} {s['cand_dt']:<9} {s['shape']}")
        cov = sorted(s["cov"])
        report[ch] = dict(n=s["n"], eq=s["eq"], maxd=s["maxd"],
                          worst=s["worst"], dtypes=s["dt"],
                          ref_dtype=s["ref_dt"], cand_dtype=s["cand_dt"],
                          shape=s["shape"],
                          chunks=sorted({c[0] for c in cov}),
                          forwards=sorted({c[1] for c in cov}),
                          n_blocks=len({c[2] for c in cov}),
                          compile_eq=NEGSTAT[ch]["eq"],
                          compile_n=NEGSTAT[ch]["n"],
                          compile_max=NEGSTAT[ch]["maxd"])

    print(f"\n[r3] ===== verdict (torch.equal, zero tolerance) =====")
    print(f"   {'chain':<7} {'addcmul':<11} {'compile(neg ctl)':<20} "
          f"chunks   fwds        blocks")
    for ch in CHAINS:
        r = report.get(ch)
        if not r:
            continue
        v = "EXACT" if r["eq"] == r["n"] else "NOT EXACT"
        cv = (f"{r['compile_eq']}/{r['compile_n']} "
              f"max {r['compile_max']:.2e}") if r["compile_n"] else "n/a"
        print(f"   {ch:<7} {v:<11} {cv:<20} "
              f"{str(r['chunks']):<8} {str(r['forwards']):<11} {r['n_blocks']}")

    json.dump(report, open(f"{args.out_dir}/p2d1a2r3.json", "w"),
              indent=1, default=str)
    print(f"\n[r3] wrote {args.out_dir}/p2d1a2r3.json")


if __name__ == "__main__":
    main()
