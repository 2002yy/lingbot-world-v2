#!/usr/bin/env python
"""P2d-1a2: elementwise exactness MATRIX over all candidate chains.

WHY EXHAUST THE CHAINS BEFORE FUSING ANY
----------------------------------------
One chain passing does not imply the others will. The chains differ in operand
dtype, broadcast shape, the ORDER of the add/mul, where the cast sits, whether the
reference materialises `y * gate` first, and whether addcmul dispatches to the
same kernel for that shape/dtype. So all of them get measured, and each chain is
judged independently and with zero tolerance: any real sample that is not
torch.equal blocks that chain from repro.

CHAINS (all taken from the actual block)
    mod1   norm1(x).float() * (1 + e1) + e0
    mod2   norm2(x).float() * (1 + e4) + e3
    resA   x + y_attn * e2
    resB   x + y_ffn  * e5
    cam    (1 + cam_scale) * x + cam_shift

`1 + scale` is kept OUT of the fusion on purpose: it is computed exactly as the
model computes it, so the probe varies only the mul+add fusion and not a second
thing.

CANDIDATES
    addcmul  torch.addcmul(a, b, c)         -- explicit ATen op; in P2d-1a this
                                               was bit-exact on the modulation
                                               chain while torch.compile was not
    compile  torch.compile of the same form -- measured here per chain so the
                                               matrix says which chains, not just
                                               which mechanism

JUDGEMENT: torch.equal is the only gate. max|diff| must be 0. A 1-ulp difference
is a FAIL, because a recurrent rollout is demonstrably intolerant of tiny
perturbations (the P2a lesson).

CORPUS: captured from a real repro run with hooks, tagged so the report can show
which forwards and chunks were covered (early/mid/late denoise, KV update, chunk
0 and non-zero, several blocks). Tensors are held on CPU to keep VRAM flat.
"""
import argparse
import collections
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
import wan.modules.model_fast as mf
from wan.configs import WAN_CONFIGS
from wan.utils.cam_utils import get_Ks_transformed, interpolate_camera_poses, \
    compute_relative_poses, get_plucker_embeddings
from einops import rearrange

PROMPT = "A first-person view of a natural landscape with smooth camera motion."

SAMPLES = collections.defaultdict(list)   # chain -> list of dict(a,b,c,meta)
CAP = {"on": True, "per_chain": 40, "fwd": 0, "cid": 0, "bi": 0}
HOLD = {}


def note(chain, a, b, c, meta):
    if not CAP["on"] or len(SAMPLES[chain]) >= CAP["per_chain"]:
        return
    SAMPLES[chain].append(dict(a=a.detach().to("cpu"), b=b.detach().to("cpu"),
                               c=c.detach().to("cpu"), meta=meta))


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
    ap.add_argument("--out_dir", default="output/p2d1a2")
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
    print(f"[m] layers={pipe.model.config.num_layers} dtype={pdt} "
          f"mode={args.mode}", flush=True)

    # ---------------- hooks ----------------
    def simple_hook(tag):
        def hook(mod, args_in, out):
            HOLD[tag] = out.detach() if torch.is_tensor(out) else out[0].detach()
        return hook

    def pre_hook(tag):
        def hook(mod, args_in):
            if args_in and torch.is_tensor(args_in[0]):
                HOLD[tag] = args_in[0].detach()
        return hook

    for i, blk in enumerate(pipe.model.blocks):
        blk.norm1.register_forward_hook(simple_hook(f"{i}:n1"))
        blk.norm2.register_forward_hook(simple_hook(f"{i}:n2"))
        blk.self_attn.register_forward_hook(simple_hook(f"{i}:yattn"))
        blk.cross_attn.register_forward_hook(simple_hook(f"{i}:ycross"))
        blk.ffn.register_forward_hook(simple_hook(f"{i}:yffn"))
        blk.register_forward_pre_hook(pre_hook(f"{i}:xin"))

    orig_cb = mf.CausalWanAttentionBlock.forward

    # cam_scale / cam_shift are captured with hooks on the projection Linears;
    # the per-block forward patch below reads them out of HOLD.
    for i, blk in enumerate(pipe.model.blocks):
        def mk(i_):
            def h1(mod, args_in, out):
                HOLD[f"{i_}:camscale"] = out.detach()

            def h2(mod, args_in, out):
                HOLD[f"{i_}:camshift"] = out.detach()
            return h1, h2
        h1, h2 = mk(i)
        blk.cam_scale_layer.register_forward_hook(h1)
        blk.cam_shift_layer.register_forward_hook(h2)

    def make_patch2(bi, blk):
        # NOTE: this is assigned as `blk.forward = patched`, i.e. an INSTANCE
        # attribute, so it is NOT bound -- a leading `self` parameter would
        # swallow the real `x` argument. Capture the block by closure instead.
        def patched(x, e, *a, **kw):
            out = orig_cb(blk, x, e, *a, **kw)
            if CAP["on"] and len(SAMPLES["mod1"]) < CAP["per_chain"]:
                cs = HOLD.get(f"{bi}:camscale")
                ct = HOLD.get(f"{bi}:camshift")
                if cs is not None and ct is not None:
                    HOLD[f"{bi}:cam"] = (cs, ct)
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    ec = (blk.modulation.unsqueeze(0) + e).chunk(6, dim=2)
                meta = dict(chunk=CAP["cid"], fwd=CAP["fwd"], block=bi)
                xin = HOLD.get(f"{bi}:xin")
                yattn = HOLD.get(f"{bi}:yattn")
                ycross = HOLD.get(f"{bi}:ycross")
                yffn = HOLD.get(f"{bi}:yffn")
                n1 = HOLD.get(f"{bi}:n1")
                n2 = HOLD.get(f"{bi}:n2")
                if n1 is not None:
                    note("mod1", n1, 1 + ec[1].squeeze(2), ec[0].squeeze(2), meta)
                if n2 is not None:
                    note("mod2", n2, 1 + ec[4].squeeze(2), ec[3].squeeze(2), meta)
                if xin is not None and yattn is not None:
                    xA = xin + yattn * ec[2].squeeze(2)
                    note("resA", xin, yattn, ec[2].squeeze(2), meta)
                    if cs is not None:
                        xCam = (1.0 + cs) * xA + ct
                        note("cam", xA, 1.0 + cs, ct, meta)
                        if ycross is not None:
                            xCross = xCam + ycross
                            if yffn is not None:
                                note("resB", xCross, yffn, ec[5].squeeze(2), meta)
            return out
        return patched

    for i, blk in enumerate(pipe.model.blocks):
        blk.forward = make_patch2(i, blk)

    # ---------------- run a real rollout ----------------
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

    d = f"examples/p2d1a2_{scene}"
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
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=CS)
    from wan.modules.model_fast import bump_cam_epoch
    bump_cam_epoch()
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
    print(f"[m] captured: " + ", ".join(
        f"{k}={len(v)}" for k, v in sorted(SAMPLES.items())), flush=True)

    # ---------------- exactness ----------------
    def ref_form(a, b, c):
        return a + b * c

    chains = sorted(SAMPLES.keys())
    fused = {}
    for ch in chains:
        try:
            fused[ch] = torch.compile(ref_form, mode=args.mode, fullgraph=True)
        except Exception as e:
            fused[ch] = None
            print(f"[m] compile unavailable for {ch}: {e}")

    report = {}
    print(f"\n[m] ===== elementwise exactness matrix =====")
    print(f"   {'chain':<7} {'cand':<9} {'equal':>10} {'max|diff|':>12} "
          f"{'dtype':<9} {'shape (a)':<20}")
    for ch in chains:
        samples = SAMPLES[ch]
        rep = report.setdefault(ch, {})
        for cname in ("addcmul", "compile"):
            eq = 0
            maxd = 0.0
            worst = None
            for idx, s in enumerate(samples):
                a = s["a"].to(dev); b = s["b"].to(dev); c = s["c"].to(dev)
                with torch.no_grad():
                    r = ref_form(a, b, c)
                    try:
                        if cname == "addcmul":
                            cand = torch.addcmul(a, b, c)
                        else:
                            fn = fused[ch]
                            if fn is None:
                                continue
                            cand = fn(a, b, c)
                    except Exception as ex:
                        worst = f"{type(ex).__name__}: {str(ex)[:60]}"
                        continue
                d = (r.float() - cand.float()).abs().max().item()
                if torch.equal(r, cand):
                    eq += 1
                if d > maxd:
                    maxd = d
                    worst = (f"sample {idx} {s['meta']} max|d|={d:.3e}")
            n = len(samples)
            print(f"   {ch:<7} {cname:<9} {eq:>5}/{n:<4} {maxd:>12.3e} "
                  f"{str(samples[0]['a'].dtype).replace('torch.',''):<9} "
                  f"{str(tuple(samples[0]['a'].shape)):<20}")
            rep[cname] = dict(equal=eq, n=n, max_abs=maxd, worst=worst)
        # coverage
        met = [s["meta"] for s in samples]
        rep["coverage"] = dict(
            chunks=sorted({m["chunk"] for m in met}),
            forwards=sorted({m["fwd"] for m in met}),
            blocks=sorted({m["block"] for m in met})[:6],
            n_blocks=len({m["block"] for m in met}))

    def verdict(rep):
        out = {}
        for cname in ("addcmul", "compile"):
            r = rep.get(cname)
            if not r or r["n"] == 0:
                out[cname] = "no data"
            elif r["equal"] == r["n"]:
                out[cname] = "EXACT"
            elif r["max_abs"] > 0:
                out[cname] = "NOT EXACT"
            else:
                out[cname] = "?"
        return out

    print(f"\n[m] verdict per chain (torch.equal, zero tolerance):")
    for ch in chains:
        v = verdict(report[ch])
        cov = report[ch]["coverage"]
        print(f"   {ch:<7} addcmul={v['addcmul']:<10} compile={v['compile']:<10} "
              f"| chunks={cov['chunks']} forwards={cov['forwards']} "
              f"blocks={cov['n_blocks']}")
    json.dump({k: {kk: vv for kk, vv in v.items()}
               for k, v in report.items()},
              open(f"{args.out_dir}/p2d1a2.json", "w"), indent=1, default=str)
    print(f"\n[m] wrote {args.out_dir}/p2d1a2.json")


if __name__ == "__main__":
    main()
