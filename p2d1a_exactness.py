#!/usr/bin/env python
"""P2d-1a: can "eliminate elementwise kernel boundaries" coexist with bit-exactness?

THE QUESTION
------------
P2d's whole selling point is being repro-eligible (bit-exact), unlike P2a. The
non-attention residual is ~145 ms of diffuse elementwise work. Before writing any
fused kernel we need to know whether fusing elementwise ops AT ALL preserves the
bits on this path. If even a mild fusion changes the last bits, P2d is just
another fast-only gamble and should be closed the way P2a was.

SCOPE -- POST-NORM ONLY
-----------------------
The probe deliberately does NOT include the LayerNorm reduction. Fusing a
reduction would confound two questions (does changing the reduction order matter,
and does mul/add fusion matter) and the reduction is the one that is almost
guaranteed to change. The reference keeps the existing norm implementation and
only the chain after it is re-expressed:

    reference :  y = norm(x); y = y.float(); y = y * (1 + scale); y = y + shift
    candidate :  y = norm(x); y = fused_modulation(norm(x), scale, shift)

STOP CRITERION -- INSPECT BYTES, NOT PSNR
-----------------------------------------
    torch.equal        must be True
    max_abs_diff       must be 0
    hash               must match
No tolerance. A one-ulp difference is a FAIL, because a recurrent rollout is
demonstrably intolerant of tiny perturbations (that is the P2a lesson).

REAL CORPUS, NOT torch.randn
----------------------------
Inputs are captured from an actual repro run: the layer-norm outputs and the
modulation scale/shift that the block actually feeds into this chain, taken
across several chunks, several blocks, all four forward kinds (3 denoise steps +
KV update). The question we want answered is "is this exact for the data LingBot
actually produces", not "is it exact for synthetic data that happened to pass".

CANDIDATES
    compile  torch.compile of the chain (Inductor fusion, mode from --mode)
    addcmul  torch.addcmul(shift, n_float, 1 + scale) -- an ATen op that is
             fused by construction and typically contracts to FMA, i.e. the
             adversarial case

Run:
  LINGBOT_FP8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python p2d1a_exactness.py --scene 04 --chunks 2
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
import time

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

CORPUS = []          # list of (norm_out, scale, shift, tag)
CAPTURE = {"on": True}
MAX_SAMPLES = 48   # keep VRAM flat; 48 already
                   # covers several blocks x all 4 forward kinds


def ref_chain(n, scale, shift):
    y = n.float()
    y = y * (1 + scale)
    y = y + shift
    return y


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
    ap.add_argument("--out_dir", default="output/p2d1a")
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
    print(f"[p2d1a] layers={pipe.model.config.num_layers} "
          f"dtype={pdt} compile_mode={args.mode}", flush=True)

    # ---------- capture the REAL inputs to the post-norm chain ----------
    # norm1 / norm2 outputs
    norm_out = {}

    def mk_norm_hook(name):
        def hook(mod, args_in, out):
            if CAPTURE["on"] and len(CORPUS) < MAX_SAMPLES:
                norm_out[name] = out.detach().to("cpu")
        return hook

    for i, blk in enumerate(pipe.model.blocks):
        blk.norm1.register_forward_hook(mk_norm_hook(f"{i}.norm1"))
        blk.norm2.register_forward_hook(mk_norm_hook(f"{i}.norm2"))

    # modulation chunks: replicate the block's own computation and stash them
    orig_block_forward = mf.CausalWanAttentionBlock.forward

    def patched_forward(self, x, e, *a, **kw):
        if CAPTURE["on"] and len(CORPUS) < MAX_SAMPLES:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                ec = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
            # norm1 path uses shift=e[0], scale=e[1]; norm2 path e[3], e[4]
            for nm, si, sc in (("norm1", 0, 1), ("norm2", 3, 4)):
                key = None
                for k in norm_out:
                    if k.endswith(nm):
                        key = k
                if key is not None:
                    CORPUS.append((norm_out.pop(key),
                                   ec[sc].squeeze(2).detach().to("cpu"),
                                   ec[si].squeeze(2).detach().to("cpu"),
                                   nm))
        return orig_block_forward(self, x, e, *a, **kw)

    mf.CausalWanAttentionBlock.forward = patched_forward

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

    d = f"examples/p2d1a_{scene}"
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
        with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
            pipe.model(x=[x0], t=torch.stack([timesteps[-1] * 0.0]).to(dev),
                       cross_attn_first_call=False, **kw)
    CAPTURE["on"] = False
    mf.CausalWanAttentionBlock.forward = orig_block_forward
    print(f"[p2d1a] captured {len(CORPUS)} real (norm, scale, shift) samples "
          f"from {n_test} chunks x 30 blocks x 4 forwards", flush=True)
    if not CORPUS:
        print("[p2d1a] FATAL: no samples captured; hooks did not fire")
        return

    # ---------- exactness ----------
    fused = torch.compile(ref_chain, mode=args.mode, fullgraph=True)

    def addcmul_chain(n, scale, shift):
        return torch.addcmul(shift, n.float(), 1 + scale)

    stats = collections.defaultdict(lambda: dict(n=0, eq=0, maxd=0.0,
                                                 worst=None, ulp_exact=0))
    for idx, (n_c, sc_c, sh_c, tag) in enumerate(CORPUS):
        n = n_c.to(dev); sc = sc_c.to(dev); sh = sh_c.to(dev)
        with torch.no_grad():
            r = ref_chain(n, sc, sh)
        for name, fn in (("compile", fused), ("addcmul", addcmul_chain)):
            with torch.no_grad():
                try:
                    c = fn(n, sc, sh)
                except Exception as ex:
                    stats[name]["worst"] = f"{type(ex).__name__}: {str(ex)[:80]}"
                    stats[name]["n"] += 1
                    continue
            s = stats[name]
            s["n"] += 1
            same = torch.equal(r, c)
            d = (r.float() - c.float()).abs().max().item()
            if same:
                s["eq"] += 1
            if d == 0.0:
                s["ulp_exact"] += 1
            if d > s["maxd"]:
                s["maxd"] = d
                s["worst"] = f"sample {idx} ({tag}) max|d|={d:.3e}"

    print(f"\n[p2d1a] ===== post-norm modulation chain exactness =====")
    print(f"   corpus: {len(CORPUS)} real samples "
          f"(bf16 norm out -> fp32 chain -> bf16 consumer)")
    print(f"   {'candidate':<12} {'equal':>10} {'n':>6} {'max|diff|':>12}")
    for name, s in stats.items():
        print(f"   {name:<12} {s['eq']:>5}/{s['n']:<4} {s['n']:>6} "
              f"{s['maxd']:>12.3e}")
        if s["worst"]:
            print(f"      worst: {s['worst']}")

    def verdict(s):
        if s["eq"] == s["n"] and s["n"] > 0:
            return "A: FULLY EXACT"
        if s["maxd"] > 0 and s["maxd"] < 1e-3:
            return "B: LSB-LEVEL DIFFERENCE -> P2d-repro FAIL"
        return "C: CLEARLY NOT EXACT -> close modulation fusion for repro"
    print("\n[p2d1a] verdicts:")
    for name, s in stats.items():
        print(f"   {name:<12} {verdict(s)}")

    # a sanity anchor: the reference against itself must be exact
    with torch.no_grad():
        a0 = CORPUS[0][0].to(dev); a1 = CORPUS[0][1].to(dev); a2 = CORPUS[0][2].to(dev)
        r0 = ref_chain(a0, a1, a2)
        r1 = ref_chain(a0, a1, a2)
    print(f"\n[p2d1a] anchor: reference vs reference equal = {torch.equal(r0, r1)}")

    json.dump(dict(corpus_size=len(CORPUS), mode=args.mode,
                   stats={k: dict(n=v["n"], equal=v["eq"], max_abs=v["maxd"],
                                  worst=v["worst"]) for k, v in stats.items()},
                   verdicts={k: verdict(v) for k, v in stats.items()}),
              open(f"{args.out_dir}/p2d1a.json", "w"), indent=1, default=str)
    print(f"\n[p2d1a] wrote {args.out_dir}/p2d1a.json")


if __name__ == "__main__":
    main()
