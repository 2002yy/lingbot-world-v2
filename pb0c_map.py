#!/usr/bin/env python
"""P2b-0c: the full reclamation map -- FP8 weight-only overhead per Linear family.

P2b-0b found that ffn.2 costs 2.939 ms in FP8 against 1.607 ms as a dequantised
bf16 Linear on the same captured input, with bit-identical output. That is +160
ms/chunk (10.2%) for one family.

Before proposing anything, price every quantized family the same way, on real
captured inputs. The point is the TRADE: each family costs some VRAM (fp8 vs
bf16 storage) and buys back some latency. A family is worth switching when the
ms returned per MiB spent is high.

Fidelity: inputs are captured from a real forward, so shape, dtype, device,
stride, contiguity and autocast state are the production ones by construction.
Outputs are compared bit-wise against the shipping module, since both paths use
the same dequantised weight and must therefore agree exactly.
"""
import argparse
import gc
import hashlib
import json
import os
import shutil
import statistics

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
CS_DEFAULT = 3


def bench(fn, n=40, warm=6):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--scene", default="04")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--area", default="512x320")
    ap.add_argument("--local_attn_size", type=int, default=6)
    ap.add_argument("--blocks", type=int, default=1,
                    help="blocks to sample; overhead is per-block and repeats")
    ap.add_argument("--out_dir", default="output/p2b0c")
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"

    W, H = (int(x) for x in args.area.lower().split("x"))
    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    scene, sd, CS = args.scene, args.seed, args.chunk_size

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_attn_size,
        sink_size=1, infer_mode="causal_fast", assets_dir=args.assets_dir)
    dev, pdt, dtype = pipe.device, pipe.param_dtype, pipe.pipe_dtype

    CAP = {}

    def mk(name):
        def hook(mod, inp, out):
            if name not in CAP:
                CAP[name] = (mod, inp[0].detach())
        return hook

    handles = []
    for i, blk in enumerate(pipe.model.blocks[:args.blocks]):
        for nm, m in (("sa.q", blk.self_attn.q), ("sa.k", blk.self_attn.k),
                      ("sa.v", blk.self_attn.v), ("sa.o", blk.self_attn.o),
                      ("ffn.0", blk.ffn[0]), ("ffn.2", blk.ffn[2]),
                      ("cam.in1", blk.cam_injector_layer1),
                      ("cam.in2", blk.cam_injector_layer2),
                      ("cam.scale", blk.cam_scale_layer),
                      ("cam.shift", blk.cam_shift_layer)):
            if isinstance(m, torch.nn.Linear):
                handles.append(m.register_forward_hook(mk(f"b{i}.{nm}")))
        ca = blk.cross_attn
        for nm in ("q", "k", "v", "o"):
            m = getattr(ca, nm, None)
            if isinstance(m, torch.nn.Linear):
                handles.append(m.register_forward_hook(mk(f"b{i}.ca.{nm}")))

    key = hashlib.sha256(PROMPT.encode()).hexdigest()
    ctx = pipe.text_encoder([PROMPT], torch.device("cpu"))
    pipe._t5_cache[key] = [t.to(pipe.device) for t in ctx]
    pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache()

    vae_stride, patch_sz = pipe.vae_stride, pipe.patch_size
    ma = pipe.model.config
    frames_n = (CS - 1) * 4 + 1
    frames_n = ((frames_n - 1) // 4) * 4 + 1
    lat_f = (frames_n - 1) // 4 + 1
    lat_f = int(lat_f - (lat_f % CS))

    d = f"examples/p2b0_{scene}"
    os.makedirs(d, exist_ok=True)
    for f in ("intrinsics.npy", "image.jpg"):
        shutil.copy(f"examples/{scene}/{f}", f"{d}/{f}")
    img_pil = Image.open(f"{d}/image.jpg").convert("RGB")
    th = int(np.sqrt(W * H * (480 / 832)) // 8 * 8)
    tw = int(np.sqrt(W * H / (480 / 832)) // 8 * 8)
    img = (torch.nn.functional.interpolate(
        torch.from_numpy(np.array(img_pil)).permute(2, 0, 1)[None].float(),
        size=(th, tw), mode='bicubic').squeeze(0) / 255.0 - 0.5) / 0.5
    hh, ww = img.shape[1:]
    lat_h, lat_w = hh // vae_stride[1], ww // vae_stride[2]
    fsl = (lat_h * lat_w) // (patch_sz[1] * patch_sz[2])
    max_seq_len = CS * fsl
    kv_size = fsl * args.local_attn_size
    pipe.prewarm(img_pil, max_area=W * H, frame_num=frames_n, chunk_size=CS)
    mf.bump_cam_epoch()
    Ks = get_Ks_transformed(
        torch.from_numpy(np.load(f"{d}/intrinsics.npy")).float(),
        480, 832, hh, ww, hh, ww)[0].to(dev)
    y = pipe.vae.encode([torch.concat([
        img[None].transpose(0, 1).to(dev),
        torch.zeros(3, frames_n - 1, hh, ww, device=dev)], dim=1)])[0]
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

    gg = torch.Generator(device=dev); gg.manual_seed(sd)
    cur = torch.randn(16, CS, lat_h, lat_w, generator=gg, device=dev)
    pp = get_plucker_embeddings(rel_all[0:CS], Ks[None], hh, ww)
    pp = rearrange(pp, 'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                   c1=int(hh // lat_h), c2=int(ww // lat_w))[None]
    plk = rearrange(pp, 'b (f h w) c -> b c f h w', f=CS,
                    h=lat_h, w=lat_w).to(pdt)
    with torch.amp.autocast("cuda", dtype=pdt), torch.no_grad():
        pipe.model(x=[cur.to(dev)], t=torch.stack([timesteps[0]]).to(dev),
                   context=[pipe._t5_cache[key][0]], seq_len=max_seq_len,
                   y=[y.split(CS, dim=1)[0]],
                   dit_cond_dict={"c2ws_plucker_emb": plk.chunk(1, dim=0)},
                   kv_cache=self_kv, crossattn_cache=cross_kv,
                   current_start=0, max_attention_size=kv_size,
                   frame_seqlen=fsl, cross_attn_first_call=True)
    for h in handles:
        h.remove()
    torch.cuda.synchronize()

    print("=" * 92)
    print("  P2b-0c: FP8 weight-only overhead per Linear family (real inputs)")
    print("=" * 92)
    print(f"  captured {len(CAP)} Linear callsites over "
          f"{args.blocks} block(s)\n")

    N_PER_CHUNK = {"sa.q": 120, "sa.k": 120, "sa.v": 120, "sa.o": 120,
                   "ffn.0": 120, "ffn.2": 120,
                   "cam.in1": 30, "cam.in2": 30, "cam.scale": 30,
                   "cam.shift": 30,
                   "ca.q": 120, "ca.k": 120, "ca.v": 120, "ca.o": 120}

    rows = []
    from torchao.quantization import Float8WeightOnlyConfig, quantize_
    for name in sorted(CAP):
        mod, xin = CAP[name]
        if not hasattr(mod, "weight"):
            continue
        w = mod.weight
        is_fp8 = type(w).__name__ == "Float8Tensor"
        if not is_fp8:
            continue
        try:
            wb = w.dequantize().to(torch.bfloat16)
        except Exception as e:
            print(f"  {name}: dequantize failed: {e}")
            continue
        K_in = int(mod.in_features)
        N_out = int(mod.out_features)
        # Production dtype varies by callsite (some Linears are fed float32).
        # The FP8 float8_tensor path requires mat1 dtype == mat2 dtype, so keep
        # whichever dtype the module actually accepts for the fp8 leg, and use
        # the same for the bf16 leg so the comparison is like for like.
        xin_use = xin
        try:
            mod(xin)
        except RuntimeError:
            xin_use = xin.to(torch.bfloat16)
        xb = xin_use.to(torch.bfloat16).contiguous()
        xf = xb.reshape(-1, K_in)

        plain = torch.nn.Linear(K_in, N_out, bias=mod.bias is not None).to(dev)
        with torch.no_grad():
            plain.weight.copy_(wb)
            if mod.bias is not None:
                plain.bias.copy_(mod.bias.to(torch.bfloat16))
        plain = plain.to(torch.bfloat16)

        t_fp8 = bench(lambda: mod(xin_use))
        t_bf = bench(lambda: plain(xf))

        a = mod(xin_use)
        b = plain(xf)
        dq = (a.float().reshape(-1).unsqueeze(0) -
              b.float().reshape(-1).unsqueeze(0)).abs().max().item()
        beq = torch.equal(a.reshape(-1), b.reshape(-1))

        fam = name.split(".", 1)[1]
        nch = N_PER_CHUNK.get(fam, 120)
        dms = (t_fp8 - t_bf) * nch
        P = K_in * N_out
        vram_mb = P * (2 - 1) / 2**20          # fp8(1B) -> bf16(2B)
        rows.append(dict(name=name, fam=fam, M=int(xf.shape[0]), K=K_in,
                         N=N_out, ms_fp8=t_fp8, ms_bf16=t_bf,
                         delta_ms=t_fp8 - t_bf, per_chunk=nch, save_ms=dms,
                         params=P, vram_mb=vram_mb, bit_equal=beq,
                         max_diff=float(dq)))

    print(f"  {'family':<12} {'M':>5} {'K':>6} {'N':>6} {'fp8 ms':>8} "
          f"{'bf16 ms':>8} {'delta':>8} {'x/chunk':>7} {'save/chunk':>10} "
          f"{'paramM':>7} {'+MB':>6} {'bit-eq':>6}")
    print("  " + "-" * 88)
    tot_ms = 0.0
    tot_mb = 0.0
    for r in sorted(rows, key=lambda r: -r["save_ms"]):
        print(f"  {r['fam']:<12} {r['M']:>5} {r['K']:>6} {r['N']:>6} "
              f"{r['ms_fp8']:>8.4f} {r['ms_bf16']:>8.4f} {r['delta_ms']:>8.4f} "
              f"{r['per_chunk']:>7} {r['save_ms']:>10.1f} "
              f"{r['params']/1e6:>7.2f} {r['vram_mb']:>6.1f} "
              f"{str(r['bit_equal']):>6}")
        tot_ms += r["save_ms"]
        tot_mb += r["vram_mb"]

    print("  " + "-" * 88)
    blk = args.blocks
    # NOTE: N_PER_CHUNK already holds the TOTAL per chunk across all 30 blocks
    # (120 = 30 blocks x 4 forwards), so tot_ms IS the whole-model per-chunk
    # figure. An earlier version multiplied by 30 again and reported 12395
    # ms/chunk, i.e. 795% of the chunk -- an arithmetic impossibility that
    # should have been caught immediately by that sanity check alone.
    print(f"\n  TOTAL per chunk (all 30 blocks): {tot_ms:8.1f} ms")
    print(f"  VRAM cost for that switch      : {tot_mb:8.1f} MB "
          f"({tot_mb/1024:0.2f} GiB)")
    print(f"  as a fraction of 1560 ms       : {tot_ms/1560*100:8.1f}%")
    print(f"  sanity: must be < 1560 ms      : "
          f"{'OK' if tot_ms < 1560 else 'IMPOSSIBLE -- check the call counts'}")
    print(f"\n  all bit-equal: {all(r['bit_equal'] for r in rows)}")
    print(f"  (both paths use the same dequantised weight, so equality is expected)")

    with open(os.path.join(args.out_dir, "p2b0c.json"), "w") as f:
        json.dump(dict(rows=rows, total_per_chunk_ms=tot_ms,
                       total_mib=tot_mb, sampled_blocks=blk), f, indent=2)
    print(f"\n[P2b-0c] wrote {args.out_dir}/p2b0c.json")


if __name__ == "__main__":
    main()
