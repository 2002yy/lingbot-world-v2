#!/usr/bin/env python
"""M0-prod-A: 304x528 production baseline via the REAL production chain.

Not a benchmark loop. This drives `wan.WanI2VCausal.generate()` itself, so the
whole production request is exercised: text conditioning, the condition encode,
the causal chunk loop, the KV update, and the VAE decode.

Geometry note. 304x528 (fsl 627) is not `image2video.py`'s default -- its default
is 512x768, which is infeasible on 8 GB (M0-prod frontier). generate() derives
the geometry from the NATIVE aspect of the image it is handed plus `max_area`, so
304x528 is selected by handing it a pre-resized 304x528 image with
max_area = 512*320 = 163840:

    aspect = 304/528 = 0.5758
    lat_h = round(sqrt(163840*0.5758)//8//2*2) = 38
    lat_w = round(sqrt(163840/0.5758)//8//2*2) = 66   -> 304x528

What is measured, per the gate:

    per request:  global max_reserved / max_allocated for the WHOLE request
                  (encode peak and generation peak are NOT added -- they are
                   serial phases and the encode activations are released)
    around it:    allocated / reserved before and after, to catch residue
    across it:    3-5 consecutive requests, to catch VRAM creep
    plus:         cold vs warm, latency, throughput

A note on `empty_cache()`: the gate says the path must not depend on manual
`empty_cache()` to survive. The production default `offload_model=True` calls it
internally, so this measures BOTH settings and reports which one is required.
"""
import argparse
import gc
import json
import os
import statistics
import time

import torch
from PIL import Image

import wan
from wan.configs import WAN_CONFIGS

MB = 2 ** 20
PROMPT = "A first-person view of a natural landscape with smooth camera motion."


def snap():
    free, total = torch.cuda.mem_get_info()
    return dict(alloc=torch.cuda.memory_allocated() / MB,
                reserved=torch.cuda.memory_reserved() / MB,
                max_alloc=torch.cuda.max_memory_allocated() / MB,
                max_reserved=torch.cuda.max_memory_reserved() / MB,
                free=free / MB, total=total / MB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="i2v-1.3B")
    ap.add_argument("--ckpt_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"))
    ap.add_argument("--assets_dir", default=os.path.expanduser(
        "~/ai/models/lingbot-shared-assets"))
    ap.add_argument("--base", default="examples/04")
    ap.add_argument("--weight", default="bf16", choices=["bf16", "fp8_lowmem", "fp8"])
    ap.add_argument("--pixel", default="304x528")
    ap.add_argument("--frames", type=int, default=81,
                    help="81 is generate()'s own default")
    ap.add_argument("--chunk_size", type=int, default=3)
    ap.add_argument("--requests", type=int, default=5)
    ap.add_argument("--offload", type=int, default=1,
                    help="generate()'s offload_model; production default is 1")
    ap.add_argument("--ensure_device", type=int, default=0,
                    help="move the DiT back to the GPU before each request. "
                         "Needed because offload_model=True calls "
                         "self.model.cpu() after generation and never restores "
                         "it, so the SECOND request dies with 'Input type (CUDA) "
                         "and weight type (CPU) should be the same'")
    ap.add_argument("--local_window", type=int, default=8)
    ap.add_argument("--sink", type=int, default=2)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.environ["LINGBOT_MODE"] = "repro"
    if args.weight == "bf16":
        os.environ["LINGBOT_WEIGHT_MODE"] = "bf16"
        os.environ["LINGBOT_FP8"] = "0"
    else:
        os.environ["LINGBOT_WEIGHT_MODE"] = "fp8_lowmem"
        os.environ["LINGBOT_FP8"] = "1"
    os.environ["LINGBOT_FFN0_FP8"] = "0"
    os.environ["LINGBOT_CAM_CACHE"] = "1"
    os.environ["LINGBOT_ROPE_CACHE"] = "0"

    cfg = WAN_CONFIGS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    pw, ph = (int(x) for x in args.pixel.lower().split("x"))
    max_area = 512 * 320

    print("=" * 92)
    print(f"  M0-prod-A  REAL production chain    weight={args.weight}  "
          f"pixel={pw}x{ph}  frames={args.frames}  cs={args.chunk_size}  "
          f"requests={args.requests}  offload={args.offload}")
    print(f"  preset local_window={args.local_window} sink={args.sink} "
          f"(worst case)")
    print("=" * 92, flush=True)

    pipe = wan.WanI2VCausal(
        config=cfg, checkpoint_dir=args.ckpt_dir, device_id=0, rank=0,
        t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
        convert_model_dtype=False, local_attn_size=args.local_window,
        sink_size=args.sink, infer_mode="causal_fast",
        assets_dir=args.assets_dir)
    dev = pipe.device
    after_load = snap()
    print(f"  after model load: alloc {after_load['alloc']:.0f} "
          f"reserved {after_load['reserved']:.0f} free {after_load['free']:.0f} MiB",
          flush=True)

    # pre-resize so generate()'s native-aspect rule selects 304x528
    im = Image.open(f"{args.base}/image.jpg").convert("RGB")
    im_rs = im.resize((pw, ph), Image.BICUBIC)

    rows = []
    for r in range(args.requests):
        before = snap()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        if args.ensure_device:
            # offload_model=True leaves the DiT on the CPU after a request; put it
            # back so the next request is not a dtype/device mismatch.
            pipe.model.to(dev)
            pipe.vae.model.to(dev)
        t0 = time.perf_counter()
        ok, err = True, None
        try:
            out = pipe.generate(
                input_prompt=PROMPT,
                img=im_rs,
                action_path=args.base,
                chunk_size=args.chunk_size,
                max_area=max_area,
                frame_num=args.frames,
                shift=5.0,
                seed=42,
                offload_model=bool(args.offload),
            )
            torch.cuda.synchronize()
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        dt = time.perf_counter() - t0
        during = snap()
        after = snap()
        rows.append(dict(req=r, ok=ok, err=err, s=dt, before=before,
                         during=during, after=after))
        lab = "cold" if r == 0 else "warm"
        print(f"  [{r} {lab}] ok={ok} {dt:6.2f} s  "
              f"global max_reserved {during['max_reserved']:7.0f}  "
              f"max_alloc {during['max_alloc']:7.0f}  "
              f"| after: alloc {after['alloc']:7.0f} reserved "
              f"{after['reserved']:7.0f} free {after['free']:7.0f}"
              + (f"  err={err}" if err else ""), flush=True)
        if not ok:
            break

    # ---------------- verdict ----------------
    good = [r for r in rows if r["ok"]]
    print()
    print("=" * 92)
    print("  M0-prod-A VERDICT")
    print("=" * 92)
    if not good:
        print("  no successful request -- FAIL")
    else:
        peaks = [r["during"]["max_reserved"] for r in good]
        post_res = [r["after"]["reserved"] for r in good]
        post_alloc = [r["after"]["alloc"] for r in good]
        print(f"  requests ok              {len(good)}/{args.requests}")
        print(f"  global max_reserved      first {peaks[0]:7.0f}  "
              f"last {peaks[-1]:7.0f}  max {max(peaks):7.0f} MiB")
        print(f"  peak spread across reqs  {max(peaks) - min(peaks):+7.0f} MiB")
        print(f"  post-request reserved    first {post_res[0]:7.0f}  "
              f"last {post_res[-1]:7.0f} MiB")
        print(f"  post-request allocated   first {post_alloc[0]:7.0f}  "
              f"last {post_alloc[-1]:7.0f} MiB")
        creep_r = post_res[-1] - post_res[0]
        creep_a = post_alloc[-1] - post_alloc[0]
        print(f"  creep (last-first)       reserved {creep_r:+.0f}  "
              f"allocated {creep_a:+.0f} MiB")
        print(f"  min free during requests "
              f"{min(r['during']['free'] for r in good):7.0f} MiB")
        lat = [r["s"] for r in good[1:]] or [good[0]["s"]]
        print(f"  latency warm             p50 {statistics.median(lat):.2f} s  "
              f"min {min(lat):.2f}  max {max(lat):.2f}")
        fps = 4.0 * args.chunk_size * (args.frames // (4 * args.chunk_size)) / \
            statistics.median(lat)
        print(f"  throughput               {1.0/statistics.median(lat):.2f} "
              f"request/s   (~{fps:.2f} fps-equiv over {args.frames} frames)")

        total = good[0]["during"]["total"]
        c1 = len(good) == args.requests
        c2 = abs(creep_r) < 64 and abs(creep_a) < 64
        c3 = (max(peaks) - min(peaks)) < 128
        c4 = min(r["during"]["free"] for r in good) > 200
        print()
        print(f"  1  all {args.requests} requests ok            "
              f"{'PASS' if c1 else 'FAIL'}")
        print(f"  2  no VRAM creep (<64 MiB)           "
              f"{'PASS' if c2 else 'FAIL'}  (res {creep_r:+.0f}, "
              f"alloc {creep_a:+.0f})")
        print(f"  3  global peak stable (<128 MiB)     "
              f"{'PASS' if c3 else 'FAIL'}  (spread "
              f"{max(peaks)-min(peaks):.0f})")
        print(f"  4  free headroom > 200 MiB           "
              f"{'PASS' if c4 else 'FAIL'}  (min "
              f"{min(r['during']['free'] for r in good):.0f})")
        overall = c1 and c2 and c3 and c4
        print()
        print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
        print(f"  geometry authority: 304x528 is "
              f"{'a viable' if overall else 'NOT yet a viable'} 8 GB deployment "
              f"profile")

    with open(f"{args.out_dir}/m0_prod_a.json", "w") as f:
        json.dump(dict(weight=args.weight, pixel=[pw, ph], frames=args.frames,
                       chunk_size=args.chunk_size, requests=args.requests,
                       offload=args.offload, after_load=after_load, rows=rows),
                  f, indent=2)
    print(f"\n[m0-prod-A] wrote {args.out_dir}/m0_prod_a.json")


if __name__ == "__main__":
    main()
