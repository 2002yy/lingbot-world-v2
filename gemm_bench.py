#!/usr/bin/env python
"""P1-Pre part 2: roofline on the captured REAL shapes, plus an M sweep.

Part 1 captured the shapes from a real forward pass (M is measured, not guessed):

    role                      M      K      N
    ffn.0                    627   1536   8960      <- 47.9% of chunk, the target
    ffn.2                    627   8960   1536
    self_attn.q/k/v/o        627   1536   1536
    cross_attn.k/v           512   1536   1536
    cross_attn.q/o           627   1536   1536
    cam.*                    627   1536   1536

and already showed something important on the 1536x1536 shapes:

    627,1536,1536   bf16 0.1372 ms   FP8-WO 0.2491   FP8-rowwise 0.2650
                    -> FP8 is ~1.9x SLOWER, both variants

i.e. for these sizes the GEMM is too small for FP8 tensor cores to pay off and
the quantise overhead dominates. This script covers the FFN shapes (which were
missed by a role-name matching bug) and sweeps M, because whether a GEMM is
compute-bound depends on M and that decides whether FP8 can help at all.

Configs:
    A  bf16                                 production today
    B  torchao Float8WeightOnlyConfig       what LINGBOT_FP8 applies today
    C  torchao Float8DynamicActivation...   rowwise FP8 compute

Run:
  python gemm_bench.py
"""
import argparse
import json
import math
import os
import statistics
import time

import torch
import torch.nn as nn

SHAPES = [
    ("ffn.0 (up)",      627, 1536, 8960),
    ("ffn.2 (down)",    627, 8960, 1536),
    ("self_attn.qkvo",  627, 1536, 1536),
    ("cross_attn.kv",   512, 1536, 1536),
]
M_SWEEP = [627, 1254, 1881, 3762, 7524, 15048, 30096]


def bench(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def tflops(M, K, N, ms):
    return (2.0 * M * K * N) / (ms * 1e-3) / 1e12


def build(K, N, dev):
    lin = nn.Linear(K, N, bias=False).to(dev, torch.bfloat16)
    with torch.no_grad():
        lin.weight.normal_(0, 0.02)
    return lin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out_dir", default="output/gemm_roofline")
    args = ap.parse_args()

    dev = "cuda"
    print(f"torch {torch.__version__} cap {torch.cuda.get_device_capability()}")
    import torchao
    from torchao.quantization import (
        Float8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig,
        quantize_)
    print(f"torchao {torchao.__version__}")
    print(f"device {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).multi_processor_count} SMs")

    def run_cfg(M, K, N, cfg):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        lin = build(K, N, dev)
        if cfg == "B":
            quantize_(lin, Float8WeightOnlyConfig())
        elif cfg == "C":
            quantize_(lin, Float8DynamicActivationFloat8WeightConfig())
        with torch.no_grad():
            return bench(lambda: lin(x), n=args.n)

    print("\n===== real captured shapes =====")
    print("{:<18} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>8} {:>9} {:>9}".format(
        "shape", "M", "K", "N", "A bf16 ms", "B WO ms", "C row ms",
        "C/A", "A TFLOP/s", "C TFLOP/s"))
    print("-" * 108)
    rows = []
    for name, M, K, N in SHAPES:
        ta = run_cfg(M, K, N, "A")
        tb = run_cfg(M, K, N, "B")
        tc = run_cfg(M, K, N, "C")
        print("{:<18} {:>6} {:>6} {:>6} {:>10.4f} {:>10.4f} {:>10.4f} "
              "{:>7.2f}x {:>9.1f} {:>9.1f}".format(
                  name, M, K, N, ta, tb, tc, ta / tc,
                  tflops(M, K, N, ta), tflops(M, K, N, tc)), flush=True)
        rows.append(dict(name=name, M=M, K=K, N=N, a=ta, b=tb, c=tc,
                         c_over_a=ta / tc, a_tflops=tflops(M, K, N, ta),
                         c_tflops=tflops(M, K, N, tc)))

    print("\n===== M sweep on the FFN-up shape (K=1536, N=8960) =====")
    print("{:>8} {:>10} {:>10} {:>10} {:>8} {:>9} {:>9}".format(
        "M", "A bf16 ms", "B WO ms", "C row ms", "C/A", "A TFLOP/s", "C TFLOP/s"))
    print("-" * 76)
    K, N = 1536, 8960
    sweep = []
    for M in M_SWEEP:
        ta = run_cfg(M, K, N, "A")
        tb = run_cfg(M, K, N, "B")
        tc = run_cfg(M, K, N, "C")
        print("{:>8} {:>10.4f} {:>10.4f} {:>10.4f} {:>7.2f}x {:>9.1f} {:>9.1f}"
              .format(M, ta, tb, tc, ta / tc,
                      tflops(M, K, N, ta), tflops(M, K, N, tc)), flush=True)
        sweep.append(dict(M=M, a=ta, b=tb, c=tc, c_over_a=ta / tc,
                          a_tflops=tflops(M, K, N, ta),
                          c_tflops=tflops(M, K, N, tc)))

    # Amdahl on the measured production shapes
    ffn = [r for r in rows if r["name"].startswith("ffn")]
    if ffn:
        # ffn.0 and ffn.2 each run 120x/chunk; average their ratios
        ratio = statistics.mean(r["c_over_a"] for r in ffn)
        print(f"\nmean C/A on the two FFN shapes: {ratio:.3f}x")
        print("Amdahl projections (chunk = 1.0, FFN share 0.479, Linear share 0.787):")
        for share, label in ((0.479, "FFN only"), (0.787, "all Linear")):
            if ratio > 1:
                print(f"  {label:<12} -{100*share*(1-1/ratio):.1f}% of chunk")
            else:
                print(f"  {label:<12} FP8 is SLOWER; no gain available here")

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(dict(shapes=rows, m_sweep=sweep,
                   note="A=bf16, B=torchao weight-only FP8, "
                        "C=torchao rowwise FP8 compute"),
              open(f"{args.out_dir}/gemm_bench.json", "w"), indent=1,
              default=str)
    print(f"\nwrote {args.out_dir}/gemm_bench.json")


if __name__ == "__main__":
    main()
