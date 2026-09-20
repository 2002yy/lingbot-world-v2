#!/usr/bin/env python
"""P1-Pre part 3: the M mismatch, and what it does to the FP8 decision.

DISCOVERY: every experiment harness in this project built its denoise loop with
ONE latent frame per forward (cur = randn(16, 1, lat_h, lat_w)), giving M = 627.
Production `generate()` defaults to `chunk_size = 3`, so it feeds THREE latent
frames per forward and M = 3 x 627 = 1881, with max_seq_len = 1881.

That matters because M is exactly the variable that decides whether FP8 pays:
part 2 measured the FFN-up shape (K=1536, N=8960) as

    M= 627   bf16 0.555  rowwise 0.565   C/A 0.98x
    M=1254   bf16 1.074  rowwise 0.839   C/A 1.28x
    M=1881   bf16 1.651  rowwise 1.219   C/A 1.35x
    M=3762   bf16 3.059  rowwise 1.967   C/A 1.56x

So at the harness M FP8 loses, and at the production M it wins. This script
measures BOTH FFN shapes (up and down) at both M values to settle it, and
reports the Amdahl projection at the production M.

Run:
  python gemm_bench_m1881.py
"""
import json
import os
import statistics
import time

import torch
import torch.nn as nn


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


def main():
    dev = "cuda"
    from torchao.quantization import (
        Float8WeightOnlyConfig, Float8DynamicActivationFloat8WeightConfig,
        quantize_)
    print(f"torch {torch.__version__}  torchao "
          f"{__import__('torchao').__version__}  "
          f"{torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).multi_processor_count} SMs)")

    # (label, K, N) for the two FFN GEMMs
    GEMMS = [("ffn.0 (up)  ", 1536, 8960), ("ffn.2 (down)", 8960, 1536)]
    MS = [627, 1881, 3762]

    print("\n{:<14} {:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>8} {:>9} {:>9}"
          .format("gemm", "M", "K", "N", "A bf16", "B WO", "C row", "C/A",
                  "A TF/s", "C TF/s"))
    print("-" * 106)
    rows = []
    for name, K, N in GEMMS:
        lin0 = nn.Linear(K, N, bias=False).to(dev, torch.bfloat16)
        with torch.no_grad():
            lin0.weight.normal_(0, 0.02)
        for M in MS:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)

            def mk(cfg):
                import copy
                lin = copy.deepcopy(lin0)
                if cfg == "B":
                    quantize_(lin, Float8WeightOnlyConfig())
                elif cfg == "C":
                    quantize_(lin, Float8DynamicActivationFloat8WeightConfig())
                with torch.no_grad():
                    return bench(lambda: lin(x))

            ta, tb, tc = mk("A"), mk("B"), mk("C")
            print("{:<14} {:>6} {:>6} {:>6} {:>10.4f} {:>10.4f} {:>10.4f} "
                  "{:>7.2f}x {:>9.1f} {:>9.1f}".format(
                      name, M, K, N, ta, tb, tc, ta / tc,
                      tflops(M, K, N, ta), tflops(M, K, N, tc)), flush=True)
            rows.append(dict(gemm=name.strip(), M=M, K=K, N=N, a=ta, b=tb, c=tc,
                             c_over_a=ta / tc))

    # Production has 2 FFN GEMMs per block-forward; both run at M=1881.
    print("\n===== decision at the PRODUCTION M=1881 =====")
    for M in MS:
        rs = [r for r in rows if r["M"] == M]
        # weight each GEMM by its bf16 cost, which is what the share refers to
        tot_a = sum(r["a"] for r in rs)
        tot_c = sum(r["c"] for r in rs)
        blended = tot_a / tot_c
        print(f"  M={M:<6} blended FFN C/A = {blended:.3f}x   "
              f"(up {rs[0]['c_over_a']:.2f}x, down {rs[1]['c_over_a']:.2f}x)")
        if blended > 1:
            print(f"           Amdahl if FFN is 47.9% of chunk: "
                  f"-{100*0.479*(1-1/blended):.1f}% of chunk time")
            print(f"           Amdahl if all Linear (78.7%):     "
                  f"-{100*0.787*(1-1/blended):.1f}% of chunk time")
        else:
            print("           FP8 slower than bf16; no gain available")

    os.makedirs("output/gemm_roofline", exist_ok=True)
    json.dump(dict(rows=rows, note="harness M=627 vs production M=1881 "
                                   "(generate chunk_size=3)"),
              open("output/gemm_roofline/gemm_bench_m1881.json", "w"),
              indent=1, default=str)
    print("\nwrote output/gemm_roofline/gemm_bench_m1881.json")


if __name__ == "__main__":
    main()
