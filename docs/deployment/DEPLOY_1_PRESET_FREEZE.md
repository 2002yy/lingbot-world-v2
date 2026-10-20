# §Deploy-1 — RTX 5060 Laptop 8GB Preset Freeze

Status: **CLOSED** — frozen 2026-09-30 (commit `b214b33`).

## Objective

Freeze the first publishable RTX 5060 Laptop 8GB deployment presets from the real
repeated deployment path.

## Authority configuration

Common:

- geometry: 304x528 (latent 38x66, frame_seqlen 627);
- real repeated `WanI2VCausal.generate()` deployment path, 81 frames (its own
  default), chunk_size 3, 4 timesteps;
- preset local_window 6 / sink 1;
- request-entry device restore fix enabled;
- streamed condition encode enabled by default (`LINGBOT_STREAM_ENCODE=1`);
- deployment measurements, not benchmark-harness projections.

Arm A:

```text
preset = performance
LINGBOT_WEIGHT_MODE=bf16
```

Arm B:

```text
preset = lowmem
LINGBOT_WEIGHT_MODE=fp8_lowmem
```

## Frozen measurements

| Metric | BF16 performance | FP8 lowmem |
|---|---:|---:|
| cold first request | 119.86 s | 131.10 s |
| warm request p50 | 29.25 s | 30.84 s |
| warm request p95 | 31.49 s | 31.29 s |
| warm min / max | 27.60 / 46.29 s | 29.56 / 33.32 s |
| output FPS | 2.769 | 2.626 |
| max reserved | 7104 MiB | 5846 MiB |
| min free VRAM | 821 MiB | 1161 MiB |
| reserved plateau | 5999 MiB | 5720 MiB |
| repeated request count | 25 | 25 |
| allocated creep | +0 MiB | +0 MiB |
| reserved late-window slope | +0 MiB | +63 MiB |
| condition y hash | 7741be09201d5d6d | 7741be09201d5d6d |

DiT / condition-encode / decode phase split: reported in
`docs/P2B_PROD_AMDAHL.md` from the instrumented run (bf16 DiT 10.79 s, fp8 DiT
14.01 s; encode and decode identical across arms). That breakdown predates the
streamed-encode default flip, so the condition-encode figure there is a
whole-clip number; the DiT figures are unaffected.

## Supported request-length envelope

| frames | BF16 performance | FP8 lowmem |
|---|---|---|
| 81 | 25/25 ok | 25/25 ok |
| 249 | 2/2 ok | 2/2 ok |
| 501 | 1/2 ok (2nd OOMs) | 2/2 ok |
| 777 | 1/2 ok (2nd OOMs) | 2/2 ok |

At 501 and 777 frames fp8_lowmem still completes two consecutive requests while
bf16 completes only one; the second bf16 request fails because the generation
itself exceeds 8 GB. This is the measured basis for the resilience role.

## Correctness gates

Both presets:

- complete repeated requests without device mismatch — PASS (25/25 each);
- no allocated-memory leak — PASS (allocated flat across all 25);
- reserved reaches a plateau — PASS (late-window slope +0 / +63 MiB);
- condition latent bit-identical between the whole-clip and streamed encoders —
  PASS (M1-0 and M1-2);
- streamed prepare does not contaminate the hot path — PASS (G4-hotpath, DiT,
  decode and combined latency all within ±0.3% under ABBA interleaving).

## Product interpretation

- BF16: performance deployment mode — original weights, 5.4% faster request
  latency, 1258 MiB more peak, 821 MiB minimum headroom; for a free GPU seeking
  the lowest request latency; caveat at 501 frames and beyond.
- FP8-lowmem: resilience / compatibility mode — weight-only FP8, about 1.26 GiB
  less memory, 1161 MiB minimum headroom; for tight memory, preview, other GPU
  workloads and WSL variance; still completes repeated requests at long horizons.

Do not collapse the result to "X is better". The two presets optimize different
constraints.

## Exit criterion

1. both columns filled from the same deployment contract — DONE;
2. repeated-request stability passes — DONE;
3. table copied into deployment documentation — DONE
   (`docs/DEPLOYMENT_PRESETS_FROZEN.md`);
4. preset names become the sole public 8GB preset authority — DONE.

## Note on package provenance

This document arrived as part of an externally prepared roadmap package with every
value marked TBD and the status set to NEXT/OPEN. The freeze had in fact already
been completed; the TBD cells are filled here from the measured run rather than
re-run, and the status corrected. No measurement in this file is a projection.
