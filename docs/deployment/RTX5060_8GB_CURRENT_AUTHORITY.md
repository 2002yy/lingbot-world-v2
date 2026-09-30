# RTX 5060 Laptop 8GB — Current Deployment Authority

This document defines what may currently be called an RTX 5060 Laptop 8GB deployment result.

## Scope

Target hardware:

- NVIDIA RTX 5060 Laptop GPU;
- 8GB VRAM;
- Blackwell / sm_120;
- local single-session deployment.

Target model family:

- LingBot-World V2 / LingBot-World-Infinity 1.3B causal-fast.

## Geometry

Current validated 8GB deployment profile:

- pixel geometry: **304x528**;
- historical benchmark/harness geometry such as 320x480 must be labeled as benchmark-only;
- upstream/model nominal geometry 512x768 must not be substituted silently.

## Reference path

Reference numerical path:

- original BF16 weights;
- FA2 reproducible attention path.

Validated same-seed BF16-vs-BF16 checks were bit-identical in the relevant reference experiments.

## Deployment weight modes

### `bf16`

Role: performance deployment mode.

Properties:

- original weights;
- faster DiT than the current weight-only FP8 implementation;
- higher VRAM use.

### `fp8_lowmem`

Role: resilience / compatibility mode.

Properties:

- weight-only FP8;
- lower VRAM;
- slower than BF16 in the measured 5060 path;
- intended for tighter VRAM conditions / extra display-preview pressure / system variance.

## Condition encoder

Default:

```text
LINGBOT_STREAM_ENCODE=1
```

The streamed encoder mirrors the causal VAE encoder's native temporal block boundaries and has been validated as bit-identical for condition latents and final outputs in tested cases.

Long requests benefit most because the path avoids materializing the full zero-filled pixel-time tensor.

## Repeated requests / offload

Repeated request correctness requires the request-entry device restore fix.

A request is not considered deployment-valid if it succeeds once but leaves the model on CPU and fails the next request.

## Memory stability

Memory stability is evaluated with:

- allocated trend;
- reserved plateau;
- late-window reserved slope;
- minimum free headroom.

Do not classify allocator warm-up as a leak solely from first-request vs last-request reserved memory.

## Measurement authority

Results must declare one denominator:

- micro/kernel;
- DiT + lightweight decoder;
- full deployment request.

Full deployment request is the only denominator allowed for end-user request-latency claims.

## Historical latency terminology

The following classes are not authoritative input-to-display measurements:

- formula-derived control latency;
- command-line configured cadence;
- decode-return timestamps;
- mixed-clock-base proxies.

Until §Latency-1 is implemented, call them estimates or decode-complete proxies.

## Open item

§Deploy-1 must freeze final BF16-performance and FP8-lowmem numbers using the real repeated deployment chain with streamed encode enabled.
