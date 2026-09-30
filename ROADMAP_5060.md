# RTX 5060 Laptop 8GB LingBot Deployment Roadmap

## 0. Project goal

This project targets a reproducible, open-source deployment of LingBot-World V2 / LingBot-World-Infinity 1.3B causal-fast on an RTX 5060 Laptop 8GB, with a minimal interactive runtime (WASD / directional camera control) rather than benchmark-only execution.

The project is considered successful when a normal user can:

1. install it on Windows + WSL2 or Linux;
2. run a stable 8GB deployment preset;
3. launch a single authoritative interactive runtime;
4. move / steer with WASD or arrow-key camera controls;
5. see separately reported native generation FPS, display FPS, and measured input-to-present latency;
6. reproduce the documented VRAM, latency, and correctness results.

## 1. Current phase transition

The project is no longer primarily a kernel-optimization project.

Priority is now:

1. correctness;
2. stable long-running operation on 8GB;
3. input-to-display latency;
4. interactive usability;
5. install / deployment reproducibility;
6. additional kernel/FPS optimization only when expected impact is material.

Do not reopen low-level kernel work unless a candidate has a clear expected end-to-end benefit (normally >= 5%) or fixes correctness / memory viability.

## 2. Frozen technical conclusions

### 2.1 Reference mode

Reference authority:

- original BF16 weights;
- FA2 reproducible path;
- same seed / same input must be bit-identical where the path claims reproducibility.

FP8, SageAttention, torch.compile, and other approximate/alternate numerical paths are evaluated relative to this reference.

### 2.2 Weight modes

`LINGBOT_WEIGHT_MODE=bf16`

- performance deployment mode;
- original BF16 weights;
- faster DiT path;
- higher VRAM use.

`LINGBOT_WEIGHT_MODE=fp8_lowmem`

- resilience / compatibility mode;
- weight-only FP8;
- saves roughly 1.3 GiB in the measured 304x528 deployment profile;
- slower than BF16 on the RTX 5060 Laptop in this implementation.

Do not label FP8 as "fast".

### 2.3 Condition encoder

`LINGBOT_STREAM_ENCODE=1` is the deployment default.

Reason:

- bit-identical condition latent output;
- bit-identical final output in validated cases;
- no measurable hot-path regression;
- removes long-request whole-clip condition-materialization OOM cliffs;
- short requests remain effectively neutral.

`LINGBOT_STREAM_ENCODE=0` is legacy/debug only.

### 2.4 Attention / compile

FA2 remains the reproducibility path.

Hybrid attention / compile may be retained as optional performance experiments, but must not be described as long-horizon trajectory preserving. Long causal rollout experiments showed that small numerical-path differences can produce a different valid world trajectory after enough chunks.

CUDA Graph work is closed for the current architecture unless new evidence materially changes the cost/benefit.

### 2.5 Geometry authority

Distinguish:

- model nominal geometry: 512x768;
- historical harness geometry: 320x480;
- RTX 5060 Laptop 8GB deployment profile under current validation: 304x528.

Rule:

> Model nominal geometry is not deployment geometry authority.

Any performance/VRAM number must name its geometry.

### 2.6 Measurement denominator

Every performance result must be classified as exactly one of:

1. micro/kernel benchmark;
2. DiT + lightweight-decoder benchmark profile;
3. full deployment request.

Do not compare percentages across denominators without stating the denominator change.

## 3. What has already been validated

The repository work has established, among other items:

- RTX 5060 Laptop 8GB / Blackwell sm_120 execution;
- BF16 and FP8-lowmem weight paths;
- local-window / KV-cache memory behavior;
- allocator warm-up vs true memory leak methodology;
- streamed causal VAE condition encoding;
- repeated real requests;
- offload device-restore correctness across requests;
- camera controller / hot-swap foundations;
- pose warp / preview / lightweight decode experiments;
- host-sync cleanup;
- torch.compile characterization;
- SageAttention 2.2 Blackwell build and workload-aware dispatch;
- short vs long causal-rollout correctness gates;
- RoPE attribution and stop-loss closure;
- FP8 cost attribution showing FP8 is a memory trade rather than a speed optimization on this 5060 path;
- G4 hot-path pollution gate for streamed condition encoding.

## 4. Open-source deployment target

The intended final user flow is approximately:

```bash
git clone <repo>
./setup.sh            # Linux / WSL2
./run.sh play --preset performance
```

Windows may also expose PowerShell wrappers:

```powershell
.\setup.ps1
.\run.ps1 play --preset performance
```

Minimum interactive controls:

- `W/S`: forward/back;
- `A/D`: strafe or yaw (final mapping must be documented);
- arrow keys or mouse: camera yaw/pitch;
- `R`: reset;
- `P`: pause;
- `Esc`: quit.

## 5. Execution roadmap

### §Deploy-1 — Freeze RTX 5060 Laptop 8GB presets

Status: **CLOSED** (frozen 2026-09-30, commit `b214b33`). The table is in
`docs/deployment/DEPLOY_1_PRESET_FREEZE.md` and
`docs/DEPLOYMENT_PRESETS_FROZEN.md`.

Run the real repeated deployment chain at 304x528 with:

- offload restore fix;
- streamed condition encode default;
- BF16 performance mode;
- FP8-lowmem resilience mode.

Freeze, for each preset:

- cold first-request latency;
- warm request p50/p95;
- output FPS;
- peak allocated / reserved;
- minimum free VRAM;
- reserved plateau;
- repeated-request leak/creep result;
- supported request-length envelope;
- known constraints.

Exit criterion:

A two-column deployment table is stable enough to publish and becomes the only authority for 8GB preset claims.

### §Interactive-1 — Single interactive runtime authority

Create one authoritative runtime entry point.

Example:

```bash
python -m lingbot.play --preset performance
```

or:

```bash
./run.sh play --preset performance
```

Historical harnesses remain research tools and must not independently claim "production" or "interactive deployment" authority.

The runtime must own:

- input queue;
- event IDs;
- tick/chunk assignment;
- in-flight vs committed state;
- camera state;
- decoder/display path;
- metrics hooks.

### §Interactive-2 — WASD / camera control

Integrate the already-proven camera/hot-swap foundations into the authoritative runtime.

Required correctness:

- events applied exactly once;
- stale events detected;
- prewarm never mutates committed runtime state;
- explicit prepare/in-flight/commit lifecycle;
- camera/control sequence reproducible under reference mode where applicable.

### §Latency-1 — Input-to-Present Contract

Define authoritative timestamps:

- `t0`: input accepted into authoritative queue;
- `t1`: input assigned to tick/chunk;
- `t2`: generated state committed;
- `t3`: first affected real frame decoded;
- `t4`: frame submitted to renderer;
- `t5`: frame actually presented.

Publish:

- input->commit;
- input->warp;
- input->preview;
- input->first-real-frame;
- input->present.

Only `t5 - t0` may be called measured input-to-display / control-to-real-display latency.

Historical formula-based, config-based, or decode-return-only values must remain labeled as proxies.

### §Display-1 — Generation/display decoupling

Optional display improvements may include:

- pose warp;
- preview;
- interpolation;
- generated-frame replacement/correction.

Always report separately:

- native world-model generation FPS;
- display FPS;
- input->warp;
- input->preview;
- input->real;
- input->present.

Never market interpolated/display FPS as native generation FPS.

### §Release-1 — Packaging

Prepare:

- `setup.sh`;
- `setup.ps1`;
- `run.sh`;
- `run.ps1`;
- hardware/driver checks;
- model download flow;
- first-run diagnostics;
- performance/lowmem preset docs;
- WSL2 guide;
- Linux guide;
- known limitations;
- troubleshooting.

## 6. Research / deployment boundary

Deployment authority must come from the authoritative deployment path.

Research scripts may:

- isolate kernels;
- use alternate geometry;
- use simplified decoders;
- bypass display;
- use cached condition latents.

But their outputs must be labeled by denominator and geometry and must not silently become deployment claims.

## 7. Stop rules

Before implementing an optimization:

1. prove the suspected production bottleneck exists in the real code path;
2. mirror production shape, dtype, device, stride, operation order, and call mix;
3. run a magnitude / physical-upper-bound sanity check;
4. use microbenchmarks only to rank candidates;
5. announce benefit only after end-to-end validation.

For causal rollout, same-latent or short-horizon similarity is not sufficient evidence of long-horizon trajectory preservation.

## 8. Current next action

§Deploy-1 is CLOSED and §Latency-1A (input-to-present contract and
t0-t5 reachability audit) is CLOSED. Proceed with:

> §Interactive-1 — single interactive runtime authority.

The Latency-1A audit found all six of t0-t5 to be GAPs, two of them
architecturally unrepresentable (no renderer, no present signal). That is the
design input for §Interactive-1: there is currently no single runtime
authority, no event identity, no commit seam and no frame lineage.

Do not open a new kernel optimization branch before §Interactive-1 is closed
unless it fixes a blocking correctness issue.
