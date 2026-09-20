"""Performance mode presets: reproducibility vs speed.

THE FINDING THAT FORCED THIS
----------------------------
Long-horizon testing (65 chunks, ~16 s of world time) established:

    FA2   + eager     858.7 ms   -- deterministic; two identical runs are
                                    bit-for-bit identical (PSNR 100.00,
                                    LPIPS 0.0000 over all 21 measured chunks)
    Hybrid+ eager     810.6 ms   -5.61%
    Hybrid+ compile   798.5 ms   -7.02%

but BOTH accelerated configurations diverge from FA2's trajectory. By chunk 32
the world has changed, not just the texture:

    metric         B vs A     D vs A
    LPIPS          0.4701     0.5207
    SSIM           0.3646     0.3117
    edge-SSIM      0.2971     0.2560
    DINO cosine    0.5176     0.6521

Falling back from compile to plain eager does NOT fix this -- the two are the
same order. The cause is the attention numerical path itself: any change to it
is chaotically amplified by the recurrent rollout. A zero-dependency SDPA swap
would be expected to do the same.

So the two goals are mutually exclusive, and the honest conclusion is:

    the accelerated backends are not WRONG, they produce a DIFFERENT, equally
    plausible trajectory. They simply cannot be the default when a seed is
    expected to reproduce a specific video.

OBSERVED BOUNDARY (evidence strength matters here)
    ~20 chunks: structure preserved in the tested scene/seed
    ~32 chunks: structural divergence observed
This boundary comes from ONE seed in ONE scene. It is NOT a universal safe
threshold and must not be written as a guarantee. A multi-seed, multi-scene
sweep would be needed before any rollout-length limit could be enforced.

MODES
    repro  (default)  backend=fa2,    compile=0   exact-trajectory reproduction
    fast              backend=hybrid, compile=1   -7.0%, different trajectory

    LINGBOT_MODE=repro|fast
    LINGBOT_ATTN_BACKEND=fa2|sdpa|sage|hybrid   per-field override
    LINGBOT_COMPILE=0|1                         per-field override
    LINGBOT_COMPILE_MODE=default (Inductor)     NOT reduce-overhead, NOT CUDA Graph

CUDA Graph remains a separate, closed line: CUDAGraph Trees skips capture
because the KV cache is a mutated eager input, and forcing it dies in
_cuda_setCheckpointPoolState.
"""
import os

PRESETS = {
    # default: bit-exact reproduction of the original trajectory
    "repro": dict(backend="fa2", compile=False),
    # explicit opt-in: ~7% faster, produces a different trajectory at long horizon
    "fast": dict(backend="hybrid", compile=True),
}

VALID_MODES = tuple(PRESETS)


def resolve():
    """Resolve env vars into an effective configuration, applying overrides."""
    mode = os.environ.get("LINGBOT_MODE", "repro").strip().lower()
    if mode not in PRESETS:
        raise ValueError(
            f"LINGBOT_MODE={mode!r} is not one of {VALID_MODES}")
    eff = dict(PRESETS[mode])
    eff["mode"] = mode

    ov_b = os.environ.get("LINGBOT_ATTN_BACKEND", "").strip().lower()
    if ov_b:
        eff["backend"] = ov_b
    ov_c = os.environ.get("LINGBOT_COMPILE", "").strip()
    if ov_c != "":
        eff["compile"] = ov_c not in ("0", "false", "False", "no")
    eff["compile_mode"] = os.environ.get("LINGBOT_COMPILE_MODE", "default")

    eff["reproducible"] = (eff["backend"] == "fa2" and not eff["compile"])
    return eff


def describe(eff=None):
    eff = eff or resolve()
    b = eff["backend"]
    c = f"+compile({eff['compile_mode']})" if eff["compile"] else "+eager"
    return f"LINGBOT_MODE={eff['mode']} backend={b}{c}"


def apply(model, eff=None, log=print):
    """Set the attention backend and optionally torch.compile the model.

    Returns the (possibly wrapped) model. Called at the end of the pipeline
    constructor so all attribute lookups on the original model have already
    happened.
    """
    eff = eff or resolve()
    import wan.modules.sage_backend as sb
    sb.set_backend(eff["backend"])

    log(f"[perf_mode] {describe(eff)}")
    if not eff["reproducible"]:
        log("[perf_mode] NOTE: this mode changes the attention numerical path. "
            "The causal rollout is chaotically sensitive to it, so long "
            "rollouts (~32+ chunks in the tested scene/seed) diverge into a "
            "different, equally plausible world. Use LINGBOT_MODE=repro when a "
            "seed must reproduce a specific video.")

    if not eff["compile"]:
        return model
    import torch
    return torch.compile(model, mode=eff["compile_mode"], fullgraph=False)


# --------------------------------------------------------------------------
# P2a: selective FFN up-projection in rowwise FP8
# --------------------------------------------------------------------------
# Measured at M=1881 on this machine, per call:
#
#     ffn.0 up   (K=1536 N=8960)  bf16 1.6260 | weight-only 2.7110 | rowwise 1.1058
#     ffn.2 down (K=8960 N=1536)  bf16 1.6896 | weight-only 2.7811 | rowwise 2.4603
#
# The production path uses weight-only FP8, so the real lever on ffn.0 is
# 2.7110 -> 1.1058, i.e. 1.6052 ms/call. At 120 calls per chunk that projects to
# ~193 ms/chunk, about -12.7% of the 1524.5 ms production baseline -- roughly
# twice what comparing against bf16 would suggest. The down-projection is left
# alone because rowwise loses there (narrow N=1536 cannot amortise the
# quantisation).
#
# This CHANGES the numerical path, so it is a fast-mode-class lever: it must
# pass the 21-chunk and 65-chunk rollout gates before being trusted.
def ffn0_fp8_enabled():
    import os
    return os.environ.get("LINGBOT_FFN0_FP8", "0") not in ("0", "false", "False")


def apply_ffn0_fp8(model, log=print):
    """Replace each block's FFN up-projection with a rowwise-FP8 compute module.

    Must run AFTER the weight-only pass, which skips ffn.0 whenever
    LINGBOT_FFN0_FP8 is set (see lingbot_fp8.lingbot_fp8_filter), so the rowwise
    quantisation sees the original bf16 weights rather than quantised ones.

    LINGBOT_FFN0_FP8_DEFER=1 makes this a no-op. Needed by A/B harnesses that
    want to hold BOTH the bf16 and the rowwise module and swap them at runtime:
    the flag still makes the weight-only pass skip ffn.0 (so bf16 is preserved),
    but leaves the conversion to the harness.
    """
    import os
    if os.environ.get("LINGBOT_FFN0_FP8_DEFER", "0") not in ("0", "false", "False"):
        log("[perf_mode] P2a: deferred to the caller "
            "(LINGBOT_FFN0_FP8_DEFER=1)")
        return 0
    if not ffn0_fp8_enabled():
        return 0
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig, quantize_)
    n = 0
    for blk in getattr(model, "blocks", []):
        ffn = getattr(blk, "ffn", None)
        up = ffn[0] if ffn is not None and len(ffn) > 0 else None
        if up is None:
            continue
        quantize_(up, Float8DynamicActivationFloat8WeightConfig())
        n += 1
    log(f"[perf_mode] P2a: {n} FFN up-projections converted to rowwise FP8 "
        f"(numerical path changed -> fast-mode class; needs rollout gates)")
    return n
