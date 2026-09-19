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
