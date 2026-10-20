#!/usr/bin/env python
"""Evidence-driven local-window presets for LingBot-World.

Design principle: more history is NOT universally better. The 24-run memory
matrix showed a scene x config interaction, so 8/2 is a *temporal-stability
mode* for specific scenes, not a higher quality tier.

    DEFAULT            local_window=6, secondary=1   (development default)
    TEMPORAL_STABILITY local_window=8, secondary=2   (opt-in, evidence gated)

Upgrade to 8/2 only when a short calibration probe shows, on >= 2 repeats:
    1. closure nGood:  ngood_82 >= 1.30 * ngood_61
    2. memory:         dino_82 <= dino_61 - 0.03   OR   dino_82 <= 0.85 * dino_61
    3. no catastrophic failure: dino_82 <= 0.6 and ngood_82 >= 0.5 * ngood_61
"""

PRESETS = {
    "default_6_1": dict(local_window=6, secondary=1),
    "temporal_stability_8_2": dict(local_window=8, secondary=2),
}

DEFAULT_PRESET = "default_6_1"

NG_MIN_RATIO = 1.30      # 8/2 must beat 6/1 by >= 30% on closure nGood
DINO_ABS_GAIN = 0.03     # absolute DINO distance improvement
DINO_REL_GAIN = 0.15     # or 15% relative improvement
DINO_CATASTROPHIC = 0.60
NG_FLOOR_RATIO = 0.50
MIN_REPEATS = 2


def catastrophic(ngood, dino, ngood_baseline):
    if dino > DINO_CATASTROPHIC:
        return True
    if ngood_baseline > 0 and ngood < NG_FLOOR_RATIO * ngood_baseline:
        return True
    return False


def gate(ngood_61, ngood_82, dino_61, dino_82, ngood_baseline=None):
    """Return (passes, reason) for a single repeat."""
    base = ngood_baseline if ngood_baseline is not None else ngood_61
    if catastrophic(ngood_82, dino_82, base):
        return False, "catastrophic"
    if not (ngood_82 >= NG_MIN_RATIO * ngood_61):
        return False, f"ngood {ngood_82:.0f} < 1.3*{ngood_61:.0f}"
    dino_ok = (dino_82 <= dino_61 - DINO_ABS_GAIN) or \
              (dino_82 <= (1.0 - DINO_REL_GAIN) * dino_61)
    if not dino_ok:
        return False, f"dino {dino_82:.3f} vs {dino_61:.3f} (no gain)"
    return True, "pass"


def select_preset(repeats, min_repeats=MIN_REPEATS):
    """repeats: list of dicts with ngood_61, ngood_82, dino_61, dino_82,
    and optionally ngood_baseline. Returns (preset_name, detail)."""
    passed, details = 0, []
    for i, r in enumerate(repeats):
        ok, why = gate(r["ngood_61"], r["ngood_82"], r["dino_61"], r["dino_82"],
                       r.get("ngood_baseline"))
        passed += int(ok)
        details.append((i, ok, why))
    name = ("temporal_stability_8_2" if passed >= min_repeats else DEFAULT_PRESET)
    return name, dict(passed=passed, needed=min_repeats, details=details)


if __name__ == "__main__":
    # regression: the two measured scenes
    cases = {
        "scene04 (expected 8/2)": [
            dict(ngood_61=27, ngood_82=65, dino_61=0.241, dino_82=0.126),
            dict(ngood_61=16, ngood_82=53, dino_61=0.162, dino_82=0.110),
        ],
        "scene01 (expected 6/1)": [
            dict(ngood_61=87, ngood_82=102, dino_61=0.039, dino_82=0.117),
            dict(ngood_61=139, ngood_82=123, dino_61=0.043, dino_82=0.070),
        ],
    }
    for name, reps in cases.items():
        preset, detail = select_preset(reps)
        print(f"{name:26s} -> {preset:24s} {detail['passed']}/{detail['needed']}  "
              f"{[d[2] for d in detail['details']]}")
