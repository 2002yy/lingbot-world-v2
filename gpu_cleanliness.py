#!/usr/bin/env python
"""A fixed GPU cleanliness contract, so a gate cannot be negotiated down per run.

WHY THIS EXISTS

While bringing up §Latency-3B the same free-memory threshold was lowered three times --
7200, then 6000, then 4500 MiB -- each time to make an experiment fit. That is the
dangerous pattern, not the 3.2 GiB itself: an A arm measured under 1.5 GiB of ghost
occupancy and a B arm under 3 GiB differ in allocator pressure, clock and power state,
WSL scheduling and OOM headroom, and every one of those differences would be available to
be misread as the effect being measured.

So the thresholds here are CONSTANTS. They are chosen once, from observations, and a run
that cannot meet them is refused rather than accommodated.

WHAT IS MEASURED, AND WHY THE DELTA IS THE KEY NUMBER

    host_used_mib        nvidia-smi memory.used, device level
    wsl_attributed_mib   the sum nvidia-smi --query-compute-apps reports for WSL
    unattributed_mib     the difference

An empty machine still reports a few hundred MiB used, and that is normal. What is not
normal is 1.5-3.2 GiB used with ZERO attributed processes, which is what has been observed
here repeatedly. That delta is the number this contract gates on: it directly measures the
occupancy that no one can explain.

Observed on this machine, for the record:

    clean      16, 60, 262, 419 MiB used with no process   -> unattributed well under 700
    ghost      1671, 3195 MiB used with no process        -> unattributed over 1500

Usage:  python gpu_cleanliness.py            # one sample, verdict, exit code
        python gpu_cleanliness.py --json     # machine readable
        from gpu_cleanliness import sample, verdict
"""
from __future__ import annotations

import json
import subprocess
import sys
import time

# ---- FIXED. Do not tune these per experiment. ---------------------------------
MAX_UNATTRIBUTED_MIB = 700
MIN_FREE_MIB = 6000
# The unattributed floor above is deliberately far below the smallest ghost seen
# (1671) and far above the largest clean reading (419), so it separates the two
# populations with no judgement call.

_QUERY = ("name,memory.total,memory.used,memory.free,utilization.gpu,"
          "pstate,power.draw,temperature.gpu")


def sample():
    """One device sample, plus WSL's own attribution, with the delta computed."""
    t = time.time()
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={_QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        parts = [x.strip() for x in out.split(",")]
        (name, total, used, free, util, pstate, power, temp) = parts
    except Exception as e:
        return dict(ok=False, sampled_at=t, error=f"{type(e).__name__}: {e}",
                    thresholds=dict(max_unattributed_mib=MAX_UNATTRIBUTED_MIB,
                                    min_free_mib=MIN_FREE_MIB))

    attributed = 0
    try:
        a = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        attributed = sum(int(x.strip()) for x in a.splitlines() if x.strip().isdigit())
    except Exception:
        attributed = 0

    used_i, free_i, total_i = int(used), int(free), int(total)
    unattributed = used_i - attributed
    return dict(
        ok=True, sampled_at=t, name=name,
        total_mib=total_i, used_mib=used_i, free_mib=free_i,
        wsl_attributed_mib=attributed, unattributed_mib=unattributed,
        util_pct=util, pstate=pstate, power_w=power, temp_c=temp,
        thresholds=dict(max_unattributed_mib=MAX_UNATTRIBUTED_MIB,
                        min_free_mib=MIN_FREE_MIB))


def verdict(s=None):
    """(is_clean, reasons, record). Fixed thresholds, no negotiation."""
    s = s or sample()
    if not s.get("ok"):
        return False, [f"could not sample the device: {s.get('error')}"], s
    reasons = []
    if s["unattributed_mib"] > MAX_UNATTRIBUTED_MIB:
        reasons.append(
            f"{s['unattributed_mib']} MiB is used by NO attributable process "
            f"(limit {MAX_UNATTRIBUTED_MIB}). This is the ghost occupancy; a run "
            f"started now would not be comparable to one started on a clean device.")
    if s["free_mib"] < MIN_FREE_MIB:
        reasons.append(
            f"only {s['free_mib']} MiB free (fixed floor {MIN_FREE_MIB}). "
            f"NOT 'close enough' -- the floor is not to be lowered to fit a run.")
    return (not reasons), reasons, s


def describe(s):
    return (
        f"gpu [{time.strftime('%H:%M:%S', time.localtime(s['sampled_at']))}] "
        f"used={s['used_mib']} free={s['free_mib']} of {s['total_mib']} MiB  "
        f"wsl_attributed={s['wsl_attributed_mib']} "
        f"UNATTRIBUTED={s['unattributed_mib']} (limit {MAX_UNATTRIBUTED_MIB})  "
        f"util={s['util_pct']}% pstate={s['pstate']} "
        f"power={s['power_w']}W temp={s['temp_c']}C")


def main():
    clean, reasons, s = verdict()
    if "--json" in sys.argv:
        print(json.dumps(dict(clean=clean, reasons=reasons, sample=s), indent=1))
        return 0 if clean else 1
    print(describe(s))
    if clean:
        print("  CLEAN: sample is admissible")
        return 0
    print("  NOT CLEAN: refusing to sample")
    for r in reasons:
        print(f"    - {r}")
    print("  Fix the device (wait for the driver to reclaim it, or reset the VM from")
    print("  Windows with  wsl --shutdown ). Do not lower the thresholds.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
