#!/usr/bin/env python
"""Per-motion breakdown of the inertia T50/T20/Trev from inertia.json."""
import json
import numpy as np

d = json.load(open("output/inertia/inertia.json"))
rows = []
for k, v in d.items():
    # key: "(scene, motion, seed, branch)"
    kk = k.strip("()").replace("'", "")
    parts = [p.strip() for p in kk.split(",")]
    scene, motion, seed, branch = parts[0], parts[1], int(parts[2]), parts[3]
    rows.append(dict(scene=scene, motion=motion, seed=seed, branch=branch,
                     T50=v["T50"], T20=v["T20"], T0=v["T0"], Trev=v["Trev"]))

print(f"{'motion':10s} {'branch':9s} {'n':>3s} {'T50':>6s} {'T20':>6s} "
      f"{'Trev':>6s} {'Trev_ok':>8s}")
for motion in ["yaw", "fwd", "strafe", "walk_turn"]:
    for branch in ["STOP", "REVERSE"]:
        sub = [r for r in rows if r["motion"] == motion and r["branch"] == branch]
        if not sub:
            continue
        def mean_ok(key):
            vals = [r[key] for r in sub if r[key] >= 0]
            return (np.mean(vals) if vals else float("nan"),
                    f"{len(vals)}/{len(sub)}")
        t50, o50 = mean_ok("T50")
        t20, o20 = mean_ok("T20")
        trev, ot = mean_ok("Trev")
        print(f"{motion:10s} {branch:9s} {len(sub):3d} {t50:6.2f} {t20:6.2f} "
              f"{trev:6.2f} {ot:>8s}")

print()
print("Trev (chunk where the REVERSE branch actually flips sign):")
for motion in ["yaw", "fwd", "strafe", "walk_turn"]:
    sub = [r for r in rows if r["motion"] == motion and r["branch"] == "REVERSE"]
    vals = sorted(r["Trev"] for r in sub)
    print(f"  {motion:10s} Trev = {vals}  (n_never = "
          f"{sum(1 for v in vals if v < 0)})")

print()
print("By scene:")
for scene in sorted({r["scene"] for r in rows}):
    for branch in ["STOP", "REVERSE"]:
        sub = [r for r in rows if r["scene"] == scene and r["branch"] == branch]
        t50 = [r["T50"] for r in sub if r["T50"] >= 0]
        t20 = [r["T20"] for r in sub if r["T20"] >= 0]
        print(f"  {scene} {branch:9s} T50={np.mean(t50):.2f} "
              f"({len(t50)}/{len(sub)})  T20={np.mean(t20):.2f}")
