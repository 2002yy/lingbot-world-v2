#!/usr/bin/env python
"""Per-region distribution of the matched-control residual E_state."""
import json
import numpy as np

d = json.load(open("output/control/control.json"))
rows = d["rows"]
regions = sorted({r["region"] for r in rows})
print(f"{'region':>8s} {'n':>4s} {'E_state mean':>13s} {'sd':>7s} "
      f"{'p50':>7s} {'p95':>7s} {'max':>7s} {'min':>7s} {'|E|>10':>7s}")
for rg in regions:
    e = np.array([r["e_state"] for r in rows if r["region"] == rg])
    print(f"{rg:>8s} {len(e):4d} {e.mean():13.2f} {e.std():7.2f} "
          f"{np.percentile(e,50):7.2f} {np.percentile(e,95):7.2f} "
          f"{e.max():7.2f} {e.min():7.2f} {int((np.abs(e)>10).sum()):7d}")

allE = np.array([r["e_state"] for r in rows])
print(f"\nALL      {len(allE):4d} {allE.mean():13.2f} {allE.std():7.2f} "
      f"{np.percentile(allE,50):7.2f} {np.percentile(allE,95):7.2f} "
      f"{allE.max():7.2f} {allE.min():7.2f} {int((np.abs(allE)>10).sum()):7d}")
print(f"\nfraction |E_state| <= 10 : {np.mean(np.abs(allE)<=10)*100:.1f}%")
print(f"fraction E_state  <= 10 : {np.mean(allE<=10)*100:.1f}%  "
      f"(one-sided, the direction that means 'more contamination than control')")

print("\nworst 5 samples:")
for r in sorted(rows, key=lambda r: -r["e_state"])[:5]:
    print(f"  chunk {r['chunk']:3d} {r['region']:>8s} "
          f"E_live {r['e_live']:+8.2f} E_ctrl {r['e_ctrl']:+8.2f} "
          f"E_state {r['e_state']:+8.2f}")

# does the residual grow with chunk index (i.e. accumulate)?
print("\nE_state vs chunk index (mean per chunk, all regions):")
per = [np.mean([r["e_state"] for r in rows if r["chunk"] == c])
       for c in range(max(r["chunk"] for r in rows) + 1)]
first_half = float(np.mean(per[:len(per)//2]))
second_half = float(np.mean(per[len(per)//2:]))
print(f"  first half mean {first_half:+.2f}   second half mean {second_half:+.2f} "
      f"  drift {second_half-first_half:+.2f}")
sw = [22, 32, 40]
print(f"  at switch chunks: "
      f"{[round(per[c],2) for c in sw if c < len(per)]}")
print(f"  at non-switch chunks mean: "
      f"{np.mean([per[c] for c in range(len(per)) if c not in sw]):+.2f}")
