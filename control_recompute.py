#!/usr/bin/env python
"""§41F v2: matched-control background contamination.

§41D-2 failed the `ratio` gate (2.43 > 2.00) but the growth started at chunk 1,
BEFORE any damage -- suggesting shared scene drift rather than wall pollution.
Testing that needs a matched control: identical seed / initial frame / camera
sequence / chunk count / refresh strategy / renderer + anchor logic, with the
ONLY difference being no damage, no stage crossing, no damage-anchor
canonicalisation.

Three quantities per chunk (object-external ROIs):
    E_live(t)  = excess in the §41D-2 run
    E_ctrl(t)  = excess in the no-change control, same roi, same chunk
    E_state(t) = E_live(t) - E_ctrl(t)      <- the part attributable to the
                                               state change / re-anchoring

`E_state` is the primary statistic. `ratio = live/control` is demoted to a
diagnostic because its denominator is only ~2-3 grey levels, which turns
ordinary noise into a 2.4x reading. If a ratio is wanted, the denominator
floor epsilon must come from this control distribution, not be guessed.

  python control_recompute.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from world_metrics import bg_excess, tile_bank, bb_to_roi  # noqa: E402

WALL_BB = (0.28, 0.56, 0.52, 0.90)


def main():
    live = np.load("output/damage2/frames.npy")
    live_d0 = np.load("output/damage2/D0_frames.npy")
    ctrl = np.load("output/control/frames.npy")
    ctrl_d0 = np.load("output/control/D0_frames.npy")
    n = min(len(live), len(ctrl), len(live_d0), len(ctrl_d0))
    print(f"[cc] matched control: live {len(live)} chunks, "
          f"control {len(ctrl)} chunks -> comparing {n}", flush=True)

    H, W = live[0].shape[:2]
    roi = bb_to_roi(WALL_BB, H, W)
    th, tw = roi[1] - roi[0], roi[3] - roi[2]

    # object-external ROIs: the wall's immediate surroundings (where pollution
    # would actually show), plus two far regions
    near = {
        "above": (max(0, roi[0] - th), max(0, roi[0] - th) + th, roi[2],
                  min(W, roi[2] + tw)),
        "below": (min(H - th, roi[1]), min(H - th, roi[1]) + th, roi[2],
                  min(W, roi[2] + tw)),
        "left": (roi[0], min(H, roi[0] + th), max(0, roi[2] - tw),
                 max(0, roi[2] - tw) + tw),
        "right": (roi[0], min(H, roi[0] + th), min(W - tw, roi[3]),
                  min(W - tw, roi[3]) + tw),
    }
    far = {
        "far_TL": (int(0.06 * H), int(0.06 * H) + th, int(0.06 * W),
                   int(0.06 * W) + tw),
        "far_TR": (int(0.06 * H), int(0.06 * H) + th, int(0.72 * W),
                   int(0.72 * W) + tw),
    }
    regions = {}
    for k, v in near.items():
        if v[1] <= H and v[3] <= W and v[1] > v[0] and v[3] > v[2]:
            regions[k] = v
    for k, v in far.items():
        if v[1] <= H and v[3] <= W:
            regions[k] = v
    print(f"[cc] regions: {list(regions.keys())}", flush=True)

    # one shared bank for both runs (excludes the object + a margin)
    allrois = list(regions.values())
    ex = [(roi[0] - th, roi[1] + th, roi[2] - tw, roi[3] + tw)] + \
         [(a - th // 2, b + th // 2, c - tw // 2, e + tw // 2)
          for (a, b, c, e) in allrois]
    bank = tile_bank(H, W, (th, tw), n=64, exclude=ex)
    print(f"[cc] shared control bank: {len(bank)} tiles", flush=True)

    rows = []
    for cid in range(n):
        for rname, r in regions.items():
            el = bg_excess(live[cid], live_d0[cid], r, bank)
            ec = bg_excess(ctrl[cid], ctrl_d0[cid], r, bank)
            rows.append(dict(chunk=cid, region=rname,
                             e_live=el["excess"], e_ctrl=ec["excess"],
                             e_state=el["excess"] - ec["excess"],
                             ratio_live=el["ratio"], ratio_ctrl=ec["ratio"],
                             ctrl_raw=ec["raw"], ctrl_bank=ec["ctrl"]))
    print(f"[cc] measured {len(rows)} (chunk, region) samples", flush=True)

    # ---------- report ----------
    print("\n[cc] ===== per-region summary (all chunks) =====")
    print(f"  {'region':>8s} {'E_live':>9s} {'E_ctrl':>9s} {'E_state':>9s} "
          f"{'ctrl_bank':>10s} {'n':>4s}")
    per = {}
    for rname in list(regions.keys()):
        sub = [r for r in rows if r["region"] == rname]
        if not sub:
            continue
        per[rname] = dict(
            e_live=float(np.mean([s["e_live"] for s in sub])),
            e_ctrl=float(np.mean([s["e_ctrl"] for s in sub])),
            e_state=float(np.mean([s["e_state"] for s in sub])),
            ctrl_bank=float(np.mean([s["ctrl_bank"] for s in sub])),
            n=len(sub))
        p = per[rname]
        print(f"  {rname:>8s} {p['e_live']:9.2f} {p['e_ctrl']:9.2f} "
              f"{p['e_state']:9.2f} {p['ctrl_bank']:10.2f} {p['n']:4d}")

    es = np.array([r["e_state"] for r in rows])
    el = np.array([r["e_live"] for r in rows])
    ec = np.array([r["e_ctrl"] for r in rows])
    print(f"\n[cc] ===== E_state (matched-control residual) =====")
    print(f"  mean {es.mean():+.2f}  sd {es.std():.2f}  "
          f"p90 {np.percentile(es,90):+.2f}  max {es.max():+.2f}")
    print(f"  E_live mean {el.mean():+.2f}  E_ctrl mean {ec.mean():+.2f}")
    thr = float(np.ceil(np.percentile(np.abs(es), 95) + 0.5))
    print(f"\n  => PASS threshold for background contamination:")
    print(f"       E_state <= {thr:.1f} grey levels   (control |E_state| p95 + 0.5)")

    # control-derived denominator floor for ratio
    cb = np.array([r["ctrl_bank"] for r in rows])
    eps = float(np.ceil(np.percentile(cb, 90)))
    print(f"       ratio denominator floor epsilon >= {eps:.1f} grey levels")
    print(f"         (control bank p90; below this the ratio is unstable)")

    # ---------- verdict on §41D-2 ----------
    print("\n[cc] ===== §41D-2 verdict =====")
    live_cfg = json.load(open("output/damage2/damage2.json"))
    sw = [c["chunk"] for c in live_cfg["canon_hist"]]
    print(f"  switch chunks were {sw}")
    for c in sw:
        sub = [r for r in rows if r["chunk"] == c]
        if sub:
            m = float(np.mean([s["e_state"] for s in sub]))
            lr = float(np.mean([s["ratio_live"] for s in sub]))
            print(f"    chunk {c}: E_state {m:+.2f}  raw ratio {lr:.2f}")
    live_ratio_max = max(r["ratio_live"] for r in rows)
    print(f"\n  live raw ratio reached {live_ratio_max:.2f} (> 2.00 -> §41D-2 gate 5 "
          f"FAIL)")
    print(f"  but the matched-control residual E_state stays "
          f"{'within' if es.max() <= thr else 'ABOVE'} the control bound")
    if es.max() <= thr:
        print("  -> the ratio>2 was SHARED DRIFT, not wall pollution")
        print("  -> §41D-2 gate 5 becomes PASS under the corrected discipline")
        print("  -> §41D-2: FULL PASS")
    else:
        print("  -> some of the excess is genuinely attributable to the state "
              "change; needs investigation")

    json.dump(dict(rows=rows, per_region=per,
                   e_state_mean=float(es.mean()), e_state_max=float(es.max()),
                   e_state_p90=float(np.percentile(es, 90)),
                   threshold=thr, ratio_eps=eps,
                   live_ratio_max=float(live_ratio_max),
                   full_pass=bool(es.max() <= thr)),
              open("output/control/control.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
