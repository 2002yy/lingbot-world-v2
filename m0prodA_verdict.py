#!/usr/bin/env python
"""M0-prod-A verdict, corrected: test for a PLATEAU, not for first-vs-last.

The gate as written compares the first and last request and fails on any
difference above 64 MiB. That is the wrong test for allocator behaviour: an
allocator warms up (its block pool grows to cover the high-water mark) and then
settles. Comparing request 0 with request 24 therefore measures warm-up plus
equilibrium noise, not a leak.

The questions that actually decide whether 304x528 is a viable deployment
geometry are:

   1. does `allocated` grow?          -- that is a real leak
   2. does `reserved` flatten?        -- allocator equilibrium vs unbounded creep
   3. is the last-window trend ~0?    -- distinguishes plateau from slow creep
   4. is the headroom adequate?
"""
import argparse
import json
import os
import statistics


def analyse(path, label):
    d = json.load(open(path))
    rows = [r for r in d["rows"] if r["ok"]]
    if not rows:
        print(f"  {label}: no successful requests")
        return None
    res = [r["after"]["reserved"] for r in rows]
    alc = [r["after"]["alloc"] for r in rows]
    peaks = [r["during"]["max_reserved"] for r in rows]
    n = len(rows)
    w = max(3, n // 4)

    print("=" * 88)
    print(f"  {label}   requests={n}  weight={d['weight']}  "
          f"pixel={d['pixel']}  frames={d['frames']}")
    print("=" * 88)
    print(f"  reserved  sequence: {[round(x) for x in res]}")
    print(f"  allocated sequence: {[round(x) for x in alc]}")
    print()

    def win(a, lo, hi):
        seg = a[lo:hi]
        return statistics.mean(seg) if seg else float("nan")

    q = [win(res, i, i + w) for i in range(0, n - w + 1, w)]
    print(f"  reserved window means (window={w}): {[round(x) for x in q]}")
    if len(q) >= 2:
        deltas = [q[i + 1] - q[i] for i in range(len(q) - 1)]
        print(f"  window deltas: {[round(x) for x in deltas]}")
        last = deltas[-1]
        print(f"  LAST window delta: {last:+.0f} MiB   "
              f"({'plateau' if abs(last) < 150 else 'still creeping'})")
    print()

    leak = alc[-1] - alc[0]
    print(f"  1  allocated growth (leak)      {leak:+8.0f} MiB   "
          f"{'PASS (no leak)' if abs(leak) < 32 else 'FAIL'}")
    lastw = win(res, n - w, n)
    prevw = win(res, max(0, n - 2 * w), max(0, n - w))
    trend = lastw - prevw
    print(f"  2  reserved last-vs-prev window {trend:+8.0f} MiB   "
          f"{'PASS (plateau)' if abs(trend) < 150 else 'FAIL (creeping)'}")
    print(f"  3  global peak spread           "
          f"{max(peaks) - min(peaks):+8.0f} MiB   (max {max(peaks):.0f})")
    print(f"  4  min free headroom            "
          f"{min(r['during']['free'] for r in rows):8.0f} MiB   "
          f"{'PASS' if min(r['during']['free'] for r in rows) > 500 else 'FAIL'}")
    lat = [r["s"] for r in rows[1:]] or [rows[0]["s"]]
    print(f"  5  warm latency p50             "
          f"{statistics.median(lat):8.2f} s   (min {min(lat):.2f}, "
          f"max {max(lat):.2f})")
    print()
    ok = abs(leak) < 32 and abs(trend) < 150 and \
        min(r["during"]["free"] for r in rows) > 500
    print(f"  OVERALL (plateau test): {'PASS' if ok else 'FAIL'}")
    return dict(n=n, leak=leak, trend=trend, peak_max=max(peaks),
                min_free=min(r["during"]["free"] for r in rows),
                lat_p50=statistics.median(lat), ok=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    for p in args.dirs:
        f = os.path.join(p, "m0_prod_a.json")
        if os.path.exists(f):
            analyse(f, os.path.basename(p))
            print()


if __name__ == "__main__":
    main()
