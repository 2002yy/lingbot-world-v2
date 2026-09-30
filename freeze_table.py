#!/usr/bin/env python
"""8GB deployment preset freeze: produce the final two-tier table.

Reads the runs produced by _run_freeze.sh and reports, per weight mode:

    warm request p50 / p95, cold first request, output FPS
    peak max_reserved, min free, steady reserved plateau
    leak / creep over 25 requests (plateau test, not first-vs-last)
    81-frame result health, 249-frame viability, 501/777 prepare viability
"""
import argparse
import json
import os
import statistics


def load(d):
    f = os.path.join(d, "m0_prod_a.json")
    return json.load(open(f)) if os.path.exists(f) else None


def plateau_test(res, w):
    n = len(res)
    if n < 2 * w:
        return None
    last = statistics.mean(res[n - w:])
    prev = statistics.mean(res[n - 2 * w:n - w])
    return last - prev


def summarise(d, label):
    if d is None:
        return dict(label=label, ok=0, total=0)
    rows = d["rows"]
    good = [r for r in rows if r["ok"]]
    if not good:
        return dict(label=label, ok=0, total=len(rows),
                    err=rows[0].get("err") if rows else None)
    lat = [r["s"] for r in good]
    warm = lat[1:] or lat
    res = [r["after"]["reserved"] for r in good]
    alc = [r["after"]["alloc"] for r in good]
    peaks = [r["during"]["max_reserved"] for r in good]
    frees = [r["during"]["free"] for r in good]
    w = max(3, len(res) // 4)
    tr = plateau_test(res, w)
    frames = d["frames"]
    cs = d["chunk_size"]
    out_fps = frames / statistics.median(warm)
    return dict(label=label, weight=d["weight"], frames=frames,
                ok=len(good), total=len(rows),
                cold_s=lat[0], warm_p50=statistics.median(warm),
                warm_p95=sorted(warm)[int(len(warm) * 0.95) - 1],
                warm_min=min(warm), warm_max=max(warm),
                out_fps=out_fps,
                peak_max_reserved=max(peaks), min_free=min(frees),
                plateau_reserved=statistics.mean(res[-w:]),
                leak_alloc=alc[-1] - alc[0], creep_reserved=tr,
                y_hash=d.get("y_hash"), stream=d.get("stream_encode"),
                all_ok=len(good) == len(rows))


def viability(d, label):
    if d is None:
        return f"{label}: no data"
    rows = d["rows"]
    good = [r for r in rows if r["ok"]]
    first_ok = rows[0]["ok"] if rows else False
    err = rows[0].get("err") if rows and not rows[0]["ok"] else ""
    return (f"{label}: first-request {'OK' if first_ok else 'FAIL'} "
            f"({len(good)}/{len(rows)} ok)"
            + (f"  err={err[:60]}" if err else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/freeze")
    args = ap.parse_args()

    print("=" * 100)
    print("  8GB DEPLOYMENT PRESET FREEZE   (304x528, streamed encode = default)")
    print("=" * 100)

    main = {}
    for w in ("bf16", "fp8_lowmem"):
        d = load(os.path.join(args.root, f"{w}_25"))
        main[w] = summarise(d, w)

    fields = [("cold first request (s)", "cold_s", "{:.2f}"),
              ("warm request p50 (s)", "warm_p50", "{:.2f}"),
              ("warm request p95 (s)", "warm_p95", "{:.2f}"),
              ("warm min / max (s)", None, None),
              ("output FPS", "out_fps", "{:.3f}"),
              ("peak max_reserved (MiB)", "peak_max_reserved", "{:.0f}"),
              ("min free VRAM (MiB)", "min_free", "{:.0f}"),
              ("steady reserved plateau (MiB)", "plateau_reserved", "{:.0f}"),
              ("alloc leak over 25 req (MiB)", "leak_alloc", "{:+.0f}"),
              ("reserved last-vs-prev window (MiB)", "creep_reserved", "{:+.0f}"),
              ("requests ok", None, None)]

    print(f"\n  {'metric':<38} {'BF16 performance':>18} {'FP8 lowmem':>16}")
    print("  " + "-" * 76)
    for lab, key, fmt in fields:
        if key is None:
            if lab.startswith("warm min"):
                a = f"{main['bf16'].get('warm_min',0):.2f} / " \
                    f"{main['bf16'].get('warm_max',0):.2f}"
                b = f"{main['fp8_lowmem'].get('warm_min',0):.2f} / " \
                    f"{main['fp8_lowmem'].get('warm_max',0):.2f}"
            else:
                a = f"{main['bf16'].get('ok',0)}/{main['bf16'].get('total',0)}"
                b = f"{main['fp8_lowmem'].get('ok',0)}/" \
                    f"{main['fp8_lowmem'].get('total',0)}"
            print(f"  {lab:<38} {a:>18} {b:>16}")
            continue
        a = main["bf16"].get(key)
        b = main["fp8_lowmem"].get(key)
        fa = fmt.format(a) if isinstance(a, (int, float)) else "-"
        fb = fmt.format(b) if isinstance(b, (int, float)) else "-"
        print(f"  {lab:<38} {fa:>18} {fb:>16}")

    print()
    print("  condition y hash (must match across weights is NOT expected;")
    print("  weights differ). Within a weight it must be stable:")
    for w in ("bf16", "fp8_lowmem"):
        print(f"    {w:<12} y={main[w].get('y_hash')}  "
              f"stream_encode={main[w].get('stream')}")

    print()
    print("  DIFFERENTIAL (BF16 - FP8):")
    d50 = main["bf16"].get("warm_p50", 0) - main["fp8_lowmem"].get("warm_p50", 0)
    dpeak = main["bf16"].get("peak_max_reserved", 0) - \
        main["fp8_lowmem"].get("peak_max_reserved", 0)
    dfree = main["bf16"].get("min_free", 0) - main["fp8_lowmem"].get("min_free", 0)
    p50a = main["bf16"].get("warm_p50", 1)
    print(f"    warm p50      {d50:+.2f} s  ({d50/p50a*100:+.1f}%)")
    print(f"    peak VRAM     {dpeak:+.0f} MiB")
    print(f"    min free      {dfree:+.0f} MiB")

    print()
    print("=" * 100)
    print("  VIABILITY AT LONGER HORIZONS")
    print("=" * 100)
    for w in ("bf16", "fp8_lowmem"):
        for F in (249, 501, 777):
            d = load(os.path.join(args.root, f"{w}_F{F}"))
            print("  " + viability(d, f"{w:<12} F={F:<4}"))

    print()
    print("=" * 100)
    print("  FREEZE DECISION")
    print("=" * 100)
    ok_all = all(main[w].get("all_ok") for w in ("bf16", "fp8_lowmem"))
    ok_leak = all(abs(main[w].get("leak_alloc", 999)) < 32
                  for w in ("bf16", "fp8_lowmem"))
    # NOTE: `x or 999` was wrong here -- a creep of exactly 0 is falsy and became
    # 999, failing the criterion for the BEST possible result. Use an explicit
    # None check instead.
    def _creep(w):
        v = main[w].get("creep_reserved")
        return 999 if v is None else abs(v)
    ok_creep = all(_creep(w) < 150 for w in ("bf16", "fp8_lowmem"))
    print(f"  all 25 requests ok in both       {'PASS' if ok_all else 'FAIL'}")
    print(f"  no alloc leak in both            {'PASS' if ok_leak else 'FAIL'}")
    print(f"  reserved plateaued in both       {'PASS' if ok_creep else 'FAIL'}")
    print()
    if ok_all and ok_leak and ok_creep:
        print("  => presets FROZEN:")
        print("       performance : bf16       + streamed condition encode")
        print("       lowmem      : fp8_lowmem + streamed condition encode")
        print("     geometry      : 304x528 (8GB deployment authority)")
        print("     offload       : offload_model=True + in-tree device restore")
    else:
        print("  => NOT frozen; see the failing criteria above")

    with open(os.path.join(args.root, "freeze_table.json"), "w") as f:
        json.dump(dict(main=main, frozen=ok_all and ok_leak and ok_creep), f,
                  indent=2)
    print(f"\n  wrote {args.root}/freeze_table.json")


if __name__ == "__main__":
    main()
