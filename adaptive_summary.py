#!/usr/bin/env python
import csv
import numpy as np

ev = list(csv.DictReader(open('output/adaptive/refresh.csv')))
fr = list(csv.DictReader(open('output/adaptive/frames.csv')))

print(f"{'clip':10s} {'mode':9s} {'n':>3s} {'mean_jump':>10s} {'p90':>7s} {'max':>7s}")
for tag in sorted({e['clip'] for e in ev}):
    for mode in ('fixed', 'adaptive'):
        e = [x for x in ev if x['clip'] == tag and x['mode'] == mode]
        j = np.array([float(x['jump']) for x in e])
        print(f"{tag:10s} {mode:9s} {len(e):3d} {j.mean():10.4f} "
              f"{np.percentile(j,90):7.4f} {j.max():7.4f}")

print()
tot = {}
for mode in ('fixed', 'adaptive'):
    j = np.array([float(x['jump']) for x in ev if x['mode'] == mode])
    tot[mode] = j
    print(f"{mode:9s} ALL n={len(j)} mean={j.mean():.4f} "
          f"p90={np.percentile(j,90):.4f} max={j.max():.4f}")
f, a = tot['fixed'], tot['adaptive']
print(f"improvement: mean {100*(1-a.mean()/f.mean()):.1f}%  "
      f"p90 {100*(1-np.percentile(a,90)/np.percentile(f,90)):.1f}%  "
      f"max {100*(1-a.max()/f.max()):.1f}%")

print()
print("state time share / mean gain (adaptive):")
for tag in sorted({x['clip'] for x in fr}):
    r = [x for x in fr if x['clip'] == tag and x['mode'] == 'adaptive']
    n = len(r)
    sh = {s: sum(1 for x in r if x['state'] == s) / n for s in ('ok','soft','reset','fail')}
    g = np.mean([float(x['gain']) for x in r])
    print(f"{tag:10s} ok={sh['ok']:.2f} soft={sh['soft']:.2f} "
          f"reset={sh['reset']:.2f} fail={sh['fail']:.2f} mean_gain={g:.3f}")
