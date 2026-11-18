#!/usr/bin/env python
import sys, numpy as np, cv2
mp4 = sys.argv[1]
cap = cv2.VideoCapture(mp4)
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
print(f'{mp4}: {len(frames)} frames, {frames[0].shape}')

orb = cv2.ORB_create(nfeatures=1500)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

def flow_metrics(f1, f2):
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    matches = bf.knnMatch(d1, d2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    inl = mask.ravel().astype(bool) if mask is not None else np.ones(len(p1), bool)
    d = p2 - p1
    dx, dy = d[:, 0][inl], d[:, 1][inl]
    if len(dx) < 5:
        return None
    med_dx, med_dy = np.median(dx), np.median(dy)
    cons = np.mean(np.sign(dx) == np.sign(med_dx))
    mag = np.hypot(dx, dy)
    return dict(n_good=len(good), inlier_ratio=float(inl.mean()),
                med_dx=float(med_dx), med_dy=float(med_dy),
                dir_cons=float(cons), med_mag=float(np.median(mag)),
                mag_std=float(np.std(mag)))

# key pairs: consecutive early, and 0 vs N
pairs = [(0, 1), (4, 5), (8, 9), (12, 13), (16, 17), (24, 25), (40, 41)]
pairs += [(0, 16), (0, 32), (0, 48), (0, len(frames) - 1)]
print(f'{"pair":>10} {"nGood":>6} {"inlR":>6} {"med_dx":>8} {"med_dy":>8} {"dirCons":>8} {"medMag":>8}')
for a, b in pairs:
    if b >= len(frames): continue
    m = flow_metrics(frames[a], frames[b])
    if m is None:
        print(f'{a:>4}->{b:<4}  (no match)'); continue
    print(f'{a:>4}->{b:<4} {m["n_good"]:>6} {m["inlier_ratio"]:>6.2f} '
          f'{m["med_dx"]:>8.2f} {m["med_dy"]:>8.2f} {m["dir_cons"]:>8.2f} {m["med_mag"]:>8.2f}')
