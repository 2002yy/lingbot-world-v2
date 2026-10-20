#!/usr/bin/env python
"""Two-layer eval for leave-and-return:
  Layer 1 (closure): ORB+RANSAC f0 vs return -> nGood, inlier, medReproj, residual flow.
  Layer 2 (memory): warp return->f0 by inv(H), then DINOv2 + LPIPS (whole + ROI).
Usage: python eval_two_layer.py <video> <return_idx> [roi=x0,y0,x1,y1]
"""
import sys, numpy as np, cv2, torch
from PIL import Image
import lpips
from transformers import AutoModel, AutoImageProcessor

_f = {}
def load_models():
    proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    dino = AutoModel.from_pretrained('facebook/dinov2-base').eval().cuda()
    lp = lpips.LPIPS(net='alex').eval().cuda()
    _f['proc'], _f['dino'], _f['lp'] = proc, dino, lp

@torch.no_grad()
def dino_feat(im):  # im: RGB uint8 HxWx3
    inp = _f['proc'](images=Image.fromarray(im), return_tensors='pt').to('cuda')
    out = _f['dino'](**inp).last_hidden_state  # [1, 1+N, 768]
    return torch.nn.functional.normalize(out.mean(dim=1), dim=-1)

@torch.no_grad()
def lpips_d(im1, im2):
    t = lambda a: torch.from_numpy(a).float().permute(2, 0, 1)[None].cuda() / 127.5 - 1.0
    return float(_f['lp'](t(im1), t(im2)).item())

def orb_closure(f0, f1):
    orb = cv2.ORB_create(2000)
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    k0, d0 = orb.detectAndCompute(g0, None); k1, d1 = orb.detectAndCompute(g1, None)
    if d0 is None or d1 is None: return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(d0, d1, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 8: return None
    p0 = np.float32([k0[m.queryIdx].pt for m in good])
    p1 = np.float32([k1[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    if H is None: return None
    inl = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - p1, axis=1)[inl]
    d = (p1 - p0)[inl]
    return dict(nGood=len(good), inlier=float(inl.mean()), H=H,
                medReproj=float(np.median(err)),
                medMag=float(np.median(np.hypot(d[:, 0], d[:, 1]))))

def crop(im, roi):
    h, w = im.shape[:2]
    x0, y0, x1, y1 = int(roi[0]*w), int(roi[1]*h), int(roi[2]*w), int(roi[3]*h)
    return im[y0:y1, x0:x1]

def main():
    vid = sys.argv[1]; ridx = int(sys.argv[2])
    roi = tuple(map(float, sys.argv[3].split('=')[1].split(','))) if len(sys.argv) > 3 else (0.15, 0.15, 0.85, 0.85)
    load_models()
    cap = cv2.VideoCapture(vid); frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    f0 = frames[0]; fr = frames[ridx]
    print(f'video={vid}  frames={len(frames)}  return_idx={ridx}  roi={roi}')
    c = orb_closure(f0, fr)
    if c is None:
        print('  LAYER1: CLOSURE FAIL (insufficient matches)'); return
    print(f'  LAYER1 closure: nGood={c["nGood"]} inlier={c["inlier"]:.2f} '
          f'medReproj={c["medReproj"]:.2f}px residualFlow={c["medMag"]:.2f}px')
    Hinv = np.linalg.inv(c['H'])
    h, w = f0.shape[:2]
    warped = cv2.warpPerspective(fr, Hinv, (w, h))
    r0 = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB); rw = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    dino_all = float((dino_feat(r0) @ dino_feat(rw).T).item())
    lp_all = lpips_d(r0, rw)
    dino_roi = float((dino_feat(crop(r0, roi)) @ dino_feat(crop(rw, roi)).T).item())
    lp_roi = lpips_d(crop(r0, roi), crop(rw, roi))
    print(f'  LAYER2 memory (return aligned -> start):')
    print(f'    DINO all={dino_all:.4f}  ROI={dino_roi:.4f}')
    print(f'    LPIPS all={lp_all:.4f}  ROI={lp_roi:.4f}')

if __name__ == '__main__':
    main()
