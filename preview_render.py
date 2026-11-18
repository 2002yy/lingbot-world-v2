#!/usr/bin/env python
"""Render preview-vs-full comparison for the trained preview head."""
import os
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from preview_head import P1

data = torch.load("output/preview/data.pt")
net = P1()
net.load_state_dict(torch.load("output/preview/P1.pt", map_location="cpu"))
net.eval()

tiles = []
with torch.no_grad():
    for f, r in data[:3]:
        if f.dim() == 4:
            f = f.unsqueeze(0)
        if r.dim() == 4:
            r = r.unsqueeze(0)
        n, c, t, h, w = f.shape
        ff = f.permute(0, 2, 1, 3, 4).reshape(n * t, c, h, w).float()
        rr = r.permute(0, 2, 1, 3, 4).reshape(n * t, 3, r.shape[-2], r.shape[-1]).float()
        ff = F.interpolate(ff, size=(144, 264), mode="bilinear", align_corners=False)
        rr = F.interpolate(rr, size=(288, 528), mode="bilinear", align_corners=False)
        pv = net(ff)
        for k in range(min(2, pv.shape[0])):
            p = ((pv[k].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            g = ((rr[k].permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            p = cv2.resize(cv2.cvtColor(p, cv2.COLOR_RGB2BGR), (264, 144))
            g = cv2.resize(cv2.cvtColor(g, cv2.COLOR_RGB2BGR), (264, 144))
            tiles.append(np.concatenate([p, g], axis=1))

tile = np.concatenate(tiles, axis=0)
out = "/mnt/c/Users/Zhang/Desktop/视频/lingbot_out/preview_vs_full.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
cv2.imwrite(out, tile)
print("wrote", out, tile.shape)
