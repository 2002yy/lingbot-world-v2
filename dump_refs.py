import cv2
import numpy as np

for s in ["04", "01"]:
    im = cv2.imread(f"examples/{s}/image.jpg")
    print(s, im.shape)
    small = cv2.resize(im, (528, 288))
    cv2.imwrite(f"/mnt/c/Users/Zhang/Desktop/视频/lingbot_out/scene{s}_ref.png", small)
