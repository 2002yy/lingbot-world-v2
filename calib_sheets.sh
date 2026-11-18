#!/usr/bin/env bash
set -e
cd ~/ai/lingbot-world-v2
OUT=/mnt/c/Users/Zhang/Desktop/视频/lingbot_out/calib
mkdir -p "$OUT"
for n in yaw_l_med yaw_r_med pitch_u_large pitch_d_large strafe_l_large strafe_r_large fwd_med bwd_med; do
  ffmpeg -y -loglevel error -i "output/calib/$n.mp4" \
    -vf "select='eq(n\,0)+eq(n\,8)+eq(n\,16)+eq(n\,24)+eq(n\,32)+eq(n\,40)',scale=264:144,tile=3x2" \
    -frames:v 1 "$OUT/${n}.png"
done
ffmpeg -y -loglevel error -i output/calib/yaw_l_med.mp4 -vf "select='eq(n\,0)+eq(n\,8)+eq(n\,16)+eq(n\,24)+eq(n\,32)+eq(n\,40)',scale=264:144,tile=3x2" -frames:v 1 "$OUT/yaw_l_med.png"
ffmpeg -y -loglevel error -i output/calib/yaw_r_med.mp4 -vf "select='eq(n\,0)+eq(n\,8)+eq(n\,16)+eq(n\,24)+eq(n\,32)+eq(n\,40)',scale=264:144,tile=3x2" -frames:v 1 "$OUT/yaw_r_med.png"
ls "$OUT"
