#!/usr/bin/env bash
set -e
cd ~/ai/lingbot-world-v2
OUT=/mnt/c/Users/Zhang/Desktop/视频/lingbot_out/interactive
mkdir -p "$OUT"
ffmpeg -y -loglevel error -i output/interactive/interactive_display.mp4 \
  -vf "select='eq(n\,0)+eq(n\,1)+eq(n\,2)+eq(n\,3)+eq(n\,40)+eq(n\,41)+eq(n\,42)+eq(n\,43)',scale=264:144,tile=4x2" \
  -frames:v 1 "$OUT/interp_check.png"
ls "$OUT"
