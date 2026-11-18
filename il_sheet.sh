#!/usr/bin/env bash
set -e
cd ~/ai/lingbot-world-v2
OUT=/mnt/c/Users/Zhang/Desktop/视频/lingbot_out/interactive
mkdir -p "$OUT"
ffmpeg -y -loglevel error -i output/interactive/interactive.mp4 \
  -vf "select='eq(n\,0)+eq(n\,16)+eq(n\,36)+eq(n\,60)+eq(n\,80)+eq(n\,96)',scale=264:144,tile=3x2" \
  -frames:v 1 "$OUT/interactive.png"
ls "$OUT"
