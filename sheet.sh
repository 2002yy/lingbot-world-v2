#!/usr/bin/env bash
set -e
cd ~/ai/lingbot-world-v2
OUT=/mnt/c/Users/Zhang/Desktop/视频/lingbot_out/ctl
mkdir -p "$OUT"
V="${1:-output/ctl/walk_turn.mp4}"
N="${2:-walk_turn}"
ffmpeg -y -loglevel error -i "$V" -vf "select='eq(n\,0)+eq(n\,12)+eq(n\,24)+eq(n\,36)+eq(n\,48)+eq(n\,60)+eq(n\,72)',scale=264:144,tile=4x2" -frames:v 1 "$OUT/${N}_sheet.png"
ls -la "$OUT/"
