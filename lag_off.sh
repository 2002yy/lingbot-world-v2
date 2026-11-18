#!/usr/bin/env bash
cd ~/ai/lingbot-world-v2 || exit 1
for D in output/motions output/motions_off2 output/motions_off4; do
  echo "--- $D ---"
  ~/ai/lingbot-env/bin/python pose_tracking_lag.py --clips "$D" 2>&1 | grep -E '04_s42|MEAN'
done
