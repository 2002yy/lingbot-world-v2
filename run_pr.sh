#!/usr/bin/env bash
cd ~/ai/lingbot-world-v2 || exit 1
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f output/predictive/*.mp4
for K in steady changepoint; do
  ~/ai/lingbot-env/bin/python -u run_predictive.py --kind "$K" --scene 04 > /tmp/pr_$K.log 2>&1
  echo "$K EXIT=$?"
done
echo "===== steady ====="
grep -A7 '=====' /tmp/pr_steady.log
echo "===== changepoint ====="
grep -A7 '=====' /tmp/pr_changepoint.log
