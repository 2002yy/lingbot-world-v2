#!/usr/bin/env bash
cd ~/ai/lingbot-world-v2 || exit 1
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for OFF in 2 4; do
  LINGBOT_POSE_OFFSET=$OFF ~/ai/lingbot-env/bin/python -u run_motions.py \
    --scenes 04 --seeds 42 --out_dir output/motions_off$OFF > /tmp/mo_off$OFF.log 2>&1
  echo "OFF=$OFF EXIT=$?"
done
