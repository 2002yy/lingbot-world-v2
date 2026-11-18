#!/usr/bin/env bash
set -u
cd "$HOME/ai/lingbot-world-v2"
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
~/ai/lingbot-env/bin/python -u state_commit2.py \
  --scene 04 --seed 42 --mode sweep --path B \
  --horizon 8 --alphas 1,2 --ks 2,4 \
  > /tmp/c2B.log 2>&1
echo "EXIT=$?"
grep -E '^\[c2\]|^  ' /tmp/c2B.log | tail -60
