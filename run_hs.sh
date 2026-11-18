#!/usr/bin/env bash
cd ~/ai/lingbot-world-v2 || exit 1
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for M in A B C D; do
  ~/ai/lingbot-env/bin/python -u hotswap_loop.py --mode $M > /tmp/hs$M.log 2>&1
  echo "MODE $M EXIT=$?"
done
for M in A B C D; do
  echo "===== $M ====="
  sed -n '/causal admission/,$p' /tmp/hs$M.log | head -14
done
