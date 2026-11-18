#!/usr/bin/env bash
cd "$HOME/ai/lingbot-world-v2"
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
~/ai/lingbot-env/bin/python -u wall_destruction.py \
  --scene 04 --seed 42 --out_chunks 12 --tail 30 \
  > /tmp/de.log 2>&1
echo "EXIT=$?"
grep -E '^\[de\]|^  |^    ' /tmp/de.log | tail -60
