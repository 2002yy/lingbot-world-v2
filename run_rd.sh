#!/usr/bin/env bash
set -u
cd "$HOME/ai/lingbot-world-v2"
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
~/ai/lingbot-env/bin/python -u object_rerender.py \
  --scene 04 --seed 42 --horizon "${1:-40}" --blends 0.5,1.0 \
  > "/tmp/rd_${1:-40}.log" 2>&1
echo "EXIT=$?"
grep -E '^\[rd\]|^  ' "/tmp/rd_${1:-40}.log" | tail -50
