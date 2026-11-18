#!/usr/bin/env bash
set -u
cd "$HOME/ai/lingbot-world-v2"
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
~/ai/lingbot-env/bin/python -u d3_disentangle.py \
  --scene 04 --seed 42 --out_chunks "${1:-40}" --tail "${2:-10}" \
  --groups "${3:-D1,D2,D3O}" \
  > "/tmp/d3_${3:-core}.log" 2>&1
echo "EXIT=$?"
grep -E '^\[d3\]|^  ' "/tmp/d3_${3:-core}.log" | tail -60
