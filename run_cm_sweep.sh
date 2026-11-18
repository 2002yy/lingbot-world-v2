#!/usr/bin/env bash
# §40C-1 dose-response sweep: how many consecutive commits are needed before
# the model sustains OPEN on its own?
set -u
cd "$HOME/ai/lingbot-world-v2"
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
for K in 1 3 6; do
  echo "##### K=$K #####"
  ~/ai/lingbot-env/bin/python -u state_commit.py \
    --scene 04 --seed 42 --n_chunks 8 --commit_until "$K" \
    > "/tmp/cmK${K}.log" 2>&1
  grep -E '^ +[0-9] |compliance|assimilate|persistence|latent delta|setup:' "/tmp/cmK${K}.log"
done
