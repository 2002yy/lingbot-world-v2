#!/usr/bin/env bash
cd ~/ai/lingbot-world-v2 || exit 1
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec ~/ai/lingbot-env/bin/python -u replicate_cv.py "$@"
