#!/usr/bin/env bash
cd ~/ai/lingbot-world-v2 || exit 1
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for P in B C; do
  if [ "$P" = "B" ]; then M=5; else M=10; fi
  ~/ai/lingbot-env/bin/python -u soak_loop.py --profile $P --minutes $M > /tmp/soak$P.log 2>&1
  echo "PROFILE $P EXIT=$?"
done
for P in A B C; do
  echo "===== PROFILE $P ====="
  awk '/§37 profile/,0' /tmp/soak$P.log
done
