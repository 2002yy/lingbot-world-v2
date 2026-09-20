#!/usr/bin/env bash
# P2a post-fix revalidation: control first (must stay bit-identical), then gate.
# The frames_*.pt from the pre-fix run are contaminated (chunk 0 used prewarm's
# camera conditioning), so they are removed rather than reused.
set -e
cd /home/zhang/ai/lingbot-world-v2
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
PY=/home/zhang/ai/lingbot-env/bin/python

rm -f output/p2a_gate/21ch/frames_*.pt

echo "########## CONTROL: repro vs repro2 (post-fix) ##########"
$PY -u p2a_gate.py --lhs repro --rhs repro2 --chunks 21 --control \
    --out_dir output/p2a_gate/21ch_postfix_ctl

echo
echo "########## GATE: repro vs fast (post-fix) ##########"
$PY -u p2a_gate.py --lhs repro --rhs fast --chunks 21 \
    --out_dir output/p2a_gate/21ch_postfix
