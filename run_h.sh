#!/usr/bin/env bash
cd ~/ai/lingbot-world-v2 || exit 1
export LINGBOT_FP8=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for H in 0.75 1.25; do
  ~/ai/lingbot-env/bin/python gen_predictive.py --kind steady --scene 04 \
    --horizon $H --modes B_cv,A_current > /tmp/gen_h$H.log 2>&1
  echo "GEN H=$H EXIT=$?"
  ~/ai/lingbot-env/bin/python -u run_predictive.py --kind steady --scene 04 \
    --horizon $H --modes B_cv --out_dir output/predictive > /tmp/pr_h$H.log 2>&1
  echo "RUN H=$H EXIT=$?"
done
~/ai/lingbot-env/bin/python eval_predictive.py --dir output/predictive --scene 04 2>&1 | tail -20
