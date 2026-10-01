#!/usr/bin/env python
"""Inspect the scheduler's timestep space so the step subsets are chosen from facts."""
import os
import sys

import torch

import wan
from wan.configs import WAN_CONFIGS

os.environ["LINGBOT_MODE"] = "repro"
os.environ.setdefault("LINGBOT_WEIGHT_MODE", "bf16")
os.environ.setdefault("LINGBOT_FP8", "0")

cfg = WAN_CONFIGS["i2v-1.3B"]
pipe = wan.WanI2VCausal(
    config=cfg,
    checkpoint_dir=os.path.expanduser(
        "~/ai/models/lingbot-world-v2-1.3b-causal-fast"),
    device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False,
    t5_cpu=True, convert_model_dtype=False, local_attn_size=6, sink_size=1,
    infer_mode="causal_fast",
    assets_dir=os.path.expanduser("~/ai/models/lingbot-shared-assets"))

ts = pipe.scheduler.timesteps
print("=" * 72)
print("  scheduler timesteps")
print("=" * 72)
print(f"  type          {type(ts).__name__}")
print(f"  shape         {list(ts.shape)}")
print(f"  dtype         {ts.dtype}")
print(f"  min / max     {ts.min().item():.1f} / {ts.max().item():.1f}")
print(f"  first 12      {[round(float(x),1) for x in ts[:12]]}")
print(f"  last 12       {[round(float(x),1) for x in ts[-12:]]}")
print()
cur = [0, 250, 750]
print(f"  production uses indices {cur} -> "
      f"{[round(float(ts[i]),1) for i in cur]}")
print()
print("  candidate subsets:")
for name, idx in (("3-step (current)", [0, 250, 750]),
                  ("2-step A", [0, 750]),
                  ("2-step B", [0, 500]),
                  ("2-step C (even)", [0, len(ts) - 1]),
                  ("1-step", [0]),
                  ("1-step mid", [250])):
    if max(idx) < len(ts):
        print(f"    {name:<18} idx {idx:<16} t="
              f"{[round(float(ts[i]),1) for i in idx]}")
print()
print(f"  note: the FIRST element is the starting timestep, so every subset must")
print(f"        begin at index 0 to start from the same noise level.")
