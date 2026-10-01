#!/usr/bin/env python
"""Preview-History-Audit: is today's 36.7 ms decoder the same model as §35D/§36A?

Three questions:
  1. what is TAEHV's parameter count, and does it match §35D's 9.84M?
  2. what workload did §35D/§36A run at, versus today's?
  3. does the timing difference come from the model or from the workload?

No training, no new model. Just identity and workload.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.expanduser("~/ai/taehv"))
from taehv import TAEHV, StreamingTAEHV  # noqa: E402

MS = 1e6


def main():
    dev = "cuda"
    print("=" * 84)
    print("  Preview-History-Audit: model identity and workload")
    print("=" * 84)

    for name in ("taew2_1.pth", "taew2_2.pth", "taehv.pth"):
        p = os.path.expanduser(f"~/ai/taehv/{name}")
        if not os.path.exists(p):
            continue
        tae = TAEHV(checkpoint_path=p).to(dev).eval()
        n = sum(x.numel() for x in tae.parameters())
        nb = sum(x.numel() for x in tae.decoder.parameters())
        print(f"  {name:<16} total {n/1e6:7.3f} M   decoder {nb/1e6:7.3f} M   "
              f"file {os.path.getsize(p)/1e6:6.1f} MB")
        del tae
        torch.cuda.empty_cache()
    print()
    print("  §35D reported 9.84 M params at ~21 ms; §36A reported ~24 ms integrated.")
    print("  A match on the parameter count would mean the 'surrogate' and today's")
    print("  decoder are the same model, and the timing difference is workload.")
    print()

    tae = TAEHV(checkpoint_path=os.path.expanduser("~/ai/taehv/taew2_1.pth")) \
        .to(dev).eval()

    # ---- workload sweep: the same model, different latent sizes and frame counts
    print(f"  {'latent (C,T,H,W)':<22} {'out frames':>11} {'ms':>8}")
    print("  " + "-" * 44)
    for (c, t, hh, ww) in ((16, 1, 66, 38),     # today: 304x528
                           (16, 1, 60, 40),     # production_loop: 320x480
                           (16, 1, 36, 66),     # §35B/§35D: 528x288
                           (16, 3, 66, 38),     # a 3-latent chunk
                           (16, 1, 33, 19),     # half resolution
                           (16, 1, 132, 76)):   # double resolution
        z = torch.randn(c, t, hh, ww, device=dev)
        x = z.permute(1, 0, 2, 3).unsqueeze(0)
        with torch.no_grad():
            for _ in range(3):
                out = tae.decode_video(x, parallel=False, show_progress_bar=False)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(15):
            with torch.no_grad():
                out = tae.decode_video(x, parallel=False, show_progress_bar=False)
        e.record(); torch.cuda.synchronize()
        ms = s.elapsed_time(e) / 15
        nf = out.shape[1]
        print(f"  {str((c,t,hh,ww)):<22} {nf:>11} {ms:>8.2f}")
        del z, x, out
        torch.cuda.empty_cache()

    print()
    print("  StreamingTAEHV first-frame timing (the §36A integration path):")
    st = StreamingTAEHV(tae)
    z = torch.randn(16, 1, 66, 38, device=dev)
    try:
        st.start()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            fr = st.step(z)
        e.record(); torch.cuda.synchronize()
        print(f"    first frame {s.elapsed_time(e):.2f} ms")
    except Exception as ex:
        print(f"    streaming path unavailable: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
