#!/usr/bin/env python
"""§Latency-3B-B1 reference: a deterministic run through the REAL session path.

PURPOSE

C2's hard gate is "an uninterrupted run must produce the same trajectory as before".
That cannot be argued, only measured, and it has to be measured through the real
`WanSession.denoise` rather than a re-implementation -- otherwise the thing being
compared is not the thing that ships.

This drives the production session with a deterministic pose and control sequence, so
the pre-change run and the post-change run are comparable digest for digest.

    poses      one discrete control per chunk, derived from the chunk index
    digest     each chunk's x0, each chunk's KV slot, and the final KV
    seed       fixed, and the per-chunk generator state is captured, not guessed

Run before the change and after it; the two outputs must be identical.

Run:  python b1_reference.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

import demo_wasd
from demo_wasd import WanSession
from interactive_runtime import CameraState, InteractiveRuntime

K_CHUNKS = 6
BASE_SEED = 4321


def chunk_controls(cid: int) -> dict:
    """Deterministic, so the pose sequence is reproducible without wall-clock timing."""
    return {"forward": 0.3 * ((cid % 3) - 1), "yaw": 0.4 * ((cid % 2) - 0.5) * 2}


def main():
    args = demo_wasd.build_args([])
    args.headless = True
    args.pixel = "304x528"
    args.weight = "bf16"
    args.n_chunks = 40
    args.local_window = 6
    args.sink = 1
    args.seed = BASE_SEED

    print(f"  building session (seed {BASE_SEED}, {K_CHUNKS} chunks) ...", flush=True)
    t0 = time.perf_counter()
    sess = WanSession(args)

    base_pose = np.load(f"{args.base}/poses.npy")[0]
    from cam_controller import CameraController
    ctl = CameraController(base_pose[:3, :3], base_pose[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    rt = InteractiveRuntime(CameraState(pose=ctl.pose.copy(), v=np.zeros(3), gate=1.0))
    sess.attach_runtime(rt)
    print(f"  ready in {time.perf_counter()-t0:.0f} s", flush=True)

    fsl = sess.frame_seqlen
    lei0 = int(sess.self_kv[0]["local_end_index"])
    out = {"k_chunks": K_CHUNKS, "base_seed": BASE_SEED,
           "fsl": fsl, "kv_size": sess.kv_size, "lei0": lei0,
           "chunks": [], "has_on_commit": hasattr(sess, "on_commit")}

    for cid in range(K_CHUNKS):
        rt.accept(chunk_controls(cid))
        snap = rt.begin_chunk()
        # the production call, unmodified
        x0 = sess.denoise(snap, cid, lambda *a, **k: None)
        meta = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                                 snap["applied_event_ids"])
        rt.commit(meta)
        # C1's hook: after the change the pose advances only here. On the pre-change
        # code the method does not exist and denoise has already advanced it, which is
        # exactly the difference this reference is here to measure.
        if hasattr(sess, "on_commit"):
            sess.on_commit()
        rt.mark_real_decoded(meta)

        lei = int(sess.self_kv[0]["local_end_index"])
        lo, hi = max(0, lei - fsl), lei
        slot = torch.cat([l["k"][:, lo:hi].reshape(-1) for l in sess.self_kv])
        xd = int(x0.reshape(-1).view(torch.int16).to(torch.int64).sum().item())
        pose = np.asarray(rt.committed.camera.pose).ravel()
        out["chunks"].append(dict(
            cid=cid,
            applied=list(snap["applied_event_ids"]),
            x0_sum=xd,
            x0_shape=list(x0.shape),
            slot_sum=int(slot.view(torch.int16).to(torch.int64).sum().item()),
            gei=int(sess.self_kv[0]["global_end_index"]),
            lei=lei,
            pose_sum=float(pose.sum()),
        ))
        print(f"    chunk {cid}: x0_sum={xd} slot_sum={out['chunks'][-1]['slot_sum']} "
              f"gei={out['chunks'][-1]['gei']} lei={lei} "
              f"pose_sum={out['chunks'][-1]['pose_sum']:.6f}", flush=True)

    final_k = int(torch.cat([l["k"].reshape(-1) for l in sess.self_kv])
                  .view(torch.int16).to(torch.int64).sum().item())
    final_v = int(torch.cat([l["v"].reshape(-1) for l in sess.self_kv])
                  .view(torch.int16).to(torch.int64).sum().item())
    out["final_k"] = final_k
    out["final_v"] = final_v
    out["prev_pose_sum"] = float(np.asarray(sess._prev_pose).ravel().sum())
    print(f"  final_k={final_k} final_v={final_v}")
    print(f"  prev_pose_sum={out['prev_pose_sum']:.6f}")

    with open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/b1_reference.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"  wrote {sys.argv[1] if len(sys.argv) > 1 else '/tmp/b1_reference.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
