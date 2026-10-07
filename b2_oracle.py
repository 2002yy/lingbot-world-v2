#!/usr/bin/env python
"""§Latency-3B-B2 P0 -- the Rebase Oracle.

THE ORACLE, AND WHY IT IS THE RIGHT ONE

    REFERENCE   both inputs are known BEFORE chunk N starts, so N is generated once,
                normally, carrying E_old + E_new

    PREEMPT     N starts carrying only E_old; E_new arrives at a forward boundary; the
                attempt is logically cancelled and rebased; N is then replayed from
                scratch carrying E_old + E_new

If the rebase is implemented correctly, the two are the SAME chunk. Comparing against
the ordinary no-preemption path would be the wrong oracle, because that path pushes
E_new into chunk N+1 -- a different world by construction, so equality there would
prove nothing.

WHAT MUST MATCH, AND ONE THING THAT MUST NOT

    identical   x0, the KV cache, all cache indices, _prev_pose, the generator state,
                applied_event_ids, chunk_index
    differs     generation_id, and it must: the rebase deliberately opens a new
                generation so that any artefact of the discarded attempt fails a
                generation check instead of silently matching

Run:  python b2_oracle.py
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

import demo_wasd
from demo_wasd import ChunkPreempted, InputMailbox, WanSession
from interactive_runtime import CameraState, InteractiveRuntime

E_OLD = {"forward": 0.6}
E_NEW = {"yaw": 0.8}
CUT_POINTS = (1, 2, 3)


class BoundaryMailbox(InputMailbox):
    """A mailbox that only reveals its pending input from the k-th peek onward.

    This models "E_new arrived between forward k-1 and forward k" without needing a
    hook inside the generation loop, and it keeps the real peek/take contract.
    """

    def __init__(self, reveal_at_peek: int):
        super().__init__()
        self.reveal_at = reveal_at_peek
        self.peeks = 0

    def peek(self):
        self.peeks += 1
        if self.peeks < self.reveal_at:
            return None
        return super().peek()


def digest_cache(kv):
    def d(t):
        i = t.reshape(-1).view(torch.int16).to(torch.int64)
        return (i.numel(), int(i.sum().item()))
    return ([d(l["k"]) for l in kv], [d(l["v"]) for l in kv],
            [(int(l["global_end_index"]), int(l["local_end_index"])) for l in kv])


def gen_sig(g):
    return tuple(int(x) for x in g.get_state())


def snapshot(sess, x0, snap, rt):
    return dict(
        x0_sum=int(x0.reshape(-1).view(torch.int16).to(torch.int64).sum().item()),
        cache=digest_cache(sess.self_kv),
        prev_pose=None if sess._prev_pose is None
        else float(np.asarray(sess._prev_pose).ravel().sum()),
        rng=gen_sig(sess._gen),
        chunk_index=snap["chunk_index"],
        applied=tuple(snap["applied_event_ids"]),
        committed_chunk_index=rt.committed.chunk_index,
        committed_applied=tuple(rt.committed.applied_event_ids),
    )


def main():
    args = demo_wasd.build_args([])
    args.headless = True
    args.pixel = "304x528"
    args.weight = "bf16"
    args.n_chunks = 40
    args.local_window = 6
    args.sink = 1
    args.seed = 777

    print("=" * 78)
    print("  §Latency-3B-B2 P0  Rebase Oracle")
    print("=" * 78, flush=True)
    t0 = time.perf_counter()
    sess = WanSession(args)
    from cam_controller import CameraController
    base_pose = np.load(f"{args.base}/poses.npy")[0]
    ctl = CameraController(base_pose[:3, :3], base_pose[:3, 3])
    ctl.cfg.yaw_rate_max, ctl.cfg.pitch_rate_max, ctl.cfg.v_max = 6.0, 2.0, 1.0
    rt = InteractiveRuntime(CameraState(pose=ctl.pose.copy(), v=np.zeros(3), gate=1.0))
    sess.attach_runtime(rt)
    print(f"  session ready in {time.perf_counter()-t0:.0f} s", flush=True)

    fail = []

    def check(name, cond, detail=""):
        if not cond:
            fail.append(name)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
              flush=True)

    # ----------------------------------------------------------- REFERENCE
    print()
    print("--- REFERENCE: both inputs known before chunk N starts ---", flush=True)
    sess._gen.manual_seed(args.seed)
    rng0 = gen_sig(sess._gen)
    a = rt.accept(dict(E_OLD))
    b = rt.accept(dict(E_NEW))
    snap = rt.begin_chunk()
    n = snap["chunk_index"]
    x0 = sess.denoise(snap, n, lambda *a_, **k: None)
    meta = rt.new_frame_meta("real", snap["chunk_index"], snap["generation_id"],
                             snap["applied_event_ids"])
    rt.commit(meta)
    sess.on_commit()
    ref = snapshot(sess, x0, snap, rt)
    ref_gen = snap["generation_id"]
    print(f"  reference chunk {n} applied={ref['applied']} gen={ref_gen} "
          f"x0_sum={ref['x0_sum']}", flush=True)

    # ------------------------------------------------------------- PREEMPT
    for cut in CUT_POINTS:
        print()
        print(f"--- PREEMPT: E_new arrives between forwards {cut-1} and {cut} ---",
              flush=True)
        # rebuild the exact starting state: same seed, same flush of the model state
        sess._gen.manual_seed(args.seed)
        rt2 = InteractiveRuntime(
            CameraState(pose=ctl.pose.copy(), v=np.zeros(3), gate=1.0))
        sess.attach_runtime(rt2)
        sess._prev_pose = None
        sess._candidate_pose = None
        for l in sess.self_kv:
            l["k"].zero_(); l["v"].zero_()
            l["global_end_index"] = 0
            l["local_end_index"] = 0
        torch.cuda.synchronize()

        box = BoundaryMailbox(reveal_at_peek=cut)
        box.post(E_NEW, 12345)
        ea = rt2.accept(dict(E_OLD))
        snap2 = rt2.begin_chunk()
        preempted = False
        try:
            x0b = sess.denoise(snap2, snap2["chunk_index"],
                               lambda *a_, **k: None, preempt=box, preempt_budget=1)
        except ChunkPreempted as px:
            preempted = True
            info = rt2.rebase_chunk(reason=f"oracle cut {cut}")
            t_obs, controls = px.request            # (t_observed_ns, controls)
            rt2.accept(controls, _now_ns=t_obs)     # cancel-before-accept: claim == N
            sess.restore_chunk_rng()
            snap2 = rt2.begin_chunk()
            x0b = sess.denoise(snap2, snap2["chunk_index"], lambda *a_, **k: None)
        meta2 = rt2.new_frame_meta("real", snap2["chunk_index"], snap2["generation_id"],
                                   snap2["applied_event_ids"])
        rt2.commit(meta2)
        sess.on_commit()
        pre = snapshot(sess, x0b, snap2, rt2)

        print(f"    preempted={preempted} chunk={pre['chunk_index']} "
              f"applied={pre['applied']} gen={snap2['generation_id']} "
              f"x0_sum={pre['x0_sum']}", flush=True)
        check(f"cut {cut}: a preemption actually happened", preempted)
        check(f"cut {cut}: the rebase bumped the generation (A -> B)",
              snap2["generation_id"] == ref_gen + 1,
              f"{ref_gen} -> {snap2['generation_id']}")
        check(f"cut {cut}: chunk_index identical", pre["chunk_index"] == ref["chunk_index"])
        check(f"cut {cut}: applied_event_ids identical (E_old AND E_new carried)",
              pre["applied"] == ref["applied"], f"{pre['applied']} vs {ref['applied']}")
        check(f"cut {cut}: x0 bit-identical", pre["x0_sum"] == ref["x0_sum"],
              f"{pre['x0_sum']} vs {ref['x0_sum']}")
        check(f"cut {cut}: KV cache identical (k, v and every index)",
              pre["cache"] == ref["cache"])
        check(f"cut {cut}: _prev_pose identical",
              pre["prev_pose"] == ref["prev_pose"],
              f"{pre['prev_pose']} vs {ref['prev_pose']}")
        check(f"cut {cut}: generator state identical at the end",
              pre["rng"] == ref["rng"])
        check(f"cut {cut}: CommittedChunk lineage identical",
              pre["committed_chunk_index"] == ref["committed_chunk_index"]
              and pre["committed_applied"] == ref["committed_applied"])

    print()
    print("=" * 78)
    print(f"  P0 Rebase Oracle: {'PASS' if not fail else 'FAIL (' + ', '.join(fail) + ')'}")
    print("=" * 78)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
