#!/usr/bin/env python
"""Release smoke test. Startup, the runtime contract, and optionally the GPU path.

Two levels:

    release_check.py            fast, no GPU: imports, the runtime state machine, the
                                preview trace schema, the trace export round trip
    release_check.py --smoke    also runs the real pipeline for a few chunks and
                                checks that a control event reaches an authoritative
                                frame through the frozen production path

This is a SMOKE test, not a benchmark. It asks whether the shipped path works, not
how fast it is.
"""
import argparse
import json
import os
import sys
import tempfile

OK, FAIL = "PASS", "FAIL"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((OK, name, detail or ""))
        print(f"  {OK}  {name:<44} {detail or ''}")
        return True
    except Exception as e:
        results.append((FAIL, name, f"{type(e).__name__}: {e}"))
        print(f"  {FAIL}  {name:<44} {type(e).__name__}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="also run the real GPU pipeline for a few chunks")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--base", default="examples/04")
    args = ap.parse_args()

    print("=" * 84)
    print(f"  Release smoke test   smoke={args.smoke}")
    print("=" * 84)

    # ------------------------------------------------------------ imports
    def _imports():
        import control_reduce, interactive_runtime, preview_trace  # noqa
        return "control_reduce, interactive_runtime, preview_trace"
    check("core modules import", _imports)

    def _wan():
        import wan  # noqa
        from wan.configs import WAN_CONFIGS
        assert "i2v-1.3B" in WAN_CONFIGS
        return f"wan ok, configs {len(WAN_CONFIGS)}"
    check("model package imports", _wan)

    # ------------------------------------------------- runtime contract
    from control_reduce import reduce_controls
    from interactive_runtime import (
        ABORTED, COMMITTED, STALE, CameraState, InteractiveRuntime,
        RuntimeStateError,
    )
    import numpy as np
    import torch

    def _runtime():
        rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
        ev = rt.accept({"forward": 1.0})
        snap = rt.begin_chunk()
        assert ev.event_id in snap["applied_event_ids"], "event not assigned"
        m = rt.new_frame_meta("real", snap["chunk_index"],
                              snap["generation_id"], snap["applied_event_ids"])
        rt.commit(m)
        rt.mark_real_decoded(m)
        r = rt.record(ev.event_id)
        assert r.terminal_status == COMMITTED
        assert r.t2_commit_ns is not None and r.t3_first_real_ns is not None
        assert r.t4_renderer_submit_ns is None and r.t5_present_ns is None, \
            "t4/t5 must stay None: there is no renderer and no present signal"
        return f"event {ev.event_id}, commit and first-real recorded, t4/t5 None"
    check("authoritative runtime: accept -> assign -> commit -> t3", _runtime)

    def _exactly_once():
        rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
        ev = rt.accept({"forward": 1.0})
        s1 = rt.begin_chunk()
        c1 = s1["candidate_camera"].copy()
        rt.fail_chunk("smoke")
        assert rt._inflight["candidate_camera"].pose.tobytes() == \
            c1.pose.tobytes(), "retry changed the candidate"
        m = rt.new_frame_meta("real", s1["chunk_index"], s1["generation_id"],
                              s1["applied_event_ids"])
        rt.commit(m)
        return "retry reuses the materialised candidate"
    check("exactly-once: a retry does not re-apply", _exactly_once)

    def _stale():
        rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
        ev = rt.accept({"forward": 1.0})
        rt.committed.chunk_index = 5          # frontier moves past the claim
        rt.begin_chunk()
        assert rt.record(ev.event_id).terminal_status == STALE
        return "an expired claim fails closed"
    check("stale detection: expired application claim", _stale)

    def _prewarm_isolated():
        rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
        before = rt.authoritative_fingerprint()
        with rt.prewarm_scope() as pw:
            pw.prewarm_frame_meta("real")
        assert rt.authoritative_fingerprint() == before
        return "prewarm changes no authoritative state"
    check("prewarm isolation", _prewarm_isolated)

    # ------------------------------------------------- trace export
    def _export():
        rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
        rt.accept({"forward": 1.0}, _now_ns=1000)
        s = rt.begin_chunk(_now_ns=2000)
        m = rt.new_frame_meta("real", s["chunk_index"], s["generation_id"],
                              s["applied_event_ids"])
        rt.commit(m, _now_ns=3000)
        rt.mark_real_decoded(m, _now_ns=4000)
        blob = json.dumps(rt.export())
        assert "_ms" not in blob, "a derived value leaked into the export"
        back = InteractiveRuntime.import_records(json.loads(blob))
        assert back[0].derived() == rt.records()[0].derived()
        return "raw-only export round-trips with identical derived()"
    check("trace export is valid and raw-only", _export)

    # ------------------------------------------------- preview schema
    def _preview():
        from preview_trace import PreviewTrace, assert_non_authoritative
        rt = InteractiveRuntime(CameraState(pose=np.eye(4), v=np.zeros(3)))
        pv = PreviewTrace(chunk_index=0, generation_id=0, applied_event_ids=(1,),
                          source_step=0, accept_ns=0, preview_decoded_ns=10)
        assert pv.ready_ms() is not None
        m = rt.new_frame_meta("preview", 0, 0, ())
        assert_non_authoritative(rt, m)
        return "preview is non-authoritative on both seams"
    check("preview cannot commit or mark t3", _preview)

    # ------------------------------------------------- production stack
    def _authority():
        """The shipped defaults must be the frozen ones."""
        import re
        src = open("play.py", encoding="utf-8").read()
        assert 'default="variant_d"' in src, "play.py default preview is not variant_d"
        assert "--blend_ms" in src and "default=50" in src, \
            "play.py default blend is not 50 ms"
        run = open("run.sh", encoding="utf-8").read()
        assert "304x528" in run and "variant_d" in run, \
            "run.sh does not pin the 304x528 variant_d stack"
        return "play.py and run.sh both default to the frozen stack"
    check("production stack is the shipped default", _authority)

    # ------------------------------------------------- assets
    def _assets():
        miss = [p for p in (args.base, f"{args.base}/image.jpg",
                            f"{args.base}/poses.npy",
                            f"{args.base}/intrinsics.npy")
                if not os.path.exists(p)]
        assert not miss, f"missing: {miss}"
        return args.base
    check("example scene present", _assets)

    # ------------------------------------------------- optional GPU smoke
    if args.smoke:
        def _gpu():
            import subprocess
            out = os.path.join(tempfile.gettempdir(), "lingbot_release_smoke")
            r = subprocess.run(
                [sys.executable, "play.py", "--weight", "bf16",
                 "--pixel", "304x528", "--n_chunks", str(args.chunks),
                 "--preview", "variant_d", "--blend_ms", "50",
                 "--out_dir", out],
                capture_output=True, text=True)
            assert r.returncode == 0, f"play.py failed: {r.stderr[-400:]}"
            j = json.load(open(os.path.join(out, "play_traces.json")))
            tr = [t for t in j["traces"]]
            # NOTE: the exported record fields are raw timestamps, so the names are
            # t1_assign_ns / t2_commit_ns / t3_first_real_ns. An earlier version of
            # this check looked for "t1_assigned" and therefore always failed.
            assigned = [t for t in tr if t.get("t1_assign_ns")]
            assert assigned, "no event was ever assigned to a chunk"
            with_real = [t for t in assigned if t.get("t3_first_real_ns")]
            assert with_real, "no event reached an authoritative frame"
            assert j.get("preview_traces") is not None, "no preview trace emitted"
            assert j.get("handoff") is not None, "no handoff row recorded"
            pv = [p for p in j["preview_traces"] if p.get("preview_decode_ms")]
            assert pv, "no preview was decoded"
            return (f"{len(with_real)} events reached a real frame; "
                    f"{len(pv)} previews; handoff rows {len(j['handoff'])}")
        check("GPU end-to-end: control -> authoritative frame", _gpu)

    print()
    bad = [r for r in results if r[0] == FAIL]
    print(f"  {len(results) - len(bad)}/{len(results)} checks passed")
    if bad:
        print("  FAILED:")
        for _, n, d in bad:
            print(f"    {n}: {d}")
        return 1
    print("  release smoke test: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
