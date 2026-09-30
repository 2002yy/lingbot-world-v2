# Latency-1A: input-to-present contract, and a t0-t5 reachability audit

## The contract, frozen before any implementation

    t0 = input accepted        the event is accepted by the authoritative input queue
    t1 = assigned              the event is bound to a specific tick / chunk
    t2 = commit                the generation state containing that event becomes
                               committed authoritative state
    t3 = first_real_decoded    the first real model frame *certainly affected* by
                               that event finishes decoding
    t4 = renderer_submit       that frame is submitted to the actual display /
                               renderer pipeline
    t5 = presented             that frame is actually presented and the user has
                               a chance to see it

Derived, also frozen:

    input_to_assign         = t1 - t0
    input_to_commit         = t2 - t0
    commit_to_real          = t3 - t2
    input_to_real           = t3 - t0
    real_to_submit          = t4 - t3
    submit_to_present       = t5 - t4
    control_to_real_display = t5 - t0     <- only this may be called
                                             measured input-to-display

The existing decode-return, cadence-formula and configuration-value figures stay
demoted.

## Audit result: 6 of 6 are GAPs, two of them architecturally unrepresentable

| pt | semantic | current code point | real timestamp | lineage | status |
|----|----------|--------------------|----------------|---------|--------|
| t0 | queue accept | `cam_controller.set_input()` | no | no event_id | GAP |
| t1 | tick assignment | none (slot_pose picks by time, not by event) | no | no | GAP |
| t2 | state commit | **no commit seam exists** | no | no | GAP (no seam) |
| t3 | affected real frame | hotswap_loop's `t_vis` (decode return) | timing exists but wrong contract | no chunk/frame lineage | GAP (no lineage) |
| t4 | renderer submit | **no renderer exists** | no | no | GAP (architectural) |
| t5 | physical present | **no present signal exists** | no | no | GAP (architectural) |

## Evidence (static code, no GPU)

1. Tree-wide search for `imshow | cv2.namedWindow | pygame | SDL | QApplication |
   glfw | swapBuffers` returns **zero hits**. Results are currently surfaced only by
   `save_video` into an .mp4.
2. Search for `def commit | committed | commit(` finds only prose in
   `state_commit2.py` and `m1_stream_encode.py`, no actual seam.
3. Search for `event_id | applied_event | event_queue` returns zero hits; the word
   "admission" appears only in prose in `kv_admission_probe.py` and
   `hotswap_loop.py`.
4. Search for `frame_kind | frame_id | chunk_index` returns zero hits.
5. The input entry is `cam_controller.py:74 set_input(...)`, which only writes
   `self.sm`: no timestamp, no event id, no queue.

## Core conclusion

This is not a missing-timer problem, it is four missing structures:

    A. no event identity      -> t0/t1 cannot have lineage
    B. no commit seam         -> t2 is unrepresentable
    C. no renderer / present  -> t4/t5 are ARCHITECTURALLY unrepresentable
    D. no frame lineage       -> t3 cannot be shown to be affected by the event

So in the current architecture **no measured input-to-display can be produced at
all**. Anything reported today can only be a decode-complete proxy -- exactly the
class of figure the G4-0 audit already demoted.

## Disposition for each GAP

    t0  queue accept     -> create explicit seam (needs an authoritative input queue)
    t1  tick assignment  -> create explicit seam (events need ids, explicitly bound
                            to a chunk)
    t2  state commit     -> create explicit seam (commit(chunk_index,
                            applied_event_ids))
    t3  affected real    -> instrument the existing decode point AND add frame
                            lineage; the point exists but is meaningless without
                            chunk/frame/event lineage
    t4  renderer submit  -> DECLARE NOT MEASURABLE in the current architecture
                            (no renderer to instrument)
    t5  physical present -> DECLARE NOT MEASURABLE in the current architecture
                            (no present signal; and even with a renderer, a
                             Python/WSL path may not receive a true present
                             acknowledgement)

Forbidden: picking the nearest convenient timestamp.

## Consequence for the roadmap

The audit is itself the design input for §Interactive-1: there is currently **no
single runtime authority**. Existing harnesses are independent, with no event
queue, no commit and no renderer. So:

    Latency-1A  contract + reachability          <- done this round
    Interactive-1  authoritative runtime (event queue + commit seam + frame lineage)
    Latency-1B  unified instrumentation (LatencyTrace)
    real input->present baseline (if t5 stays unavailable, honestly report
                                  input->submit)

## Latency-1B instrumentation spec, frozen now, implemented later

    LatencyTrace(
        event_id, assigned_chunk, generation_id, first_real_frame_id,
        t0_input, t1_assigned, t2_committed, t3_real_decoded,
        t4_submitted, t5_presented,
    )

Constraints: all timestamps `perf_counter_ns()` in one clock domain; event and
frame lineage mechanically verifiable; missing fields stay None; **never infer a
missing timestamp**; one unified trace record rather than six timers scattered
across four harnesses. That last point alone would eliminate hotswap_loop's
"relative elapsed minus script timestamp" time-base mixing.

## Scope of the first round

Performance preset only (bf16, 304x528, stream encode ON), the lowest-latency main
interaction tier. Get contract correctness, then lineage correctness, then timing
correctness. FP8_lowmem is added later as a second arm without redefining the
contract.
