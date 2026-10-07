# §Latency-1D — Renderer Submit Authority Contract

    STATUS   CONTRACT FROZEN. No timer has been added yet; this document exists so that
             the definitions are fixed BEFORE code, because a timer written first tends
             to turn "a convenient moment to read a clock" into the de-facto authority.

## 0. Scope

§Latency-1D defines and instruments the display-side boundary that follows authoritative
real-frame lineage. That lineage is already established by §Latency-1A/B/C and is not
revisited here.

It does **not** redefine input/event lineage, `processed_ack`, or commit semantics. It does
not invent a physical-present timestamp, does not treat headless output as displayed
output, and does not introduce warp or supersede semantics, which do not exist.

It answers exactly one question truthfully:

    When did an authoritative real frame first enter the real viewer/display pipeline?

and it keeps three words apart that are easy to collapse:

    decoded      the model output exists
    submitted    it has been handed to a display backend
    presented    it has actually reached the physical display

## 1. Timestamps entering this stage

    frozen already   t0 accepted   t1 assigned   t2 committed   t3 first affected real
                     frame decoded
    added here       t4 authoritative real frame first submitted to a real windowed path
                     t5 present completion

    capability       t3  MEASURABLE
                     t4  MEASURABLE, windowed only
                     t5  NOT YET MEASURABLE

`LatencyTraceRecord.t4_renderer_submit_ns` and `t5_present_ns` already exist; §Latency-1B
froze them to `None` and recorded that the runtime had no API writing them, so they could
not be set by accident. That sentence becomes historical the moment this stage lands: t4
gains its first writer. `t5` still has none, and gaining one is not part of 1D.

## D1. Which composition counts as t4

    t4 = the first submitted display update whose composition CONTAINS the authoritative
         real frame, and the end of the blend is NOT it.

The demo draws, for one input, a preview frame, then intermediate blend states, then the
authoritative frame. `t4` attaches to the first composition in which the authoritative
frame contributes pixels, because that is the first opportunity for the real model result
to enter the display pipeline.

The end of the 50 ms blend is a different product metric: it is when the pixels stop
changing. Sharing one timestamp between "the real result first reached the display" and
"the transition finished" would make both unreadable.

## D2. Blend timing is a separate metric

    t4_first_real_submit_ns        the authoritative frame's first submitted composition
    t_blend_complete_submit_ns      the submit at which the blend finished
    blend_duration_configured_ms    configuration, NOT a measurement

Derived:

    input_to_first_real_submit     t4_first_real_submit - t0
    real_decode_to_first_submit    t4_first_real_submit - t3
    first_submit_to_blend_complete t_blend_complete_submit - t4_first_real_submit

The configured 50 ms must never be reported as measured elapsed time. The measured
difference may legitimately differ from it because of frame cadence and scheduling.

## D3. t4 exists only on a real windowed path

Valid only when a real viewer backend exists, the frame is submitted to it, the backend is
not a dummy driver, and frame lineage ties the submitted frame to the committed generation.

    windowed pygame/SDL path        t4 may exist
    --headless / SDL dummy driver   t4 = None, reason "headless_dummy_driver"
    save_video / GIF / MP4          not t4
    image encode / file write       not t4

A headless `flip()` presents nothing, so stamping one would describe a call with no display
side. Headless and windowed runs therefore produce **different result schemas**, and a
headless recording must never be used to validate display latency. No t4 is synthesised or
estimated in a headless run.

## D4. Entry, not return

t4 is stamped **immediately before** the display-update call, at the moment the composition
is handed to the backend.

The alternative — stamping after the call returns — folds the swap or vsync wait into the
number and turns a submit metric into a partial display metric. `flip()` returning proves
an update call finished; it does not prove anything was presented.

## D5. t4 crosses a thread boundary, so the timestamp is taken at the submit

The demo's hard invariant is that the UI thread never touches `InteractiveRuntime`, whose
state has no synchronisation. But the UI thread is the only place a display submit happens.

Therefore, exactly as with `t0`: **the UI thread reads `perf_counter_ns()` at the submit**
and hands `(frame_id, generation_id, source_chunk_index, applied_event_ids, t4_ns)` to the
worker, which is the only writer of runtime state and records it.

The consequence must be stated rather than hidden: **the record is written later than the
instant it describes.** The timestamp is authoritative; its arrival in the runtime is not
part of any latency figure. A t4 taken by the worker when it got around to it would be
wrong by however long the worker was busy, which can be hundreds of milliseconds.

## D6. t5 stays unavailable

No verified signal exists on this path for "the submitted frame has completed presentation
and is observable on the physical display". Checked, not assumed: `pygame.display` has no
present/swap/vblank name, `_sdl2.video.Window` has none, `_sdl2.Renderer.present` is
`SDL_RenderPresent` (a swap call, not an acknowledgement), there is no
presentation-complete event type, `GL_SWAP_CONTROL` only configures, nothing requests vsync,
and SDL 2.28.4 exposes no such callback. See
`docs/LATENCY_1D_DISPLAY_BOUNDARY_AUDIT.md` for the probe evidence.

None of these may be used as t5:

    pygame.display.flip() return / SDL_UpdateWindowSurface return / SDL_RenderPresent return
    WINDOWEXPOSED
    vsync enabled
    configured cadence
    sleeping until the next frame
    decode completion

So `t5_presented_ns = None` is the only valid current authority, and `input_to_present` and
`control_to_real_display` remain **NOT MEASURED**.

**Naming note, because it matters:** the future physical-present capability is **not**
`§Latency-2`. That number is already taken by `docs/LATENCY_2A_CHUNK_DECOMPOSITION.md`
(chunk-phase accounting). The correct name is **`§Latency-4 — Present Completion
Authority`**, and it is only worth opening if a backend or capture mechanism capable of
proving presentation completion is actually adopted.

## D7. Lineage binding, exactly once

A t4 is valid only if the submitted frame is mechanically tied to the authoritative
lineage:

    event_id -> ApplicationClaim -> CommittedChunk -> generation_id
             -> authoritative FrameMeta -> renderer submit record

    RendererSubmitRecord(frame_id, generation_id, source_chunk_index,
                         applied_event_ids, t4_renderer_submit_ns)

These may be views over existing authoritative records rather than duplicated storage. The
implementation must not create a second lineage authority.

One authoritative real frame may be redrawn many times, but **only its first qualifying
submission receives t4**, later redraws must not overwrite it, and redraws are separate
display events rather than new authoritative t4 values:

    first_real_submit(frame_id) is write-once

## 2. Layers stay separate

    strict   which exact inputs produced this output     CommittedChunk.applied_event_ids,
             processed_ack(), exact frame lineage
    coarse   how far the input stream has progressed     processed_/settled_input_index
    raw      how long each stage took                    t0..t4, t5 only if a real
             completion signal ever exists

Timing must never substitute for attribution. A precise timestamp that cannot say which
input it measured is not a latency measurement.

## 3. Allowed metrics after 1D

    input -> assign / commit / first affected real decode / first real renderer submit
    commit -> real decode
    real decode -> renderer submit
    first real submit -> blend-complete submit        (when blending is enabled)

Still NOT reportable as measured: `input -> actual present`, `control-to-real-display`.

## 4. Result schemas

    windowed    t0 yes  t1 yes  t2 yes  t3 yes  t4 yes  blend-complete yes/no
                t5 unavailable
    headless    t0 yes  t1 yes  t2 yes  t3 yes  t4 unavailable (headless_dummy_driver)
                t5 unavailable

## 5. Implementation constraints

Same monotonic clock domain as 1A/B/C; `perf_counter_ns()` or the frozen equivalent; no
CUDA synchronisation added solely for timing; frame cadence, blend semantics, event
assignment and generation/commit order all unchanged; the RC tag does not move; headless
recording behaviour preserved; existing mock and GPU release checks preserved.

## 6. Minimum regression suite

    T1  first-submit semantics   with blending enabled, t4 is the first submission
                                 containing the real frame, not the blend end
    T2  write-once               many blend redraws cannot overwrite t4
    T3  headless absence         the dummy path yields t4 is None and never a synthetic
                                 submit timestamp, with the reason recorded
    T4  lineage                  every submit record resolves to the same generation_id,
                                 source_chunk_index and applied_event_ids as the
                                 authoritative frame
    T5  no t5 fabrication        no current path yields a non-null t5_presented_ns
    T6  monotonicity             t0 <= t1 <= t2 <= t3 <= t4 where the fields exist; do not
                                 assert against a missing t5
    T7  no semantic regression   the 1C acknowledgement, watermark, exactly-once and
                                 lineage suites stay unchanged and green

## 7. Exit criterion

     windowed   t3 measurable, t4 measurable and lineage-bound, t5 explicitly unavailable
     headless   t3 measurable, t4 unavailable, t5 unavailable

Closing 1D does **not** require solving physical-present measurement.

## 8. One-line interpretation

> §Latency-1D measures when an authoritative real frame first enters the real
> window/display pipeline; it does not claim to know when the user's physical display has
> actually presented that frame.
