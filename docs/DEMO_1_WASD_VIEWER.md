# Demo-1: WASD viewer — recorded plan, not yet implemented

Status: **PLANNED**. Work happens on this branch, `demo/wasd-viewer`, which is created
from the community branch. The `rtx5060-interactive-rc1` tag is not involved and does
not move.

## The correction this plan is built on

`./run.sh play` must NOT be presented as a real WASD demo as it stands. Two facts:

1. `play.py`'s input source is a hard-coded scripted timeline. Its own module docstring
   says so: *"The input source is scripted"*.
2. The runner is headless. The 50 ms preview-to-authoritative blend is computed on a
   frame buffer; there is no renderer and no present. `t4`/`t5` remain unavailable.

So the goal is not to cut a video that looks live. It is to add a thin display layer and
then record that.

## Scope

New files only: `demo_wasd.py`, plus `docs/demo/` for the exported media.

**Must not change:** `interactive_runtime.py`, anything under `wan/`, the model path, or
the `rtx5060-interactive-rc1` tag.

## Design: Pygame main thread + GPU worker thread + two queues

A chunk takes about 0.8 s. Running the model on the UI thread would freeze the window for
800 ms after every keypress, so the split is not optional.

    UI main thread                      GPU worker
      pygame window
      key event
        timestamp with perf_counter_ns()
        --> input_queue ------------->  drain at a legal seam
                                        rt.accept(controls, _now_ns=<real key time>)
                                        rt.begin_chunk()
                                        DiT step0
                                        variant D decode  --> frame_queue: PREVIEW
                                        step1 + step2
                                        state write
                                        rt.commit()
                                        full decode, t3   --> frame_queue: AUTHORITATIVE
      <-- frame_queue
      display preview / authoritative
      real ~50 ms visual blend

**The UI thread must never touch `InteractiveRuntime`.** The runtime's state is a plain
`deque`/`dict`/`_inflight` with no synchronisation, so calling `accept()` from the pygame
thread would introduce a data race into an already-closed correctness contract, purely
for a demo. Instead the UI thread puts `(timestamp_ns, controls)` on a `Queue` and the
worker calls `accept()` with `_now_ns=<key timestamp>`, so **t0 is still the real key
event time** while only one thread ever mutates the runtime.

## Input mapping, using the existing contract

One `InputEvent` is one discrete control intent, and each event applies a fixed
integration window. Held-key sampling, OS key repeat and keyup handling are **not** part
of the frozen contract, so they are not implemented here.

    W        forward +0.6
    S        forward -0.6
    A        yaw     -0.8
    D        yaw     +0.8
    W+A      forward +0.3, yaw -0.8
    W+D      forward +0.3, yaw +0.8

A new discrete intent is accepted once. This keeps behaviour identical to the verified
`control_reduce` semantics instead of defining a second set of motion rules just for the
demo.

## Frame delivery

    preview:        {kind: "preview", frame, event_id, latency_ms}
    authoritative:  {kind: "authoritative", frame, event_id}

The UI shows the preview immediately and blends to the authoritative frame over roughly
three intermediate steps (~50 ms), which is the first time Preview-2B's display
semantics become something a person can actually watch.

## HUD: four things only

    RTX 5060 Laptop 8GB · 304×528
    WASD key state
    PREVIEW / AUTHORITATIVE badge
    "Preview ~231 ms · Authority ~820 ms" with a small "model-side timing" note

Not on the video: SSIM, LPIPS, KV cache, t0..t3, BF16, FA2. Those belong in the README.

## Wording discipline

Say: **actual keyboard WASD interaction.**
Do not say: *continuous 60 Hz WASD sampling* — that is not implemented and is not in the
contract.

Say: **model-side preview ~231 ms** and **authoritative frame ~820 ms.**
Do not say: *input-to-display = 231 ms* — there is still no physical-present measurement,
and the UI draw time is not a present.

## Recording

Have `demo_wasd.py` record its own UI framebuffer rather than capturing a desktop. That
removes the desktop, the mouse, the terminal, and any HUD/video desync, and makes the
take reproducible.

    --seconds 10          run length
    --record <mp4>        write the UI framebuffer
    960x540, 30 fps, H.264

    docs/demo/rtx5060_wasd_demo.mp4     then a README GIF:
    ffmpeg -i docs/demo/rtx5060_wasd_demo.mp4 \
      -vf "fps=12,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
      -loop 0 docs/demo/rtx5060_wasd_demo.gif

Targets: MP4 8-12 s at 30 fps; GIF 8-12 s at 12 fps, 640 px, preferably under 10 MB.

README:

    ![RTX 5060 Laptop 8GB interactive WASD demo](docs/demo/rtx5060_wasd_demo.gif)

For the video, upload to GitHub and use the `user-attachments` URL, which is the form the
upstream README already uses.

## 10-second script

    0-1 s    initial world
    1-3 s    W      forward
    3-5 s    W+D    forward + turn
    5-7 s    A      turn the other way
    7-9 s    W      forward again
    9-10 s   stop, show final statistics

## Acceptance

CPU / mock, no GPU:

    the viewer queue, the input path and the state machine all run headless

GPU:

    real keyboard input produces an InputEvent
    the event's t0 comes from the key event, not from the worker's drain time
    a preview never commits
    authoritative lineage aligns with the event
    no deadlock
    it runs for the full 10 seconds
    the MP4 plays

## Why this is Demo-1 and not a performance task

It is product surface, not optimisation. After it exists the community branch can become
the repository's front page, and a stranger landing on the repo can understand what was
built without reading the findings document.
