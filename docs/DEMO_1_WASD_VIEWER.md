# Demo-1: WASD viewer — recorded plan, not yet implemented

Status: **DELIVERED** on `demo/wasd-viewer`. See "What was actually built" and
"Delivered numbers" at the end of this file; the plan above is kept unedited so the
deviations are visible. The `rtx5060-interactive-rc1` tag was not involved and did not
move.

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

---

# Delivered

`demo_wasd.py`, plus `docs/demo/rtx5060_wasd_demo.mp4` and `.gif`. No other file was
changed: not `play.py`, not `interactive_runtime.py`, nothing under `wan/`, and the tag
did not move.

## What was actually built

    demo_wasd.py
      Worker(Thread)          owns the runtime AND the session; the only thread that
                              mutates either
      WanSession              the frozen stack, wired as play.py wires it
      MockSession             no GPU: real runtime, synthetic frames
      Viewer                  pygame UI: keys, blend, HUD, recording
      preflight_gpu()         refuses to start on retained VRAM (see below)

Deviations from the plan, and why:

1. **pygame had to be installed** (2.6.1, `--no-deps`). It was absent; OpenCV is present
   but gives no keyup events, so a held `W` plus a fresh `D` cannot be told from two
   separate presses. Held state is tracked from KEYDOWN/KEYUP instead of
   `pygame.key.get_pressed()`, so posted and real events follow one path.
2. **The session bench is duplicated from play.py**, not extracted. play.py is the RC's
   frozen entrypoint; a shared module would mean editing it. It is the only duplicated
   block and it is marked as such in the file.
3. **The clock starts at the first authoritative frame.** `--seconds` and the recording
   both begin there, so the cold first chunk is excluded. It is reported separately as
   `ready_ms` (3.0 s measured) rather than hidden.
4. **The HUD shows live numbers, not the plan's fixed "~231 / ~820 ms".** The fixed pair
   turned out to describe a different measurement than this one (next section), so
   printing it would have been misleading.

Also added: `--headless` (SDL dummy, used for the recorded take), `--warmup_timeout`,
`--out_json`, and `preflight_gpu()`.

## The measured correction: boundary-aligned vs asynchronous input

The RC's frozen `input -> first real frame` is **762 ms p50** (play.py; 820 ms class in
the release notes). This viewer measures **~1.0-1.2 s** at the same geometry and weight.
Neither number is wrong; they measure different things.

play.py's input source is scripted, and `pump_input()` runs at the top of the chunk loop,
so **every event is delivered exactly at a chunk boundary and its assign wait is
identically zero for free.** A real keypress arrives at an arbitrary phase:

    input -> authoritative = (remaining time of the in-flight chunk)
                           + (one full chunk: 3 denoise steps + KV write + decode)
                           + (decode)

so it ranges from one chunk period to two. With a chunk of 650-930 ms on this machine,
that is 0.9-1.5 s, which is what the recordings show. The first measurement run
(`output/demo1/run_norec.json`) makes the structure visible directly:

    ev   t0(s)   ->assign   ->commit   ->first-real
     1   0.000      2344       3104         3130    <- cold first chunk, not representative
     2   2.005       339       1099         1125
     3   4.013       346        980         1006
     4   6.010       297        857          882

`assign` is the wait for the in-flight chunk; `assign -> commit` is the event's own chunk.
Both are real stages, and a single "input to display" number hides the first one.

**This is the honest interactive figure.** It is also the argument for any future latency
work: the cheapest win available is not a faster kernel but making the in-flight chunk
interruptible, which the EarlyExit reopening condition already describes.

## Delivered numbers

Recorded take, `--script --headless --weight bf16 --pixel 304x528 --blend_ms 50`:

    keypress -> preview decoded        p50  647 ms   403 -  941 ms
    keypress -> first affected REAL    p50 1199 ms   938 - 1455 ms
    world ready (cold chunk)                3.0 s   excluded from --seconds
    chunks generated                        15 in ~13 s  (~870 ms/chunk, this take)
    UI                                    62.1 fps sustained, 960x540

    artifact  docs/demo/rtx5060_wasd_demo.gif   640x360, 12 fps, 117 frames, 9.75 s, 3.01 MB
                                                 -> COMMITTED, embedded in README.md
              docs/demo/rtx5060_wasd_demo.mp4   h264, 960x540, 30 fps, 293 frames, 9.77 s,
                                                 1.89 MB -> NOT committed: .gitignore
                                                 excludes *.mp4. Regenerate with the
                                                 command below, or attach it to an issue
                                                 or PR for a user-attachments URL.

The viewer is not tied to one preset. Run once with `--weight fp8_lowmem` (no recording),
all six GPU checks still PASS, and the numbers show the same trade the RC recorded:

| preset | keypress → preview | keypress → authoritative | chunk period |
|---|---:|---:|---:|
| `bf16` (`performance`) — the shipped GIF | 647 ms p50 | 1199 ms p50 | ~870 ms |
| `fp8_lowmem` (`lowmem`) | 896 ms p50 | 1574 ms p50 | ~1080 ms |

So the FP8 weight-only path costs ~25-30% more interactive latency in the viewer too.
It buys VRAM, not speed, independently of the earlier weight-only microbenchmark.

Chunk time varies between runs (645-930 ms observed) with machine state, so the latency
figures move with it. A second take measured p50 1042 ms (1034-1057); a run made while
~7 GiB was still retained from a previous process measured p50 1296 ms. The take shipped
is the clean-VRAM one.

## Phase characterization (this is the number to quote, not the 4-event take)

A demo take has four input events, which is a picture and not a distribution. To
characterize what a person actually gets, `--phase_n` fires single-key events at
randomized offsets so the input's phase against the chunk boundary is uniform. 30 events,
seed 7, inter-arrival uniform(1.2, 2.6) s against a ~650 ms chunk, 91 chunks generated,
no recording:

| from the keypress, model-side | p50 | p90 | best | worst |
|---|---:|---:|---:|---:|
| → preview decoded | **593 ms** | 788 ms | **232 ms** | 827 ms |
| → first authoritative real frame | **1033 ms** | 1269 ms | 643 ms | 1301 ms |
| of which: wait for the in-flight chunk | 414 ms | 597 ms | 63 ms | 641 ms |
| of which: the event's own chunk | 591 ms | 667 ms | 553 ms | 683 ms |

All 30 committed, 0 preview-commit violations, t0 preserved 30/30.

Three things this settles.

1. **The 762 ms release figure is the zero-phase case.** The wait-for-in-flight term is
   what it sets to zero for free, and here that term is 414 ms p50. The sum checks out:
   414 + 591 + ~28 ms decode = 1033 ms.

2. **232 ms to preview is the release's ~231 ms.** That figure is not wrong; it is the
   favourable end of the phase distribution. Observed minimum 232 ms against a p50 of
   593 ms. It occurs when an intent lands just before a chunk boundary (wait 63 ms).

3. **The wait is bounded by the chunk period, not by noise.** 63–641 ms against a ~650 ms
   chunk, which is exactly what uniform phase predicts.

The run also exposed a real defect: `--n_chunks` sizes both the per-chunk noise tensor and
the condition latent, and this viewer runs open-ended, so a long run would have died with
an `IndexError` inside `split()` that says nothing about the cause. It now raises a named
error instead.

## Verified

CPU/mock (`--mock`, no GPU, SDL dummy), 12/12 PASS. This is the part that must hold in
CI, and it exercises the real runtime and the real worker loop, only the frames are
synthetic:

    worker produced no error
    no deadlock: worker joined
    intents emitted by the viewer
    runtime accepted the intents (none lost to a bad handoff)
    t0 is the KEY EVENT timestamp, preserved across the thread handoff
    every accepted event reached a terminal status
    every committed event has a first real frame
    preview NEVER advanced committed state
    authoritative lineage aligns with the committed chunk
    t1 <= t2 <= t3 for every committed event
    the blend ran and landed by assignment
    frames were drawn

On the real GPU, per recorded run:

    real keyboard events produced InputEvents      PASS (4)
    t0 came from the key event, not the drain      PASS (4/4 preserved, bit-identical)
    preview never committed                        PASS (15 previews, 0 violations)
    authoritative lineage aligned with events      PASS (4 committed)
    no deadlock (worker joined)                    PASS
    ran the full requested duration                PASS (10 s)

"t0 preserved" is checked as exact equality between the timestamp the viewer stamped at
the key event and the `t0_ns` the runtime stored, so the thread handoff is proven lossless
rather than assumed. "Preview never committed" is checked structurally at the moment each
preview is emitted: `committed.chunk_index` must still be `chunk_index - 1`.

## Three findings recorded, not fixed

1. **play.py's frame reduction picks a colour channel, not a frame.** `decode_video`
   returns `[B, T, C, H, W]`; after dropping `B` the code tests `dim() == 4` and slices
   `dim 1`, which on that layout is the CHANNEL axis. With `T=1` per chunk it stayed
   self-consistent, so the handoff-peak and blend figures are real but are single-channel
   (green) rather than image-level. Relative comparisons are unaffected; absolute peaks
   are. `demo_wasd.py` extracts RGB properly and does not reuse it. Not fixed here because
   play.py is the frozen RC entrypoint and this file may not change it.

2. **A finished CUDA process does not release its VRAM at once on this machine.** After a
   successful run, `nvidia-smi` showed 6981 MiB used with *no* process holding it; a
   back-to-back run then died with `CUDA error: out of memory` inside `prewarm`, which is
   a message that says nothing about the real cause. The memory was reclaimed on its own a
   few minutes later (16 MiB used). `preflight_gpu()` now checks before starting.

3. **The preflight has to run before torch is imported.** `torch.cuda.mem_get_info()`
   reports ~6.87 GiB free in both the healthy and the retained-VRAM case, because torch's
   own context accounts for the difference. Only the device-level reading separates them
   (7.70 GiB vs 0.90 GiB), so the check uses nvidia-smi and no torch import.

## How to run

Live, with a window:

    python demo_wasd.py

(Not wired into `run.sh`: the plan for this work was new files only, and `run.sh` is
release tooling. Adding a `demo` subcommand there is a reasonable follow-up.)

Recorded, same as the shipped take:

    python demo_wasd.py --script --headless --seconds 10 \
      --record docs/demo/rtx5060_wasd_demo.mp4 --record_fps 30 \
      --pixel 304x528 --weight bf16 --blend_ms 50 \
      --out_json output/demo1/run_record.json

CI, no GPU:

    python demo_wasd.py --mock

## What this does not claim

- **No physical present.** The 50 ms blend is executed on our own framebuffer. There is
  still no renderer and no present-completion signal, so `t4`/`t5` remain unavailable. The
  HUD says "model-side" for this reason, and the measured figure is keypress-to-model-output,
  not keypress-to-photon.
- **No 60 Hz held-key sampling.** One keydown is one discrete control intent. Held-key,
  OS key-repeat and keyup policy is not in the frozen contract and is not invented here.
- **`--script` is not a human at a keyboard.** The shipped take uses it. It posts real
  KEYDOWN/KEYUP events into the real queue, so they pass through the same handler and the
  same stamping and the same runtime path a human's keys pass through; only the source of
  the press is automated. For the live figure, pass no `--script`.
- **One scene.** The take is `examples/04`. The preview path's cross-scene behaviour is
  Preview-1C's evidence, not this file's.
