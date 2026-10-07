# Latency-1D: display boundary capability audit

    STATUS   audit DONE, contract frozen below. NO timer was added by this stage.
    scope    t3 first affected real frame decoded
             t4 renderer submit
             t5 actual present
    method   capability audit FIRST. Inserting a timer before establishing which
             boundary is real is how a proxy gets promoted into a measurement.

## The three questions

### Q1. Where does a real frame actually reach a renderer?

**Nowhere in the runtime. Only in the demo.**

    imshow | namedWindow                      <none>
    QApplication | glfw | swapBuffers         <none>
    pygame | SDL                              demo_wasd.py
    save_video | write_video | VideoWriter    generate.py, wan/utils/utils.py
    any windowing outside demo_wasd.py        <none>

`save_video` writes an mp4 to disk; it is not a renderer. The single display surface in
the tracked tree is `demo_wasd.py:969`

    pygame.display.flip()

This matters for scoping: **t4 is definable for a process that has a viewer, and is not
definable for the runtime.** `play.py` and `interactive_runtime.py` still have no renderer,
which is exactly what §Latency-1A found and declared. 1D does not change that; it locates
the boundary where it now exists.

### Q2. Can `pygame.display.flip()` serve as t4?

Yes, as a **submit** boundary, and only if it is named as one.

`flip()` routes to SDL. On the software-surface path this is
`SDL_UpdateWindowSurface`, a copy into the window surface; the present itself is then done
by the platform compositor, and on this machine that is one more layer away (WSLg, then
Windows). What `flip()` returning proves is that **one update/swap call finished**. It does
not prove a user could see that frame.

So:

    flip() entry / return   ->  renderer/display SUBMIT boundary   ->  t4 candidate
    "the user has seen it"  ->  NOT established by this call       ->  NOT t5

### Q3. Is there any trustworthy present / vblank / swap-complete signal?

**No.** Checked, not assumed:

| Candidate | Result |
|---|---|
| `pygame.display` present/swap/vblank/frame_ready names | **NONE** |
| `pygame._sdl2.video.Window` present-shaped methods | **NONE** (`size`, `title`, `show`, `focus`, `opacity`, …) |
| `_sdl2.Renderer` | has **`present`**, which is `SDL_RenderPresent`: a swap call, not an acknowledgement |
| presentation-complete event type | **NONE**. `WINDOWEXPOSED` means "needs redraw"; the rest are move/resize/focus |
| `GL_SWAP_CONTROL` | `= 0`, a `set_mode` flag that selects immediate vs vsync. It configures; it reports nothing |
| vsync requested anywhere in the tree | **nothing** |
| SDL version | 2.28.4. SDL2 exposes no present-completion callback |

A real present acknowledgement would need something this stack does not reach:
`VK_KHR_present_wait` on a Vulkan swapchain, `IDXGISwapChain::GetFrameStatistics` under
D3D, or a compositor that reports scanout. None of those is reachable from a pygame
software surface.

## Frozen capability matrix

    input -> real-decoded      MEASURABLE          t3, already implemented and tested
    input -> renderer-submit   MEASURABLE, but only in a process that has a viewer,
                               and only as a SUBMIT, never called a present
    input -> present           NOT YET MEASURABLE  no signal exists in this stack

This is the disposition §Latency-1A predicted, and 1D confirms it against the code that
now exists rather than against the code that existed then. `t4` moved from "no renderer at
all" to "a renderer exists in the demo"; `t5` did not move.

**Forbidden, and unchanged from 1A:** picking the nearest convenient callback and calling
it present. A missing time is missing.

## Two definition questions 1D must settle before any timer is added

**1. Which `t4`?** The demo draws three different things for one input: the preview frame
when it arrives, three intermediate blend frames, and the authoritative frame. "Submit"
for the authoritative frame carrying a given input is therefore ambiguous between

    (a) the first draw of that authoritative frame, and
    (b) the end of the 50 ms blend, when that frame is fully on screen

(a) is the submit of the data; (b) is when the pixels stop changing. They differ by the
blend duration by construction, so this is a real choice and not a rounding detail.

**2. What does t4 attach to when the take is headless?** The recorded GIF take runs with
`--headless`, i.e. `SDL_VIDEODRIVER=dummy`. Under that driver `flip()` presents nothing at
all, so a submit timestamp taken there would describe a call that had no display side. Any
t4 measurement must therefore be marked as existing only for a windowed run, or the
recorded take must be produced windowed.

## What 1D deliberately does NOT do

- It adds no timer, and it changes no code.
- It does not define `t5`.
- It does not introduce `warp` frames; there is no warp path in this tree.
- It does not introduce `SUPERSEDED`; an input arriving during a chunk is claimed to the
  next chunk, and preemption is §Latency-3B.

## Evidence

Read-only. `pygame 2.6.1`, `SDL 2.28.4`. The probes are reproducible from the commands in
`_l1d_audit.sh` and `_l1d_probe.sh` in the working tree:
`git grep -lIE` over the tracked tree for every display surface, and `dir()` inspection of
`pygame.display`, `pygame._sdl2.video.Window` and `pygame._sdl2.Renderer`.

No result here depends on a GPU run, and none is inferred from a figure measured
somewhere else.
