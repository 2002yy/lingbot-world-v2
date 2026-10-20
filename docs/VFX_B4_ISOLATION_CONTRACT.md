# §VFX-B4: one-way isolation contract between the world model and a display enhancement layer

    BASELINE   rc2 @ a875a0d — not modified, not moved
    AUDIT      c796095 (§VFX-A)
    STATUS     contract FROZEN; offline negative controls PASS; NOT wired to any SDK
    SCOPE      contract + SDK-independent mock enhancer + T1-T6 with injected violations.
               No SDK, no GPU dependency, no change to the production inference chain.

## Why this is frozen before anything is wired

B4 decides whether a display enhancer **can be attached safely**. B0-B3 decide whether it is
**worth attaching**. They are independent questions, and answering the second first would mean
risking the world model's authority for a module whose benefit on a 5060 is not yet known.

So the order is: freeze the boundary now, measure worth later, wire last.

## 1. The invariants

### V1 — the data flow is one-way

    permitted   authoritative world state -> committed raw frame -> display enhancement
                -> display output
    forbidden   display-enhanced frame -> VAE / DiT / KV / RNG / camera commit
                -> next generation input

An enhanced frame is not a new world state and holds no authority to update one.

### V2 — authority never depends on display success

Whether the enhancer succeeds, fails, times out, crashes or runs out of VRAM:

- the world model's commit logic is unchanged;
- the raw generated frame remains valid;
- a failed enhancement degrades to the raw frame or to no display at all;
- already-committed world state is not modified;
- a display-layer failure must never re-run generation or roll back a commit.

### V3 — object and memory isolation

Different Python references are not sufficient. The actual storage relationship is what is
checked. The display side must not hold the ability to modify authoritative image or state
storage. Zero-copy GPU sharing, if ever used, must prove buffer lifetime, read/write permission
and CUDA asynchronous ordering; otherwise separate buffers are used.

**None of the following is accepted as proof of isolation**: a different variable name, a
different object wrapper, or `detach()` on its own.

### V4 — time and input isolation

The display side may consume timestamps. It may not:

- produce or rewrite authoritative camera state;
- treat an interpolated temporal position as a new model commit instant;
- put a display frame back into the generation input queue;
- derive a next model action from a display frame;
- alter existing preemption, RNG or KV behaviour.

Real user input continues to travel the original authoritative input chain.

## 2. The tests, and how each one proves it has teeth

Every test is paired with an injected violation. A test that only ever passes proves nothing;
this project has already had to learn that once (the C1 semantic check, and the P0 oracle that
reported "the thing under test never ran").

    T1  data contamination     the enhancer mutates the frame it was given, writes noise,
                               swaps the object, or tries to write the underlying tensor.
                               Expected: the authoritative frame and its state hash are
                               unchanged, and a boundary violation raises.

    T2  reverse injection      the enhancer returns a plausible-looking interpolated frame and
                               someone tries to feed it to the next generation step.
                               Expected: refused at the interface or runtime boundary, NOT by
                               relying on callers to remember not to pass it.

    T3  failure and degrade    timeout, exception, out of VRAM, empty frame, wrong resolution,
                               early exit.
                               Expected: the enhancement path degrades and the authoritative
                               commit result is unchanged.

    T4  async races            display slower than the model, frames completing out of order,
                               the source buffer reused early, a stale request returning late.
                               Expected: display selection changes; authoritative state does
                               not; a stale display result cannot overwrite a newer commit.

    T5  toggle equivalence     same inputs, same seed, same inference configuration, enhancer
                               off versus on.
                               Must match: the committed state sequence and count, authoritative
                               frame content hashes, KV, RNG and camera state, generation input
                               and commit semantics, and any interruption or recovery result.
                               Comparison must use this project's existing determinism contract
                               and tolerances, not widened ones.
                               Display frame COUNT may differ and does not participate.

    T6  resource lifetime      with enhancement off, no extra display model is created and no
                               enhancement GPU resource is held. With it on, enhancement
                               resources are released independently and do not damage resources
                               the world model is using.

## 3. Implementation boundary for this slice

Permitted:

- new or revised isolation contract documentation;
- a new SDK-independent mock enhancer and its verification fixtures;
- new type-boundary, data-flow and reverse-injection negative-control tests;
- inspecting the real production data flow and recording future wiring points.

Forbidden:

- modifying DiT, KV, RNG, commit or preemption;
- modifying the authoritative frame generation algorithm;
- introducing the SDK or any new GPU runtime dependency;
- changing existing authoritative state semantics in order to pass a test;
- modifying or moving the frozen rc2 @ a875a0d.

## 4. Acceptance

**This offline slice is complete when** every isolation negative control passes against the mock
interface AND each test has been shown to detect a deliberately injected violation.

**It is NOT complete as production isolation.** The following still require the SDK and the real
concurrent world model:

    real GPU buffer lifetime and read/write permission, including any zero-copy path
    long-sequence authoritative state consistency with enhancement running concurrently
    the wiring point check against the real frame path rather than a mock

Only when the full real path passes does B4 move from "contract frozen" to "isolation verified".
**Passing the mock tests must not be reported as production isolation being established.**

## 5. Deliverable

This contract, the test and negative-control matrix, the run commands, and the results, with an
explicit list of what the offline tests prove, what waits on the SDK, and what can only be
verified with the real world model and display concurrent.

---

# §VFX-B4 offline slice: results

    STATUS   contract frozen, offline negative controls PASS, NOT wired to any SDK.
             Passing these does NOT mean production isolation is established.
    BASELINE rc2 @ a875a0d, untouched.

## The matrix: every guard, and every guard removed

A test that only ever passes proves nothing, so each invariant is asserted twice: once with
the guard in place, and once with the guard deliberately removed, where the test fails unless
the damage is actually observable.

| Test | Guard held | Guard removed, damage must be visible | Result |
|---|---|---|---|
| T1 data contamination | read-only flag refuses an in-place write; the authoritative frame is unchanged | with `copy=False, read_only=False` the enhancer holds a writeable VIEW and the source IS damaged | 2/2 PASS |
| T2 reverse injection | the interface refuses a display object; the runtime's commit refuses non-authoritative provenance | the display object is an ordinary `ndarray`, so without the interface check nothing in the boundary module stops it | 2/2 PASS |
| T3 failure and degrade | six failure modes (raise, empty, bad shape, none, timeout, no enhancer) all degrade; committed state untouched under 12 consecutive failures | without the wrapper the exception escapes to the caller | 3/3 PASS |
| T4 async races | a stale display result cannot overwrite a newer commit; reverse-order delivery drops the four superseded results | an unguarded stage assigns whatever arrived and does overwrite newer state | 3/3 PASS |
| T5 toggle equivalence | enhancer off vs on: identical committed state sequence and identical authoritative frame hashes | the comparison distinguishes different committed states, so it is not vacuous | 2/2 PASS |
| T6 resource lifetime | nothing is constructed when enhancement is off; one instance when on; released independently | an eager factory constructs unconditionally regardless of the flag | 3/3 PASS |
| boundary shape | no enhancer class exposes `rt` / `session` / `pipe` / `kv` / `_gen` / `committed`; a frozen `DisplayFrame` cannot be rebound | — | 2/2 PASS |

    17/17

## A finding about the mechanism, from T1's first failure

The first version of T1 asserted that unlocking a read-only frame would raise. **It does not.**
`np.array(frame, copy=True)` returns an array that owns its data, and numpy permits re-enabling
the writeable flag on such an array, so `setflags(write=True)` succeeds and the enhancer can
then write.

**The read-only flag is therefore not the isolation boundary. The copy is.** The flag is a
reminder that stops the accidental `frame[...] = x`; what actually protects the authoritative
frame is that the enhancer never holds its memory. T1 now asserts `not np.shares_memory(...)`
and, separately, that unlocking and writing leaves the source unchanged.

Recording this matters because the obvious reading -- "the frame is read-only, so it is safe" --
is the kind of assurance that would fail in exactly the case it was supposed to cover.

## The production wiring point, inspected rather than assumed

The frame path in the real code:

    Worker._run        real = self.session.decode_real(x0)
                       -> WanSession._to_hwc_u8: a.permute(1,2,0).contiguous().cpu().numpy()
                       -> a FRESH host array, not a view of the GPU tensor
                       self.frame_q.put(AuthoritativeMsg(frame=real, frame_id=meta.frame_id,
                                                         chunk_index=..., generation_id=...))
    Viewer._drain_frames   consumes it, then _scaled_f32() scales a copy for the window

**The future wiring point is the `AuthoritativeMsg` consumer in `Viewer._drain_frames`**, and
the isolation that matters is already partly established by construction: `_to_hwc_u8` ends in
`.cpu().numpy()` on a contiguous copy, so the display side never shares memory with the model's
GPU tensors regardless of what an enhancer does.

That is a property of the existing design, not of the boundary module, and it is the reason
this slice could be frozen without touching the runtime.

## What the offline tests prove

- a display object cannot reach generation: the interface refuses it, and `commit()` refuses
  non-authoritative provenance, which is an existing runtime guard rather than a new one;
- a failing, hanging, empty or malformed enhancement degrades and cannot raise into the caller;
- committed state is unchanged across twelve consecutive display failures;
- a superseded display result cannot overwrite a newer one, and reverse-order completion drops
  exactly the superseded results;
- with enhancement off versus on, the committed state sequence and the authoritative frame
  hashes are identical, verified through the real `InteractiveRuntime` and the real worker loop,
  not a stub;
- no enhancer class holds a route into `rt` / `session` / `pipe` / `kv` / `_gen` / `committed`;
- nothing is constructed when enhancement is off.

## What still waits on the SDK

- real GPU buffer lifetime, ownership and read/write permission, including any zero-copy path;
- whether VFG/VSR's own buffer handling is compatible with the isolation shape assumed here;
- the actual cost, which decides whether any of this is worth wiring at all.

## What can only be verified with the real world model and display concurrent

- long-sequence authoritative state consistency while enhancement runs concurrently, at 65
  chunks and beyond, using the project's existing determinism contract rather than a widened one;
- VRAM behaviour on 8 GB with the model, the enhancer and the super-resolver all resident;
- the wiring-point check against the real frame path rather than the mock consumer above.

**Passing 17/17 offline is not production isolation.** B4 moves from "contract frozen" to
"isolation verified" only when the full real path passes.

## Reproduce

    python test_display_isolation.py
