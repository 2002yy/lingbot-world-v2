# Interactive-1: the first trustworthy input -> first-real-frame chain

## Result, on the frozen performance preset

    preset=performance  weight=bf16  pixel=304x528  stream_encode=1  chunks=10

      ev    t0(s)   ->assign   ->commit     ->real  input->real  chunk  frame
       1  129.954        4.8      711.1      711.1        711.1      1      2
       2  132.133        2.3      751.3      751.3        751.3      4      5
       3  134.451        2.9      793.0      793.0        793.0      7      8
       4  136.023        1.6      772.3      772.3        772.3      9     10

    input -> first affected REAL frame:  n=4  p50 762 ms  min 711  max 793 ms

    input -> renderer submit    UNAVAILABLE (no renderer in this tree)
    input -> present            UNAVAILABLE (no present signal)
    measured input-to-display   UNAVAILABLE by construction

    chunk generation (DiT + decode + KV): p50 771 ms

## Why this is the first trustworthy one

1. t0 is a **real acceptance time**. `pump_input()` calls `rt.accept()` at genuine
   wall-clock moments, exactly as a keyboard callback would. This is the
   substantive difference from hotswap_loop, whose "relative elapsed minus script
   timestamp" mixed two clock bases.
2. t1 is a **real chunk binding**: `begin_chunk()` binds the queued events to that
   chunk and writes `assigned_chunk` into the trace.
3. t2 is a **fail-closed commit**: `commit(meta)` requires chunk_index,
   generation_id and applied_event_ids all to match the submitted snapshot.
4. t3 is the **first real frame with lineage**: `mark_real_decoded()` accepts only
   `frame_kind="real"` and sets the field exactly once.
5. t4 and t5 are reported **UNAVAILABLE** rather than filled with the nearest
   convenient timestamp.

## Reading the numbers

    input->assign   1.6-4.8 ms   the event is bound to the next chunk almost
                                 immediately; the queue never backs up
    input->commit   711-793 ms   one chunk of generation (3 DiT steps + KV update)
    input->real     711-793 ms   the same order as commit, because TAE decode sits
                                 immediately adjacent to it

Note that input->commit and input->real are nearly equal: TAE decode is short
relative to the DiT, so t3 and t2 occur at almost the same moment. That is a
property of the TAE path, not a measurement error.

This 762 ms is **not comparable** to the demoted historical "control-to-real
~950 ms / ~1.38 s" figures, which were formulas, configuration values or
decode-return timestamps with different semantics and denominators. This figure
means: input accepted by the queue -> the first real frame certainly affected by it
finishes decoding.

## Denominator stack, now five deep

    micro / kernel benchmark
            |
    DiT + TAE benchmark profile
            |
    full deployment request          <- the Deploy-1 frozen table
            |
    input -> first-real-frame        <- new this round: 762 ms
            |
    input -> present                 <- still unavailable (no renderer)

## State machine behaviour in the real chain

- events 1/2/3/4 were bound to chunks 1/4/7/9 respectively, each to exactly one
  chunk, with no duplicates;
- `committed_chunk` advanced monotonically 0 to 9 with no regression;
- no commit was refused, so lineage matching held;
- chunk 0 carried no applied event, as expected since the script's first input
  arrives at t=2.0 s.

## Still open, recorded honestly

    t4 renderer submit   no renderer exists in this tree (Latency-1A ruled this
                         architecturally unrepresentable)
    t5 physical present  no present signal

So measured input-to-display remains unavailable in this architecture. Obtaining it
requires first introducing a real renderer and determining whether that renderer
exposes a present-completion signal at all.

## Next

    Interactive-2  wire the proven camera / hot-swap foundations into this runtime
    Latency-1B     unify the traces into the formal LatencyTrace record
    Display-1      a real renderer is its precondition if t4/t5 are ever wanted
