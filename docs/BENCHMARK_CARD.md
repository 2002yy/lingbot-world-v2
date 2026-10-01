# Frozen benchmark card

One page. Final effective numbers only, no experimental branches.

## Platform

    RTX 5060 Laptop 8GB, sm_120, driver 591.86
    WSL2 Ubuntu, torch 2.8.0+cu128
    geometry 304x528 (latent 38x66, frame_seqlen 627)
    bf16, FA2 repro path, streamed condition encode

## Interactive timeline, frozen production stack

    input -> assigned          ~1 ms class
    preview visible            ~231 ms      variant D, 8-chunk integration: 27.2 ms decode
    authoritative ready        ~790 ms
    100% authoritative UI      ~840 ms      display semantics, 50 ms blend

    handoff peak, hard replace   0.0475
    handoff peak, blended        0.0119     0.25x

## Preview path, frozen

    variant D: skip TAEHV's last spatial upsample, then bicubic upscale
    training          zero
    new weights       zero
    cross-scene       PASS, no gross failures on 3 scenes
    handoff direction 0/21 wrong-direction, direction cosine 0.867
    decode            26.7 ms, against 36.0 ms for the full TAEHV

## Authoritative path, frozen

    3 denoise steps          541.9 ms
    1 clean-state write      170.7 ms   structurally required; holds the clean x0
    TAE decode                36.6 ms
    control plane              2.4 ms

    fewer DiT steps          REJECTED: KV divergence is monotonic and unplateaued
                             over 10 chunks (relL2 0.065 -> 0.274 for 2-step)
    KV update optimisation   CLOSED: 0.945x a denoise step, rolling memmove 1.8%

## Deployment presets, frozen

    performance (bf16)       peak 7104 MiB, min free 821 MiB, 29.25 s p50
    lowmem (fp8_lowmem)      peak 5846 MiB, min free 1161 MiB, 30.84 s p50

    repeated requests        25/25 both, no alloc leak, reserved plateaued
    length envelope          bf16 81 and 249 frames; fp8 up to 777

## Correctness

    interactive runtime tests        54/54 across four suites
    release smoke test               11/11 including the GPU path
    condition latent                 bit-identical, streamed vs whole-clip
    reference numerics               bf16 + FA2, bit-identical vs itself

## Not measured

    t4 renderer submit, t5 present, input-to-display: no instrumentation exists
