# M0-prod-A: 304x528 passes the real production stability gate

## Model nominal geometry is NOT deployment geometry authority

    model / upstream authority   image2video.py default, native aspect,
                                 max_area 480*832 -> 512x768
                                 nominal, but INFEASIBLE on 8 GB
    deployment authority         304x528, chosen inside the feasible frontier
                                 (see docs/M0_PROD_FRONTIER.md)

The two must never again be called "the default resolution".

## Configuration: the real production chain, not a benchmark loop

Drives `wan.WanI2VCausal.generate()` itself, so the whole request runs: text
conditioning, condition encode, the causal chunk loop, the KV update, and the VAE
decode.

    pixel 304x528 (lat 38x66, fsl 627)   frames 81 (generate's own default)
    chunk_size 3   timesteps [0,250,500,750]   seed 42
    preset local_window 8, sink 2 (worst-case KV window)
    weight bf16

Geometry selection matters: `generate()` derives the geometry from the NATIVE
aspect of the image it is handed plus `max_area`. So 304x528 is selected by
pre-resizing the image to 304x528 and passing max_area = 512*320 = 163840:

    aspect = 304/528 = 0.5758
    lat_h = round(sqrt(163840*0.5758)//8//2*2) = 38
    lat_w = round(sqrt(163840/0.5758)//8//2*2) = 66   -> 304x528

## Result: bf16 at 304x528, 25 consecutive requests -- PASS

    reserved  (first 12 = warm-up, last 13 = equilibrium):
      5262,5422,5302,5482,5462,5662,5882,5842,5702,5802,5902,5942,
      5742,6022,6082,6082,6282,6142,6022,6182,6142,6002,5862,6102,6022
    allocated: 731 flat across all 25

    reserved window means (window=6): 5432, 5845, 6059, 6052
    window deltas:                    +413, +213,   -7

    1  allocated growth (leak)     +0 MiB     PASS (no leak)
    2  reserved last vs prev win    -53 MiB    PASS (plateau)
    3  global peak spread          +622 MiB   (max 7202)
    4  min free headroom            721 MiB   PASS
    5  warm latency p50              26.54 s  (min 26.11, max 35.19)
    OVERALL (plateau test): PASS

## Correction to the gate itself

The gate says "no obvious VRAM creep", which I first implemented as a
first-vs-last comparison under 64 MiB. That is the wrong test: an allocator warms
up (its block pool grows to the high-water mark) and then settles, so comparing
request 0 with request 24 measures warm-up plus equilibrium noise, not a leak.

The right three questions are:

    1. does `allocated` grow?     -- that is a real leak
    2. does `reserved` flatten?   -- equilibrium vs unbounded creep
    3. is the last-window trend ~0? -- distinguishes slow plateau from slow creep

With the corrected test, reserved plateaus at ~6050 MiB (last window -7 MiB) and
allocated is perfectly flat, so the verdict flips from FAIL to PASS. Implemented
as `m0prodA_verdict.py`.

## Two real production-path defects found

### 1. `offload_model=True` leaves the model on the CPU

    bf16 + offload=True (generate's own default):
      [0 cold] ok=True   79.17 s
      [1 warm] FAIL -- not OOM: "Input type (CUDABFloat16Type) and weight type
                                 (CPUBFloat16Type) should be the same"

`image2video.py:956` calls `self.model.cpu()` after generation and the next
`generate()` never moves it back, so **every second request necessarily fails**.
Worked around here with an explicit `pipe.model.to(dev)` before each request
(`--ensure_device 1`).

### 2. `offload_model=False` OOMs at 304x528 with bf16

    bf16 + offload=False:  [0 cold] FAIL, global peak 7308 MiB, free 0
    fp8  + offload=False:  requests 0-3 ok (peak 7250-7370), [4] OOM, min free 0

So the real-time configuration (no offload) is not safe at 304x528.

## New production baseline at 304x528, bf16, real chain

    warm latency p50       26.54 s per 81-frame request
    throughput             ~3.06 output fps
    cold first request     ~76-88 s
    post-request reserved  plateau ~6050 MiB
    global peak            6580 (cold) -> ~7100-7200 (warm plateau)
    min free headroom      721 MiB
    allocated              731 MiB resident, flat

## Why this differs from the harness numbers

    harness (304x528, bf16):     ~1209 ms/chunk, previously called "production"
    real production chain:       26.54 s per 81 frames ~ 3.79 s/chunk

The difference is real work the harness omits: 4 timesteps rather than 3, a FULL
VAE decode rather than TAE-HV, and the whole-clip condition encode. So
"1209 ms/chunk" is a DiT-plus-TAE figure, not a complete production request.
Both are valid; they must not both be called "production latency".

## Ruling

304x528 passes the stability gate and becomes the **8 GB Deployment Geometry
Authority candidate**, conditional on carrying the offload device-restore fix.

The identity of the earlier P2b / M0 / M1 / S3 work is now unambiguous:
**"304x528 8 GB deployment profile evidence"**, not "production data at the
model's nominal geometry".
