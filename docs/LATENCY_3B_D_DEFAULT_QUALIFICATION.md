# Latency-3B-D: default qualification for preemption policy v1

    GATE     PASSES. Policy v1 is qualified to become the default.
    BASIS    three interleaved loads plus a faithful real-keyboard hold, all under a
             fixed cleanliness contract AND a thermal pair gate.
    NOTE     the sequential measurements that came before this were wrong, twice.

## The thermal pair gate, added because the first A/B needed it

The first attempt ran the arms sequentially and produced a `hold` "regression" of +87 ms
with ONE preemption in fifty chunks, which cannot be an effect of preemption. The cause was
the device warming: the first arm ran at 46 C / P8 and every later one at 65-69 C / P4.

So the contract now includes a second gate beside the VRAM one, and it **refuses** a pair
rather than explaining it afterwards:

    start temperature of A vs B must be within 6 C, or the pair is inadmissible
    P-state, power and utilisation recorded at the head and tail of every arm
    a 10 s sustained-GEMM warm-up establishes the envelope before the sequence, so
      round 1 is not automatically refused

One pair in the hold set was refused for a 7 C start difference. That is the gate working.

## Three loads, interleaved A B A B A B

    load    A median   B median   delta     A spread   B spread   waste
    normal    956       709      -247 ms     146 ms      31 ms    3.8-6.4%
    hold      849       858        +9 ms       4 ms       0 ms    0-0.7%
    rapid     892       886        -6 ms     126 ms     323 ms    20-23%

**The two sequential "results" are gone.** The hold regression of +87/+133 ms reduced to
+9 ms, and rapid's claimed -196 ms reduced to -6 ms. Both were the machine's drift, not
preemption -- and the interleaved hold spread of **4 ms and 0 ms** is what a real effect
being absent looks like.

## The default gate

    normal   clear positive gain                            PASS  -247 ms p50 (-26%)
    hold     no material regression                         PASS  +9 ms, +1.1%
    hold     no repeated or pointless preemption             PASS  28-30 same_state
                                                                  rejections, 0-1
                                                                  preemptions per run
    rapid    activity and correctness hold                  PASS  no starvation,
                                                                  no duplicate commit,
                                                                  <=1 rebase/chunk
    rapid    latency does not materially worsen             PASS  -6 ms
    all      no starvation                                  PASS
    all      <= 1 rebase per chunk                          PASS
    all      P0 rebase oracle intact                        PASS
    all      no-preemption reference intact                 PASS
    all      GPU cleanliness and thermal validity           PASS  every arm

## D2: the real keyboard, and why the synthetic hold was not it

A real held key delivers ONE KEYDOWN and then OS key-repeat KEYDOWNs with no KEYUP, and
`_handle_key` ignores a repeat of a key already held. The synthetic `hold` load posted
KEYUP before each KEYDOWN, which forced a repeat through and tested a load the product
never has.

D2 drives the faithful sequence -- hold W, release, hold D, release -- with OS repeat at
100 ms:

    intents sent            5      from four hold segments
    peeks with input        5
    admitted                2      exactly the W -> D and D -> W transitions
    rejected_too_late       3
    rejected_same_state     0      nothing to reject: suppression happened upstream
    preemption rate         2/35   waste 1.4%, no starvation, all gates PASS

**The real keyboard is cleaner than the synthetic hold**, because the repeats never reach
the admission policy at all. The answers to D2's five questions:

    does holding W cause pointless preemption?   No. Five intents, two preemptions, and
                                                 both at real transitions.
    does OS key-repeat cause repeated rebase?    No, and it cannot: the handler drops the
                                                 repeat before the policy sees it.
    does W_UP reliably reach authoritative state?   There is no W_UP intent in this
                                                 contract, by design. One keydown is one
                                                 discrete intent with a fixed integration
                                                 window, so the motion a press describes
                                                 is already complete and a release has
                                                 nothing to undo. Worth stating plainly
                                                 because a user may expect release to
                                                 stop; it does not, and that is the
                                                 frozen contract rather than a defect.
    does W -> D preempt normally?                Yes, both transitions admitted.
    starvation or repeated rebase?               None.

## §Latency-3B-C2 is parked, on purpose

The admission cost measured 82-121 ms and is **included** in every figure above rather than
netted out. With it fully counted, normal still nets about -247 ms. There is therefore no
case for re-entering the more dangerous generation and CUDA paths to shave tens of
milliseconds off a synchronise.

## What this still does not establish

- **Three rounds per cell.** The direction is consistent and the spreads are now small, but
  it remains a small sample.
- **One geometry, one weight, one seed, one machine.** 304x528 bf16, RTX 5060 Laptop 8GB.
- **Wasted work is not converted to power, heat or long-run throughput.** 20-23% of
  forwards are discarded under rapid; what that costs over an hour on a laptop is not
  measured.
- **Physical present is still not measured.** Every figure ends at the authoritative real
  frame or the renderer submit.

## Reproduce

    python gpu_cleanliness.py                 # fixed thresholds; refuses rather than tunes
    python demo_wasd.py --shakedown --load normal --arm A --preempt_boundaries ''
    python demo_wasd.py --shakedown --load normal --arm B --preempt_boundaries 1
    python demo_wasd.py --shakedown --load hold_real --hold_segment_s 4 --arm B
    python demo_wasd.py --shakedown --load rapid --arm B --preempt_boundaries 1
