# Preview-History-Audit: what the old §35B / §35D assets actually are

The concern was real: a "train a tiny latent->RGB head" stage would have been
repeating work already done. This audit answers three questions with measurements
rather than recollection, and it changes the next step.

## 1. §35D / §36A: today's decoder IS that model

    taew2_1.pth      total 11.316 M   decoder 【9.845 M】   file 22.7 MB
    taew2_2.pth      total 11.418 M   decoder   9.924 M
    taehv.pth        total 11.316 M   decoder   9.845 M

§35D reported 9.84 M parameters. **9.845 M is that number.** And a tree-wide search
for "surrogate" returns zero hits -- there is no separate model.

So §35D's "latent->RGB surrogate" is `taew2_1.pth`, and `tae_realtime.py` (§36A)
loads exactly that checkpoint. **Today's 36.7 ms decoder is the §35D/§36A lineage.
There is no faster old asset to recover.**

## 2. The "~21 ms" does not reproduce

The same model, swept over latent geometries and frame counts:

    latent (C,T,H,W)      output frames      ms
    (16, 1, 66, 38)  304x528        1       36.41    <- today
    (16, 1, 60, 40)  320x480        1       32.37
    (16, 1, 36, 66)  528x288        1       31.99    <- §35B/§35D geometry
    (16, 3, 66, 38)                 9       88.52
    (16, 1, 33, 19)  half res       1        9.70
    (16, 1, 132, 76) double res     1      120.00

At §35D's OWN geometry the model takes 31.99 ms, not 21 ms. So the old figure was a
different workload or a different measurement basis, not a faster model. Recording
this matters because "the old one was 21 ms" would otherwise keep inviting a
recovery attempt that has nothing to recover.

## 3. §35B hooked a different, much more expensive decoder

    TAEHV has upsamples?  False
    preview_head.py hooks: vae.model.decoder.upsamples[7]   <- the FULL Wan VAE

TAEHV is a 23-child `nn.Sequential` with no `upsamples` at all. §35B's tiny head sat
on the **full Wan VAE decoder**, whose prefix to `upsamples.7` was documented as
88 ms -- more than twice today's 36 ms full TAEHV decode.

So §35B's route is strictly slower than what we already have, which is why TAEHV
superseded it. Its 0.228 ms head is real but it is attached to an 88 ms prefix.

Also worth noting: no head weights were saved anywhere in the tree. `preview_head.py`
trains its P0/P1 in-script each run, so there is no checkpoint to reload either.

## What the audit changes

Nothing to recover, but the audit produces a corrected synthesis.

The §35B idea -- a learned tiny tail on a decoder prefix -- is sound. What killed it
was the prefix, not the idea: 88 ms on the full VAE. **TAEHV's own prefix is far
cheaper**, and Preview-1B already measured it:

    stage 0 + stage 1 of TAEHV  =  4.99 + 5.85  =  10.84 ms
    (versus the full 36.01 ms)

So a tiny learned tail attached after **TAEHV stage 1** would cost ~11 ms plus the
tail, against the 26.67 ms that variant D costs by keeping stage 2's learned
convolutions. That is the version of §35B worth doing, and it is not what §35B
itself tested.

## Revised next step

§Preview-Head-1 remains justified, but its specification is now different and much
better grounded:

    input      TAEHV stage-1 features (after the second Upsample/TGrow/conv),
               at 4x spatial resolution
    target     the authoritative final frame
    budget     ~10.8 ms prefix + a tail small enough to stay under 15-20 ms total
    baseline   full TAEHV 36.01 ms, quality 0.8268 / 0.8003
    gate       quality preserved, session-start (chunks 1-2) not collapsing,
               authoritative penalty <= 3%

This is a genuinely different target from the earlier proposal, because the earlier
proposal would have trained a head on a prefix that costs more than the whole current
decode.

## The general lesson

Before building a new component, check whether the project already built it. Here the
parameter count settled it in one line: 9.845 M against a reported 9.84 M. That single
number prevented both a redundant training stage and a redundant recovery attempt --
and the same audit then showed where the old idea should have been attached all along.
