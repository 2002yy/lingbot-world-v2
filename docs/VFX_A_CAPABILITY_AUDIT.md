# §VFX-A: capability audit for NVIDIA Video Effects (VFG + VSR)

    STATUS   FEASIBLE on this GPU, and on Linux as well as Windows.
             BLOCKED on one thing: NGC credentials, which only the machine's owner can create.
    METHOD   every claim below is from the official documentation fetched on 2026-10-16, not
             from a summary. Where the docs are silent, this says so.

## 1. Is the hardware supported? Yes, and on Linux too

The VFG page, "Hardware Requirements":

> Windows and Linux x86-64: NVIDIA Ada and Blackwell architecture GPUs.
> Linux x86-64 additionally supports NVIDIA Hopper architecture GPUs.

RTX 5060 Laptop is Blackwell, compute capability **120**, and the install page lists supported
compute capabilities as **75, 86, 89, 120, 121**. So the device is supported, and -- the part
that changes the plan -- **this is not a Windows-only capability.** The index page does say the
Linux SDK "is designed and optimized for server-side (datacenter or cloud) deployments", but
that is a statement about intent, not about supported hardware. **The existing WSL path is not
excluded by the documentation**, which matters because the whole world model already runs there.

Local evidence:

    GPU            NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB, compute capability 120
    driver         591.86  (VSR on Linux requires 570.190+ / 580.82+ / 590.44+  -> satisfied)
    CUDA           13.1
    display_active Disabled   <- the NVIDIA GPU does not drive the laptop panel

That last line is the one thing to keep in mind: the NVIDIA GPU reports `Displays: Disabled`,
so it is a compute device here, not the display output. No filter in the VFG/VSR documentation
states a display requirement, but it has not been proven absent either, and it is the first
thing a smoke test should establish.

## 2. The blocker: NGC credentials

From the install page:

- the SDK Core is distributed as a `.zip` through the **NGC catalog**, behind a sign-in;
- installing any feature runs `install_feature.ps1`, which **requires `NGC_CLI_API_KEY`**, a
  personal API key generated from an NGC account.

Checked on this machine:

    ~/.ngc/config                  absent
    C:\Users\Zhang\.ngc\config     absent
    NGC_CLI_API_KEY                unset
    NGC_API_KEY                    unset
    ngc CLI                        not installed
    existing Maxine / VFX SDK      none
    nvvfx / tensorrt / pycuda      absent

**I cannot create an NGC account and will not ask for credentials.** So the acquisition is a
two-command step that only the machine's owner can run:

```powershell
# 1) download the SDK Core .zip from the NGC catalog (sign-in required)
#    https://catalog.ngc.nvidia.com/orgs/nvidia/teams/maxine/collections/maxine_vfx_sdk
#    then extract it somewhere, e.g. C:\vfx

# 2) install the two features this stage needs
Set-ExecutionPolicy Bypass -Scope Process
cd C:\vfx\VFXSDK\features\
$Env:NGC_CLI_API_KEY = "<your key>"
.\install_feature.ps1 -gpu 120 -features nvvfxvideosuperres
.\install_feature.ps1 -gpu 120 -features <framegen feature name>
```

The install page's example feature list shows `nvvfxvideosuperres`, `nvvfxdenoising`,
`nvvfxupscale`, `nvvfxgreenscreen`, `nvvfxbackgroundblur`, `nvvfxrelighting` -- **but not the
frame-generation feature name**, and it says the list is non-exhaustive. Use
`.\install_feature.ps1 -list_features` to get the real name rather than guessing it.

Windows tooling: Visual Studio 18 is installed, so a native x64 test program is buildable.
`cl.exe`/`cmake` were not found by a shallow search, so the developer prompt may be needed.

## 3. Three findings that change the earlier plan

### 3a. VFG does not scale, so VSR and VFG cannot be one stage

> VFG does not perform spatial scaling. The previous frame, current frame, and generated frame
> must have the same dimensions and use the same encoding family.

So the two filters are separate and their order is a real decision with a real cost difference:

    (a) VFG at native 304x528, then VSR on every frame INCLUDING the interpolated ones
        -> VSR runs `multiplier` times more often
    (b) VSR first (304x528 -> larger), then VFG on the upscaled frames
        -> VFG runs at `multiplier` times the pixels

The documentation does not say which is cheaper, and the answer depends on the two filters'
relative cost at each resolution -- which is exactly what a measurement should settle. This is
the first thing §VFX-B should measure, because it changes everything downstream.

### 3b. The shot-change guard does NOT cover our risk

VFG provides `NVVFXVIDEOFRAMEGENERATION_SHOT_CHANGE`, and automatic shot-change detection is on
by default. When it fires, VFG bypasses interpolation and copies the current frame.

That is a good guard against a **cut**, which is not our failure mode. A world model is
continuous: our risk is **fast camera motion, occlusion and newly revealed geometry**, where the
two frames are legitimately the same shot and interpolation will happily produce stretching or
ghosting. **Automatic detection will not fire for those**, and the earlier plan's hope that the
API "supports skipping on discontinuities" does not extend to the case that matters here.

So §VFX-B has to measure artifacts under exactly the conditions the API will not protect:
fast turns, and scene changes the model itself generates. It can also use the explicit
`Timestep` mode to place an intermediate frame at the position where the artifact is worst,
rather than only at `t=0.5`.

### 3c. VSR's recommended input floor is above our deployment geometry

> The recommended minimum input resolution for VSR is 360p.

Our deployment authority is 304x528, whose short side is **304** -- below the recommendation.
The doc says "recommended", not "required", so it may work; it may also produce artifacts at a
size the model was not tuned for. This is a measurement, not a guess, and it is the second
thing §VFX-B should establish.

Also worth knowing: the streaming modes (21, 23) need Ampere or newer, so they are available to
us; the local driver 591.86 already satisfies the Linux requirement for VSR.

## 4. What the published numbers do and do not tell us

The performance reference, at **1280x720** input and output:

    VFG Medium, 2x multiplier       RTX 4090 1.32 ms   RTX 5090 1.90 ms   RTX Spark 3.16 ms
    VSR Streaming Medium (21)       RTX 4090 0.688 ms  RTX 5090 0.642 ms  RTX Spark 2.00 ms
    VSR Streaming Ultra (23)        RTX 4090 2.01 ms   RTX 5090 1.67 ms   RTX Spark 6.36 ms
    Upscale                         RTX 4090 0.056 ms  RTX 5090 0.049 ms  RTX Spark 0.183 ms

**There is no RTX 5060 Laptop column**, and the nearest Blackwell reference is RTX Spark, which
is a different class of part. So:

- the numbers cannot be transferred to this laptop, and this document does not transfer them;
- but they are per **single intermediate frame** at **720p**. Our geometry is 304x528, about
  **4.7x fewer pixels**. A 4x multiplier needs **three sequential calls** per frame pair, and
  VFG supports batch size 1 only, so the cost is serial. That is a reason to expect the display
  cost here to be small rather than a reason to claim a figure.

The arithmetic in the earlier proposal -- 3.1 FPS of generation becomes 6.2 / 12.4 / 24.8 FPS of
display at 2x / 4x / 8x -- is an upper bound on the **display** rate and says nothing about how
often the world model responds. A world model that answers WASD three times a second does not
answer it twenty-five times a second because the display was interpolated. That distinction has
to survive into anything published.

## 5. Verdict

    feasible on this GPU                      YES  (Blackwell sm_120, driver sufficient)
    feasible on Linux / WSL, not Windows-only  YES, per the hardware requirements
    obtainable without an NGC account          NO   <- the blocker
    VFG's guard covers our actual risk         NO   (shot change is not fast motion)
    VSR at 304 px short side                   UNKNOWN, measured below the recommendation
    cost at our resolution                     NOT MEASURABLE FROM THE DOCS

**The next step is not a design, it is an acquisition.** Until the SDK Core and its two features
are on the machine, §VFX-B cannot start, and I cannot do that part.

## 6. What §VFX-B should be, once the SDK is present

    B0  smoke            load VFG and VSR, run once, report device, driver, feature name and
                         version, and whether the laptop panel state (Displays: Disabled)
                         matters. Establishes obtainability, not performance.
    B1  order            (a) VFG-then-VSR versus (b) VSR-then-VFG, at 2x, on real generated
                         frames from output/ -- which order is cheaper and which looks better
    B2  cost             per-intermediate-frame latency and VRAM for 2x / 4x / 8x, model modes
                         Low / Medium / High, measured at 304x528, with the world model ALSO
                         resident, because 8 GB is the gate that decides whether any of this
                         ships
    B3  artifacts        fast turns and model-generated scene changes, which is where the
                         API's shot-change guard does not help, plus explicit Timestep
                         placement instead of t=0.5 only
    B4  isolation        prove the display path cannot reach world state: interpolated and
                         super-resolved frames must never enter the KV cache, the committed
                         camera, or the generation input

B4 is not optional and is the least negotiable item in the list. This project has already
established that small numerical differences compound over long horizons; a display layer that
could feed back would be a new source of exactly that, and the whole point of putting it outside
the model is that it cannot.

**Nothing in §VFX-B may change DiT, KV, RNG, the commit path or the preemption policy.** rc2 at
`a875a0d` is the frozen baseline and stays that way.
