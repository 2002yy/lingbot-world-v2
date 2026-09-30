# RTX 5060 Research vs Deployment

The repository contains extensive experimental harnesses. They are valuable, but they are not all deployment authority.

## Research examples

Research harnesses may use:

- 320x480 or other non-deployment geometry;
- cached condition latents;
- TAE/lightweight decode instead of full deployment decode;
- isolated DiT timings;
- synthetic/proxy inputs;
- microbenchmarks.

Their output must say what it measured.

## Deployment authority

Deployment claims must come from the explicit deployment authority documented in:

- `ROADMAP_5060.md`
- `docs/deployment/RTX5060_8GB_CURRENT_AUTHORITY.md`
- `docs/deployment/DEPLOY_1_PRESET_FREEZE.md`

## Terminology

Preferred labels:

- `micro/kernel benchmark`
- `DiT + TAE benchmark profile`
- `full deployment request`
- `hot-path latency`
- `decode-complete proxy`
- `input-to-present` only when the present boundary is actually measured.

Avoid using "production" merely because it appears in a harness filename.
