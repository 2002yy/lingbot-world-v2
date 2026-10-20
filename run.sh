#!/usr/bin/env bash
# LingBot-World 2.0 / 1.3B causal-fast on an RTX 5060 Laptop 8GB.
#
#   ./run.sh play --preset performance
#   ./run.sh play --preset lowmem
#   ./run.sh check
#
# The default is the frozen production stack: 304x528, bf16, FA2 repro path,
# streamed condition encode, the authoritative runtime, the variant D preview and a
# 50 ms handoff blend. Nothing else is recommended.
set -euo pipefail
cd "$(dirname "$0")"

PY="${LINGBOT_PYTHON:-$HOME/ai/lingbot-env/bin/python}"
export LINGBOT_MODE="${LINGBOT_MODE:-repro}"
export LINGBOT_STREAM_ENCODE="${LINGBOT_STREAM_ENCODE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

preset_args() {
  case "${1:-performance}" in
    performance) echo "--weight bf16" ;;
    lowmem)      echo "--weight fp8_lowmem" ;;
    *) echo "unknown preset '$1'; expected performance or lowmem" >&2; exit 2 ;;
  esac
}

cmd="${1:-play}"; shift || true
preset="performance"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --preset) preset="$2"; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done

case "$cmd" in
  play)
    # shellcheck disable=SC2046
    exec "$PY" play.py --pixel 304x528 --preview variant_d --blend_ms 50 \
      $(preset_args "$preset") "${args[@]}"
    ;;
  check)
    exec "$PY" release_check.py "${args[@]}"
    ;;
  smoke)
    exec "$PY" release_check.py --smoke "${args[@]}"
    ;;
  *)
    echo "usage: $0 {play|check|smoke} [--preset performance|lowmem] [args...]" >&2
    exit 2
    ;;
esac
