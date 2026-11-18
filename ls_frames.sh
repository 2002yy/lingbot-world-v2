#!/usr/bin/env bash
cd "$HOME/ai/lingbot-world-v2/output" || exit 1
for d in push angular transrot reloc dyn crecheck commit2 d4 d5 rot destroy; do
  if [ -d "$d" ]; then
    n=$(ls "$d"/*_frames.npy 2>/dev/null | wc -l)
    printf '%-10s %2s sets : ' "$d" "$n"
    ls "$d"/*_frames.npy 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '
    printf '\n'
  fi
done
