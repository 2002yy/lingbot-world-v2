#!/usr/bin/env bash
# Dependency and asset check. It REPORTS; it does not install GPU stacks for you,
# because the exact torch/CUDA combination is hardware specific and a silent
# auto-install is how people end up with a mismatched build.
set -uo pipefail
cd "$(dirname "$0")"

PY="${LINGBOT_PYTHON:-$HOME/ai/lingbot-env/bin/python}"
CKPT="${LINGBOT_CKPT:-$HOME/ai/models/lingbot-world-v2-1.3b-causal-fast}"
ASSETS="${LINGBOT_ASSETS:-$HOME/ai/models/lingbot-shared-assets}"
TAE="${LINGBOT_TAE:-$HOME/ai/taehv/taew2_1.pth}"

fail=0
say() { printf '  %-42s %s\n' "$1" "$2"; }

echo "== python =="
if [ -x "$PY" ]; then say "interpreter" "$PY"; else say "interpreter" "MISSING: $PY"; fail=1; fi

if [ -x "$PY" ]; then
  echo "== packages =="
  "$PY" - <<'PYEOF'
import importlib, sys
want = [("torch","torch"),("torchvision","torchvision"),("numpy","numpy"),
        ("einops","einops"),("PIL","pillow"),("scipy","scipy"),
        ("diffusers","diffusers"),("transformers","transformers"),
        ("torchao","torchao"),("cv2","opencv-python")]
bad = []
for mod, pkg in want:
    try:
        importlib.import_module(mod)
        print(f"  {'ok':<8} {pkg}")
    except Exception as e:
        print(f"  {'MISSING':<8} {pkg}  ({type(e).__name__})")
        bad.append(pkg)
if bad:
    print("  install with: pip install " + " ".join(bad))
    sys.exit(3)
PYEOF
  [ $? -ne 0 ] && fail=1

  echo "== cuda =="
  "$PY" - <<'PYEOF'
import torch
print(f"  torch {torch.__version__}  cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  device {p.name}  {p.total_memory/2**30:.2f} GiB  sm_{p.major}{p.minor}")
    if p.total_memory/2**30 < 7.0:
        print("  WARNING: below 7 GiB; the lowmem preset may be required")
else:
    print("  FAIL: no CUDA device")
    raise SystemExit(4)
PYEOF
  [ $? -ne 0 ] && fail=1
fi

echo "== model assets =="
for p in "$CKPT" "$ASSETS"; do
  if [ -d "$p" ]; then say "$(basename "$p")" "present"; else say "$(basename "$p")" "MISSING: $p"; fail=1; fi
done
if [ -f "$TAE" ]; then say "taew2_1.pth" "present"; else say "taew2_1.pth" "MISSING: $TAE"; fail=1; fi

echo "== example scenes =="
n=$(ls -d examples/*/ 2>/dev/null | grep -Ev '_(official|static|yaw|outback|pilot|M)' | wc -l)
say "example scene directories" "$n"
if [ "$n" -eq 0 ]; then say "example scene directories" "NONE: play needs examples/04"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo "setup check: PASS"; else echo "setup check: FAIL (see above)"; fi
exit $fail
