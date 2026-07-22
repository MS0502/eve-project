#!/data/data/com.termux/files/usr/bin/bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIVATE_ROOT="${EVE_M2E_PRIVATE_ROOT:-$HOME/.local/share/eve-m2e-window-private}"
RUNTIME="$REPO_ROOT/scripts/habitat/m2_e_window_runtime.py"
BOOT_PENDING=0
[ "${1:-}" = "--boot" ] && BOOT_PENDING=1

mkdir -p "$PRIVATE_ROOT"
chmod 700 "$PRIVATE_ROOT"
if [ -f "$PRIVATE_ROOT/supervisor.pid" ]; then
  EXISTING_PID="$(cat "$PRIVATE_ROOT/supervisor.pid" 2>/dev/null || true)"
  if [ -n "$EXISTING_PID" ] && [ "$EXISTING_PID" != "$$" ] && kill -0 "$EXISTING_PID" 2>/dev/null; then
    exit 0
  fi
fi
echo "$$" > "$PRIVATE_ROOT/supervisor.pid"
chmod 600 "$PRIVATE_ROOT/supervisor.pid"
trap 'rm -f "$PRIVATE_ROOT/supervisor.pid"; exit 0' TERM INT EXIT
termux-wake-lock || true

while true; do
  STATUS="$(python "$RUNTIME" --private-root "$PRIVATE_ROOT" status 2>/dev/null || true)"
  case "$STATUS" in
    health=sealed*|health=frozen*)
      sleep 60
      continue
      ;;
  esac

  if [ "$BOOT_PENDING" -eq 1 ]; then
    python "$RUNTIME" --private-root "$PRIVATE_ROOT" run --boot
    CODE=$?
    BOOT_PENDING=0
  else
    python "$RUNTIME" --private-root "$PRIVATE_ROOT" run
    CODE=$?
  fi

  STATUS="$(python "$RUNTIME" --private-root "$PRIVATE_ROOT" status 2>/dev/null || true)"
  case "$STATUS" in
    health=sealed*|health=frozen*)
      sleep 60
      ;;
    *)
      # No intentional kill schedule exists here. Any non-terminal exit is
      # treated as a real process death; the persistent worker marker makes
      # the next start verify the recovery digest before resuming.
      printf '%s code=%s status=%s\n' "$(date -Iseconds)" "$CODE" "$STATUS" >> "$PRIVATE_ROOT/supervisor.log"
      sleep 5
      ;;
  esac
done
