#!/data/data/com.termux/files/usr/bin/bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIVATE_ROOT="${EVE_M2E_PRIVATE_ROOT:-$HOME/.local/share/eve-m2e-window-private}"
RUNTIME="$REPO_ROOT/scripts/habitat/m2_e_window_runtime.py"
SUPERVISOR_PATH="$REPO_ROOT/scripts/habitat/supervisor.sh"
LOG="$PRIVATE_ROOT/supervisor.log"
BOOT_PENDING=0
[ "${1:-}" = "--boot" ] && BOOT_PENDING=1

is_live_supervisor_pid() {
  local candidate_pid="${1:-}"
  local cmdline=""
  [ -n "$candidate_pid" ] || return 1
  kill -0 "$candidate_pid" 2>/dev/null || return 1
  [ -r "/proc/$candidate_pid/cmdline" ] || return 1
  cmdline="$(tr '\000' ' ' < "/proc/$candidate_pid/cmdline" 2>/dev/null || true)"
  case "$cmdline" in
    *"$SUPERVISOR_PATH"*) return 0 ;;
    *) return 1 ;;
  esac
}

mkdir -p "$PRIVATE_ROOT"
chmod 700 "$PRIVATE_ROOT"
touch "$LOG"
chmod 600 "$LOG"
if [ -f "$PRIVATE_ROOT/supervisor.pid" ]; then
  EXISTING_PID="$(cat "$PRIVATE_ROOT/supervisor.pid" 2>/dev/null || true)"
  if [ "$EXISTING_PID" != "$$" ] && is_live_supervisor_pid "$EXISTING_PID"; then
    exit 0
  fi
  rm -f "$PRIVATE_ROOT/supervisor.pid"
fi
echo "$$" > "$PRIVATE_ROOT/supervisor.pid"
chmod 600 "$PRIVATE_ROOT/supervisor.pid"
trap 'rm -f "$PRIVATE_ROOT/supervisor.pid"; exit 0' TERM INT EXIT
termux-wake-lock || true
printf '%s supervisor_start pid=%s boot=%s\n' "$(date -Iseconds)" "$$" "$BOOT_PENDING" >> "$LOG"

while true; do
  STATUS="$(python "$RUNTIME" --private-root "$PRIVATE_ROOT" status 2>>"$LOG" || true)"
  case "$STATUS" in
    health=sealed*|health=frozen*)
      sleep 60
      continue
      ;;
  esac

  if [ "$BOOT_PENDING" -eq 1 ]; then
    python "$RUNTIME" --private-root "$PRIVATE_ROOT" run --boot >>"$LOG" 2>&1
    CODE=$?
    BOOT_PENDING=0
  else
    python "$RUNTIME" --private-root "$PRIVATE_ROOT" run >>"$LOG" 2>&1
    CODE=$?
  fi

  STATUS="$(python "$RUNTIME" --private-root "$PRIVATE_ROOT" status 2>>"$LOG" || true)"
  case "$STATUS" in
    health=sealed*|health=frozen*)
      printf '%s code=%s status=%s\n' "$(date -Iseconds)" "$CODE" "$STATUS" >> "$LOG"
      sleep 60
      ;;
    *)
      # No intentional kill schedule exists here. Any non-terminal exit is
      # treated as a real process death; the persistent worker marker makes
      # the next start verify the recovery digest before resuming.
      printf '%s code=%s status=%s\n' "$(date -Iseconds)" "$CODE" "$STATUS" >> "$LOG"
      sleep 5
      ;;
  esac
done
