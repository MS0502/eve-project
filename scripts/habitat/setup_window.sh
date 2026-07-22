#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

REPO_ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PRIVATE_ROOT="${EVE_M2E_PRIVATE_ROOT:-$HOME/.local/share/eve-m2e-window-private}"
BOOT_DIR="$HOME/.termux/boot"
BOOT_HOOK="$BOOT_DIR/eve-m2e-window.sh"
SUPERVISOR="$REPO_ROOT/scripts/habitat/supervisor.sh"

case "$REPO_ROOT" in
  /*) ;;
  *) echo "repository path must be absolute" >&2; exit 2 ;;
esac
case "$PRIVATE_ROOT" in
  /*) ;;
  *) echo "private companion path must be absolute" >&2; exit 2 ;;
esac

pkg update -y
pkg install -y python git termux-api
python -m pip install --upgrade pip
python -m pip install -r "$REPO_ROOT/requirements.txt"

mkdir -p "$PRIVATE_ROOT" "$PRIVATE_ROOT/backups" "$BOOT_DIR"
chmod 700 "$PRIVATE_ROOT" "$PRIVATE_ROOT/backups" "$BOOT_DIR"

python - "$REPO_ROOT" "$PRIVATE_ROOT" <<'PY'
from pathlib import Path
import sys
repo = Path(sys.argv[1]).resolve()
private = Path(sys.argv[2]).resolve()
try:
    private.relative_to(repo)
except ValueError:
    pass
else:
    raise SystemExit("private companion root must remain outside the repository")
PY

if ! git -C "$REPO_ROOT" check-ignore --no-index -q .eve-m2e-window/probe; then
  echo ".eve-m2e-window/ is not covered by repository ignore rules" >&2
  exit 3
fi
printf '%s\n' "outside_repo_private_root=$PRIVATE_ROOT" > "$PRIVATE_ROOT/git-exclusion-proof.txt"
printf '%s\n' "fallback_ignored=.eve-m2e-window/" >> "$PRIVATE_ROOT/git-exclusion-proof.txt"
printf '%s\n' "git_check_ignore=passed" >> "$PRIVATE_ROOT/git-exclusion-proof.txt"
chmod 600 "$PRIVATE_ROOT/git-exclusion-proof.txt"

cat > "$BOOT_HOOK" <<EOF
#!/data/data/com.termux/files/usr/bin/bash
export EVE_M2E_PRIVATE_ROOT=$(printf '%q' "$PRIVATE_ROOT")
exec $(printf '%q' "$SUPERVISOR") --boot
EOF
chmod 700 "$BOOT_HOOK"

termux-wake-lock || true
if [ -f "$PRIVATE_ROOT/supervisor.pid" ] && kill -0 "$(cat "$PRIVATE_ROOT/supervisor.pid")" 2>/dev/null; then
  echo "M2-E window supervisor already running"
else
  nohup "$SUPERVISOR" --boot >> "$PRIVATE_ROOT/supervisor.log" 2>&1 &
  echo "$!" > "$PRIVATE_ROOT/supervisor.pid"
  chmod 600 "$PRIVATE_ROOT/supervisor.pid" "$PRIVATE_ROOT/supervisor.log" 2>/dev/null || true
fi

"$REPO_ROOT/scripts/habitat/window_status.sh"
