#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PRIVATE_ROOT="${EVE_M2E_PRIVATE_ROOT:-$HOME/.local/share/eve-m2e-window-private}"
exec python "$REPO_ROOT/scripts/habitat/m2_e_window_runtime.py" --private-root "$PRIVATE_ROOT" status
