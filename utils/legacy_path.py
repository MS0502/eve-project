"""
legacy/ 경로를 sys.path에 추가하는 헬퍼.
어댑터에서 한 줄로 끝내려고.

사용:
    from utils.legacy_path import enable
    enable()
    from hormone_system import HormoneSystem  # legacy/eve_modules/
"""

import os
import sys


def enable():
    """idempotent — 여러 번 호출해도 안전."""
    here = os.path.dirname(os.path.abspath(__file__))
    v41_root = os.path.dirname(here)
    candidates = [
        os.path.join(v41_root, "legacy", "eve_modules"),
        os.path.join(v41_root, "legacy", "v36_modules"),
    ]
    for p in candidates:
        if p not in sys.path and os.path.isdir(p):
            sys.path.insert(0, p)
