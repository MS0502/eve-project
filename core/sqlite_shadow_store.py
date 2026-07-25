"""M2-A shadow-store public surface.

A11 adds content-addressed persistence only. This is not a dual-read bridge and
it grants no recovery, cutover, production-default, or authority transition.
The frozen v1 implementation remains available solely for format compatibility.
"""
from core.sqlite_shadow_store_a11 import *  # noqa: F401,F403
