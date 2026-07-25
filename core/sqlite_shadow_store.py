"""M2-A shadow-store public surface.

A11 adds content-addressed persistence only when growing habitat material
actually reaches the frozen canonical boundary. This is not a dual-read bridge
and it grants no recovery, cutover, production-default, or authority transition.
The accepted v1 representation remains exact for smaller material.
"""
from core.sqlite_shadow_store_lazy_a11 import *  # noqa: F401,F403
