"""EVE v3 round56 — wrapper eve-specific integration.

Round56 inserts the EveSpecificVectorStore between the fastText primary and
the PMI+SVD fallback in the EmbeddingWrapper. This test suite verifies
that the wrapper consults the eve-specific store prior to falling back and
records telemetry accordingly. It also ensures that stats and state debug
reflect the new eve-specific layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from main import build_full_engine


def test_wrapper_eve_specific_priority_returns_eve_vector() -> None:
    """When a vector exists in the eve-specific store, wrapper should return it."""
    engine = build_full_engine()
    # Prepare an explicit operator-authorized eve-specific vector for a word
    # not present in the no-load fastText primary. This avoids default loading
    # or fabricated setup-side commits.
    store = engine.eve_specific_vector_store
    assert engine.fasttext_embedding.is_loaded() is False
    store.add_vector("민석", np.ones(300, dtype=np.float32))
    wrapper = engine.self_embedding
    # Clear previous telemetry counters
    _ = wrapper.telemetry()
    # Now lookup the eve-specific word via the wrapper
    vec = wrapper.get_vector("민석")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (300,)
    # Telemetry should record one eve-specific hit and no primary or fallback hits
    telem = wrapper.telemetry()
    assert telem.get("eve_specific_hits", 0) == 1
    assert telem.get("primary_hits", 0) == 0
    assert telem.get("fallback_uses", 0) == 0


def test_wrapper_fallback_when_eve_specific_missing() -> None:
    """If a word is not found in fastText or eve-specific store, fallback is used."""
    engine = build_full_engine()
    wrapper = engine.self_embedding
    before = wrapper.telemetry()
    # Choose a nonsense token unlikely to exist in fastText or eve-specific store
    unknown_word = "없음없는단어"
    result = wrapper.get_vector(unknown_word)
    # Result may be None or a 50d fallback embedding; ensure telemetry increments fallback
    after = wrapper.telemetry()
    assert after.get("fallback_uses", 0) == before.get("fallback_uses", 0) + 1
    # Eve-specific hits should not increment
    assert after.get("eve_specific_hits", 0) == before.get("eve_specific_hits", 0)


def test_wrapper_stats_and_state_include_eve_specific() -> None:
    """Wrapper stats and state debug must expose eve-specific fields."""
    engine = build_full_engine()
    store = engine.eve_specific_vector_store
    wrapper = engine.self_embedding
    stats = wrapper.stats()
    # eve-specific class and dimension should be present
    assert stats.get("eve_specific_class") == store.__class__.__name__
    # dimension may be non-zero only when store is configured
    assert stats.get("eve_specific_dimension", 0) == store.dimension
    # State debug snapshot should report eve-specific dimension and counts
    state = engine.state_debug.snapshot_state()
    assert state["self_embedding"].get("eve_specific_dimension", 0) == store.dimension
    assert isinstance(state["self_embedding"].get("eve_specific_count", 0), int)
    # Wrapper telemetry in state debug should include eve-specific rates
    wrapper_tele = state["wrapper_telemetry"]
    assert "eve_specific_rate" in wrapper_tele
    assert "eve_specific_hits" in wrapper_tele