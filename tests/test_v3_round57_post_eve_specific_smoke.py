"""EVE v3 round57 — post eve-specific smoke and drift baseline measurement.

This test suite verifies that after integrating the EveSpecificVectorStore into the
embedding wrapper (round56), a smoke run and drift baseline measurement capture
eve-specific telemetry. The goals are:

* Telemetry should expose eve-specific rates and hit counts even when no vectors
  have been learned yet.
* measure_seed_drift_baseline() should report the EveSpecificVectorStore as the
  learning substrate once the store is present.
* After manually adding an Eve-specific vector, smoke sampling should record
  eve-specific hits and a non-zero eve-specific rate.

The tests do not enable any automatic observation or promotion. They are
strictly read-only data collection surfaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from main import build_full_engine
from adapters.runtime_smoke_runner import run_conversation_smoke
# For round57, use the new drift baseline measurement that accounts for the
# EveSpecificVectorStore. The original measure_seed_drift_baseline remains
# unchanged for round42 baseline compatibility. Import the round57 version here.
from adapters.external_seed_manifest import measure_eve_specific_drift_baseline
from tests.fixtures.korean_conversation_fixtures import fixture_texts


def test_round57_wrapper_telemetry_includes_eve_specific_rate_and_hits() -> None:
    """Telemetry must expose eve-specific rate and hit count even with no vectors."""
    engine = build_full_engine()
    wrapper = engine.self_embedding
    # invoke one vector lookup on a known fastText word to initialize counters
    wrapper.get_vector(fixture_texts()[0])
    telem = wrapper.telemetry()
    # Telemetry should include eve-specific fields
    assert "eve_specific_rate" in telem
    assert "eve_specific_hits" in telem
    # Without any EVE-specific vectors, eve_specific hits should be zero
    assert telem.get("eve_specific_hits", 0) == 0
    # Rate should be defined and non-negative
    assert telem.get("eve_specific_rate", 0.0) >= 0.0


def test_round57_drift_baseline_includes_eve_specific_substrate_and_rate() -> None:
    """Drift baseline must reflect the EveSpecificVectorStore once present."""
    engine = build_full_engine()
    # Manually add an Eve-specific vector to ensure the store is non-empty
    store = engine.eve_specific_vector_store
    context_word = "안녕"
    assert engine.fasttext_embedding.get_vector(context_word) is not None
    added = store.add_or_update_vector("민석", [context_word], engine=engine)
    assert added is True
    # Use the round57 drift baseline to capture eve-specific substrate and rates
    report = measure_eve_specific_drift_baseline(engine)
    # After the eve-specific store is wired, the substrate should reference it
    assert report["evespecific_learning_substrate"].startswith("EveSpecificVectorStore")
    # Drift indicators should include eve-specific rate
    drift = report["drift_indicators"]
    assert "eve_specific_rate" in drift
    assert drift["eve_specific_rate"] >= 0.0


def test_round57_smoke_reports_eve_specific_hits_after_vector_added() -> None:
    """Smoke sampling should record eve-specific hits when vectors exist."""
    engine = build_full_engine()
    store = engine.eve_specific_vector_store
    context_word = "안녕"
    # Add deterministic Eve-specific vectors for both EVE-specific words used in fixtures
    store.add_or_update_vector("민석", [context_word], engine=engine)
    store.add_or_update_vector("EVE", [context_word], engine=engine)
    result = run_conversation_smoke(engine, fixture_texts())
    telem = result.get("wrapper_telemetry", {})
    # At least one eve-specific hit is expected because the fixtures include "민석" and "EVE"
    assert telem.get("eve_specific_hits", 0) >= 1
    # The eve-specific rate should be greater than zero
    assert telem.get("eve_specific_rate", 0.0) > 0.0