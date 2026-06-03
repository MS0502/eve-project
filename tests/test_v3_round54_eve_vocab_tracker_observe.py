"""EVE v3 round54 — EveVocabTracker observation implementation."""

from pathlib import Path

from adapters.eve_vocab_tracker import EveVocabTracker
from main import build_full_engine


def test_round54_observe_word_increments_count():
    tracker = EveVocabTracker()
    tracker.observe_word("EVE")
    assert tracker.get_observation_count("EVE") == 1
    tracker.observe_word("EVE")
    assert tracker.get_observation_count("EVE") == 2


def test_round54_observe_word_deterministic_count():
    tracker = EveVocabTracker()
    for _ in range(5):
        tracker.observe_word("민석")
    assert tracker.get_observation_count("민석") == 5
    assert tracker.stats()["observed_snapshot"] == {"민석": 5}


def test_round54_observe_word_with_context_records_log():
    tracker = EveVocabTracker()
    tracker.observe_word("EVE", {"source": "manual_test", "round": 54})
    stats = tracker.stats()
    assert stats["context_log_size"] == 1
    assert stats["context_snapshot"][0]["word"] == "EVE"
    assert stats["context_snapshot"][0]["context"] == {"source": "manual_test", "round": 54}
    assert stats["context_snapshot"][0]["observation_index"] == 1


def test_round54_observe_word_without_context_ok():
    tracker = EveVocabTracker()
    tracker.observe_word("EVE")
    assert tracker.get_observation_count("EVE") == 1
    assert tracker.stats()["context_log_size"] == 0


def test_round54_context_log_cap_1000():
    tracker = EveVocabTracker(context_cap=1000)
    for i in range(1005):
        tracker.observe_word(f"단어{i}", {"i": i})
    stats = tracker.stats()
    assert stats["context_log_size"] == 1000
    assert stats["context_snapshot"][0]["context"] == {"i": 5}
    assert stats["context_snapshot"][-1]["context"] == {"i": 1004}


def test_round54_is_eve_specific_unknown_in_fasttext():
    engine = build_full_engine()
    tracker = EveVocabTracker(engine=engine)
    tracker.observe_word("민석")
    assert tracker.is_eve_specific("민석") is True


def test_round54_is_eve_specific_known_in_fasttext():
    engine = build_full_engine()
    tracker = EveVocabTracker(engine=engine)
    tracker.observe_word("군대")
    assert tracker.is_eve_specific("군대") is False


def test_round54_is_eve_specific_unobserved_is_false():
    engine = build_full_engine()
    tracker = EveVocabTracker(engine=engine)
    assert tracker.is_eve_specific("민석") is False


def test_round54_get_observation_count_unknown_zero():
    tracker = EveVocabTracker()
    assert tracker.get_observation_count("없는단어") == 0


def test_round54_tracked_vocab_sorted_deterministic():
    tracker = EveVocabTracker()
    tracker.observe_word("민석")
    tracker.observe_word("EVE")
    tracker.observe_word("군대")
    assert tracker.tracked_vocab() == sorted(["민석", "EVE", "군대"])


def test_round54_eve_specific_vocab_filter():
    engine = build_full_engine()
    tracker = EveVocabTracker(engine=engine)
    for word in ["EVE", "민석", "군대", "코딩"]:
        tracker.observe_word(word)
    assert tracker.eve_specific_vocab() == ["EVE", "민석"]


def test_round54_stats_returns_data_dict():
    engine = build_full_engine()
    tracker = EveVocabTracker(engine=engine)
    tracker.observe_word("EVE")
    stats = tracker.stats()
    assert isinstance(stats, dict)
    assert stats["implementation_phase"] == "round54_observation"
    assert stats["observation_implemented"] is True
    assert stats["auto_observe_enabled"] is False
    assert stats["vector_generation_enabled"] is False
    assert stats["wrapper_integrated"] is False


def test_round54_no_auto_observe_call_from_wrapper():
    engine = build_full_engine()
    tracker = EveVocabTracker(engine=engine)
    before = tracker.stats()
    assert engine.self_embedding.get_vector("군대") is not None  # wrapper primary serves known medium token
    after = tracker.stats()
    assert after == before
    assert not hasattr(engine.self_embedding, "eve_vocab_tracker")


def test_round54_no_vector_generation_or_wrapper_integration():
    engine = build_full_engine()
    tracker = EveVocabTracker(engine=engine)
    tracker.observe_word("민석")
    assert tracker.stats()["vector_generation_enabled"] is False
    assert not hasattr(engine.self_embedding, "eve_specific_store")


def test_round54_no_random_used_in_tracker_module():
    text = Path("adapters/eve_vocab_tracker.py").read_text(encoding="utf-8")
    assert "random.random" not in text
    assert "np.random" not in text


def test_round54_report_records_observation_only():
    text = Path("ROUND_V3_R54_REPORT.md").read_text(encoding="utf-8")
    assert "round54" in text
    assert "observation" in text
    assert "Wrapper integration" in text
    assert "Not done" in text
