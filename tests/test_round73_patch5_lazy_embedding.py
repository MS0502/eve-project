from main import build_full_engine


def test_self_embedding_does_not_bootstrap_during_engine_build():
    engine = build_full_engine()
    stats = engine.self_embedding.stats()

    assert stats["train_count"] == 0
    assert stats["trained_size"] == 0
    assert stats["bootstrap_attempted"] is False


def test_self_embedding_ensure_ready_lazy_bootstraps_once_from_concepts():
    engine = build_full_engine()

    # 반복 정의어를 넣어서 min_word_freq=2 조건을 넘긴다.
    engine.concept_memory.learn("운동", "건강 체력 근육 반복 훈련")
    engine.concept_memory.learn("헬스", "건강 체력 근육 반복 훈련")
    engine.concept_memory.learn("PT", "건강 체력 근육 반복 훈련")

    assert engine.self_embedding.ensure_ready() is True
    first = engine.self_embedding.stats()
    assert first["bootstrap_attempted"] is True
    assert first["train_count"] == 1
    assert first["trained_size"] >= 5

    assert engine.self_embedding.ensure_ready() is True
    second = engine.self_embedding.stats()
    assert second["train_count"] == first["train_count"]
    assert second["trained_size"] == first["trained_size"]


def test_embedding_consumers_trigger_lazy_bootstrap_without_rebuilding():
    engine = build_full_engine()
    engine.concept_memory.learn("운동", "건강 체력 근육 반복 훈련")
    engine.concept_memory.learn("헬스", "건강 체력 근육 반복 훈련")
    engine.concept_memory.learn("PT", "건강 체력 근육 반복 훈련")

    assert engine.self_embedding.stats()["bootstrap_attempted"] is False

    # Compositor의 embedding path가 ensure_ready()를 직접 호출해야 한다.
    engine.compositor._compose_embedding("건강", "근육", top_n=3)

    stats = engine.self_embedding.stats()
    assert stats["bootstrap_attempted"] is True
    assert stats["train_count"] == 1
