"""
라운드 15: APCAdapter + CorpusAdapter (자율 학습)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine


def test_apc_observe_runs():
    """observe_and_learn 호출 가능."""
    engine = build_full_engine()
    from utils.types import Meaning
    m = Meaning(intent="emotion", entities=["PT", "운동"], emotions={"fatigue": 0.7})
    r = engine.apc_adapter.observe(m)
    print(f"  apc observe: {r}")


def test_apc_predict_returns_set():
    """predict_next_state 동작."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    sa.activate("PT", 0.6)
    sa.learn_pair("PT", "피로", strength=0.7)
    sa.learn_pair("PT", "운동", strength=0.6)
    pred = engine.apc_adapter.predict_next({"PT"}, focus="PT")
    print(f"  predict from PT: {pred}")
    assert isinstance(pred, set)


def test_corpus_learn_sentence():
    """문장 학습 → SA pair 생김."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    w_before = sa.get_weight("커피", "맛있다")
    r = engine.corpus_adapter.learn_text("커피는 정말 맛있다 아침에 마시면 좋다")
    w_after = sa.get_weight("커피", "맛있다")
    print(f"  pairs_made: {r.get('pairs_made')}, weight {w_before} → {w_after}")
    assert r.get("pairs_made", 0) > 0
    assert w_after > w_before


def test_corpus_auto_learn_in_chat():
    """chat_stream 자체가 corpus learning 트리거."""
    engine = build_full_engine()
    sa = engine.activation_adapter.sa
    w_before = sa.get_weight("커피", "아침")
    list(engine.chat_stream("커피는 아침에 마시면 좋다 정말 좋다"))
    w_after = sa.get_weight("커피", "아침")
    print(f"  weight {w_before} → {w_after}")
    # auto_learn이 페어 생성했어야


def test_corpus_discover_groups():
    """반복 학습 → 그룹 발견."""
    engine = build_full_engine()
    cl = engine.corpus_adapter
    # 같은 카테고리 셋 여러 번 학습
    for _ in range(4):
        cl.learn_text("PT는 힘들다 운동도 힘들다 피곤하다")
    for _ in range(4):
        cl.learn_text("커피는 맛있다 아침에 좋다 따뜻하다")
    groups = cl.discover()
    print(f"  groups: {dict((k, list(v)[:3]) for k, v in list(groups.items())[:3])}")


def test_corpus_long_document():
    """긴 문서 학습 — 단락별 분기."""
    engine = build_full_engine()
    long_text = (
        "PT는 정말 힘들다. 운동도 힘들다. 매일 가는 게 어렵다. "
        "그래도 가야 한다. 건강을 위해 운동은 중요하다. "
        "피곤할 때도 가야 한다. 그게 진짜 운동이다."
    )
    r = engine.corpus_adapter.learn_text(long_text)
    print(f"  document learn: {type(r).__name__}, keys={list(r.keys())[:5] if isinstance(r, dict) else '?'}")


def test_apc_corpus_in_full_chat():
    """진짜 시나리오 — 대화하면서 자동 학습 누적."""
    engine = build_full_engine()

    initial_categories = len(engine.activation_adapter.sa.activations)
    print(f"\n  초기 카테고리 수: {initial_categories}")

    list(engine.chat_stream("PT는 진짜 힘들다 매일 가야 한다"))
    list(engine.chat_stream("운동은 건강에 좋다 그래도 빡세다"))
    list(engine.chat_stream("커피 마시고 싶다 아침에 마시면 깨어난다"))

    final_categories = len(engine.activation_adapter.sa.activations)
    print(f"  3턴 후 카테고리 수: {final_categories}")
    assert final_categories > initial_categories, "카테고리 안 늘어남"

    # APC stats
    apc_stats = engine.apc_adapter.stats()
    print(f"  APC stats: {apc_stats}")

    # Corpus stats
    cl_stats = engine.corpus_adapter.stats()
    print(f"  Corpus stats: {cl_stats}")


def test_29_stack_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2, f"비결정론!\n{o1}\nvs\n{o2}"
    print(f"  ✓ 29스택 결정론")


def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    full = " ".join(out)
    assert "느껴져" in full or "지친" in full
    print(f"  ✓ 산술/환경/공감 회귀 없음")


def test_full_demo():
    """진짜 시연: 학습 → 그 카테고리로 응답."""
    engine = build_full_engine()

    print("\n  [학습] '집중력은 카페인에 영향 받는다'")
    r = engine.corpus_adapter.learn_text("집중력은 카페인에 영향 받는다")
    print(f"    → 카테고리 {r.get('pairs_made')}개 페어")

    print("\n  [응답] 카페인 뭐야?")
    for c in engine.chat_stream("카페인 뭐야?"): print(f"    {c}")


def run_all():
    tests = [
        ("APC observe", test_apc_observe_runs),
        ("APC predict", test_apc_predict_returns_set),
        ("Corpus 문장 학습", test_corpus_learn_sentence),
        ("Corpus auto (chat)", test_corpus_auto_learn_in_chat),
        ("Corpus 그룹 발견", test_corpus_discover_groups),
        ("Corpus 긴 문서", test_corpus_long_document),
        ("APC+Corpus 통합", test_apc_corpus_in_full_chat),
        ("29스택 결정론", test_29_stack_determinism),
        ("기존 회귀", test_existing_no_regression),
        ("전체 시연", test_full_demo),
    ]
    passed = 0
    failed = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ✗ ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            failed.append(name)
    print(f"\n{'='*40}")
    print(f"{passed}/{len(tests)} pass")
    if failed:
        print(f"실패: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
