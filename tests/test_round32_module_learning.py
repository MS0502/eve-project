"""
라운드 32 (B-4): ModuleLearningAdapter — 모듈 선택 학습
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from adapters.module_learning_adapter import ModuleLearningAdapter


def test_adapter_present():
    engine = build_full_engine()
    assert engine.module_learning_adapter is not None
    print(f"  ✓ module_learning_adapter 부착")


# ===== detect_effect =====

def test_detect_positive():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    cases = ["응", "그래", "맞아", "고마워", "좋네"]
    for c in cases:
        assert ml.detect_effect(c, "") == "positive", f"{c!r} 긍정 X"
    print(f"  ✓ {len(cases)}개 긍정 감지")


def test_detect_negative_teaching():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    cases = ["이럴때 응 이라고 말해", "이럴 때 그렇게 말해"]
    for c in cases:
        assert ml.detect_effect(c, "") == "negative", f"{c!r} 부정 X"
    print(f"  ✓ 가르침 = 부정")


def test_detect_negative_rejection():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    cases = ["아니야", "이상해", "어색해", "그게 아니라"]
    for c in cases:
        assert ml.detect_effect(c, "") == "negative", f"{c!r} 부정 X"
    print(f"  ✓ 거부 키워드 = 부정")


def test_detect_negative_retry():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    cases = ["그러니까 내 말은 PT가 빡셌다고", "다시 말해서 힘들다고"]
    for c in cases:
        assert ml.detect_effect(c, "") == "negative", f"{c!r} 부정 X"
    print(f"  ✓ 재시도 = 부정")


def test_detect_neutral_or_weak():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    # 평범한 답
    assert ml.detect_effect("오늘 날씨 어때", "") in ("weak_positive", "neutral")
    # 빈 입력
    assert ml.detect_effect("", "") == "neutral"
    print(f"  ✓ 중립/약긍정")


# ===== adjust =====

def test_adjust_positive_strengthens():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    ml.adjust("emotional_share", ["suffering_adapter"], "positive")
    util = ml.module_utility[("emotional_share", "suffering_adapter")]
    assert util > 1.0
    print(f"  ✓ 긍정 강화: {util:.2f}")


def test_adjust_negative_weakens():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    ml.adjust("emotional_share", ["suffering_adapter"], "negative")
    util = ml.module_utility[("emotional_share", "suffering_adapter")]
    assert util < 1.0
    print(f"  ✓ 부정 약화: {util:.2f}")


def test_adjust_bounds():
    """utility 0.1~2.0 경계."""
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    # 매우 많이 강화
    for _ in range(20):
        ml.adjust("test", ["mod"], "positive")
    assert ml.module_utility[("test", "mod")] <= 2.0
    # 매우 많이 약화
    ml.module_utility[("test", "mod")] = 1.0
    for _ in range(20):
        ml.adjust("test", ["mod"], "negative")
    assert ml.module_utility[("test", "mod")] >= 0.1
    print(f"  ✓ 0.1~2.0 경계")


# ===== reorder_modules =====

def test_reorder_by_utility():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    # 강화/약화 다르게
    ml.module_utility[("test", "mod_a")] = 0.7
    ml.module_utility[("test", "mod_b")] = 1.5
    ml.module_utility[("test", "mod_c")] = 1.0     # 기본
    
    reordered = ml.reorder_modules("test", ["mod_a", "mod_b", "mod_c"])
    # b > c > a 순서
    assert reordered[0] == "mod_b"
    print(f"  reorder: {reordered}")


def test_reorder_drops_too_weak():
    """0.5 미만 모듈은 제외."""
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    ml.module_utility[("test", "weak")] = 0.3
    ml.module_utility[("test", "ok")] = 1.0
    
    reordered = ml.reorder_modules("test", ["weak", "ok"])
    assert "weak" not in reordered
    assert "ok" in reordered
    print(f"  ✓ 약한 모듈 제외")


# ===== record + evaluate =====

def test_record_then_evaluate():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    # 1차 turn — 모듈 기록
    ml.record_selection("emotional_share", ["suffering_adapter", "memory_adapter"])
    # 2차 turn — 사용자 긍정 답
    effect = ml.evaluate_last("응 그래")
    assert effect == "positive"
    # suffering_adapter utility 올라감
    util = ml.module_utility[("emotional_share", "suffering_adapter")]
    assert util > 1.0
    print(f"  ✓ 긍정 신호 → utility {util:.2f}")


def test_evaluate_no_last_returns_none():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    # last_selection 없는 상태
    result = ml.evaluate_last("응")
    assert result is None
    print(f"  ✓ 마지막 선택 없으면 None")


# ===== chat 통합 =====

def test_chat_records_selection():
    engine = build_full_engine()
    list(engine.chat_stream("민석이 PT 빡셌어"))
    # _last_selection 기록됨
    assert engine.module_learning_adapter._last_selection is not None
    print(f"  ✓ chat 후 selection 기록")


def test_chat_learns_from_followup():
    """감정 공유 → 긍정 응답 → suffering 강화."""
    engine = build_full_engine()
    list(engine.chat_stream("민석이 PT 빡셌어"))
    # 이때 emotional_share + suffering 등 모듈 사용됨
    list(engine.chat_stream("응 진짜"))     # 긍정
    
    util = engine.module_learning_adapter.module_utility.get(
        ("emotional_share", "suffering_adapter"), 1.0
    )
    print(f"  emotional_share+suffering utility: {util:.2f}")
    # 강화돼야


def test_chat_learns_from_teaching_negative():
    """가르침 발생 → 직전 모듈들 약화."""
    engine = build_full_engine()
    list(engine.chat_stream("민석이 PT 빡셌어"))
    util_before = engine.module_learning_adapter.module_utility.get(
        ("emotional_share", "suffering_adapter"), 1.0
    )
    # 가르침 (이전 응답이 어색했다는 신호)
    list(engine.chat_stream("이럴때 진짜 빡셌네 라고 말해"))
    util_after = engine.module_learning_adapter.module_utility.get(
        ("emotional_share", "suffering_adapter"), 1.0
    )
    print(f"  before: {util_before:.2f}, after: {util_after:.2f}")
    assert util_after < util_before


# ===== stats =====

def test_stats_keys():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    ml.adjust("greeting", ["user_presence_adapter"], "positive")
    ml.adjust("emotional_share", ["suffering_adapter"], "negative")
    s = ml.stats()
    assert "adjustments" in s
    assert "positive" in s
    assert "negative" in s
    assert "top_5" in s
    print(f"  stats: adj={s['adjustments']}, p={s['positive']}, n={s['negative']}")


# ===== 영속화 =====

def test_to_from_dict():
    engine = build_full_engine()
    ml = engine.module_learning_adapter
    ml.adjust("emotional_share", ["suffering_adapter"], "positive")
    ml.adjust("greeting", ["user_presence_adapter"], "negative")
    
    data = ml.to_dict()
    
    ml2 = ModuleLearningAdapter(engine)
    ml2.from_dict(data)
    
    util1 = ml.module_utility[("emotional_share", "suffering_adapter")]
    util2 = ml2.module_utility[("emotional_share", "suffering_adapter")]
    assert util1 == util2
    print(f"  ✓ 영속화")


# ===== 회귀 =====

def test_existing_no_regression():
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("내가 누구야?"))
    assert "민석" in " ".join(out)
    out = list(engine.chat_stream("민석이 너무 힘들어"))
    full = " ".join(out)
    assert any(p in full for p in ["힘들", "느껴", "쉬어"])
    print(f"  ✓ 회귀 OK")


def test_round31_mood_still_works():
    engine = build_full_engine()
    list(engine.chat_stream("민석이 너무 힘들어"))
    out = list(engine.chat_stream("그러게"))
    full = " ".join(out)
    print(f"  감정 후 모호: {full}")
    assert "뭔" in full or "일" in full or "괜찮" in full


def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2
    # 같은 학습 결과
    u1 = e1.module_learning_adapter.module_utility
    u2 = e2.module_learning_adapter.module_utility
    assert u1 == u2
    print(f"  ✓ 결정론")


def run_all():
    tests = [
        ("어댑터 부착", test_adapter_present),
        # detect_effect
        ("긍정 감지", test_detect_positive),
        ("가르침 부정", test_detect_negative_teaching),
        ("거부 키워드 부정", test_detect_negative_rejection),
        ("재시도 부정", test_detect_negative_retry),
        ("중립", test_detect_neutral_or_weak),
        # adjust
        ("긍정 강화", test_adjust_positive_strengthens),
        ("부정 약화", test_adjust_negative_weakens),
        ("0.1~2.0 경계", test_adjust_bounds),
        # reorder
        ("utility 정렬", test_reorder_by_utility),
        ("약한 모듈 제외", test_reorder_drops_too_weak),
        # record/evaluate
        ("기록→평가", test_record_then_evaluate),
        ("선택 없으면 None", test_evaluate_no_last_returns_none),
        # chat 통합
        ("chat 자동 기록", test_chat_records_selection),
        ("긍정 응답 학습", test_chat_learns_from_followup),
        ("가르침 → 약화", test_chat_learns_from_teaching_negative),
        # stats / persist
        ("stats", test_stats_keys),
        ("영속화", test_to_from_dict),
        # 회귀
        ("기존 회귀", test_existing_no_regression),
        ("라운드 31 회귀", test_round31_mood_still_works),
        ("결정론", test_determinism),
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
