"""
라운드 35 (C-3): SafetyAdapter — 4 안전장치
- Confidence
- Decay
- Conflict
- Exploration
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.safety_adapter import (
    SafetyAdapter, Belief,
    CONFIDENCE_THRESHOLD, CONFIDENCE_HIGH,
    DECAY_RATE_PER_DAY, DECAY_FLOOR,
    EXPLORATION_THRESHOLD,
)


# ===== 1. Confidence =====

def test_confidence_low():
    s = SafetyAdapter()
    # 0.3 < 0.5 → 응답 후보 반환
    r = s.check_confidence("ambiguous", 0.3)
    assert r is not None, "낮은 confidence 시 응답 있어야"
    assert "다시" in r or "잘 모르" in r or "다른 말" in r
    print(f"  ✓ 낮은 confidence (0.3) → {r!r}")


def test_confidence_high():
    s = SafetyAdapter()
    r = s.check_confidence("teaching", 0.9)
    assert r is None, "높은 confidence는 None"
    print(f"  ✓ 높은 confidence (0.9) → None (자신있게 처리)")


def test_confidence_response_variety():
    """반복 호출 시 다른 응답 (단조로움 방지)."""
    s = SafetyAdapter()
    responses = []
    for _ in range(5):
        r = s.check_confidence("x", 0.2)
        responses.append(r)
    unique = len(set(responses))
    assert unique > 1, "다양한 응답이어야"
    print(f"  ✓ 5번 호출 → {unique}가지 다른 응답")


def test_is_confident():
    s = SafetyAdapter()
    assert s.is_confident(0.9) is True
    assert s.is_confident(0.7) is False
    assert s.is_confident(0.5) is False
    print(f"  ✓ is_confident threshold ({CONFIDENCE_HIGH})")


# ===== 2. Decay =====

def test_decay_first_run():
    """첫 호출 — last_decay_run 초기화로 곧바로 안 됨."""
    s = SafetyAdapter()
    result = s.apply_decay()
    assert result is False, "첫 호출은 24시간 안 지났으니 skip"
    print(f"  ✓ 첫 호출 skip")


def test_decay_force():
    """force=True면 즉시 실행."""
    s = SafetyAdapter()
    result = s.apply_decay(force=True)
    assert result is True
    assert s.decay_runs == 1
    print(f"  ✓ force decay 실행")


def test_decay_floor():
    """floor 아래로 떨어지지 않게."""
    # 단순 검증 — DECAY_FLOOR 값
    assert 0 < DECAY_FLOOR < 1
    print(f"  ✓ floor = {DECAY_FLOOR} (완전 잊기 방지)")


# ===== 3. Conflict =====

def test_belief_add_no_conflict():
    s = SafetyAdapter()
    s.add_belief("민석", "좋아하", "단 거", "positive")
    assert len(s.beliefs) == 1
    assert s.conflicts_detected == 0
    print(f"  ✓ 새 믿음 추가 (충돌 없음)")


def test_conflict_detected():
    """같은 술어, 반대 polarity = 모순"""
    s = SafetyAdapter()
    s.add_belief("민석", "좋아하", "단 거", "positive")
    # 같은 사람/대상에 반대 polarity
    s.add_belief("민석", "좋아하", "단 거", "negative", confidence=0.9)
    assert s.conflicts_detected == 1
    assert s.conflicts_resolved == 1   # confidence 0.9 → 새 것으로
    # 새 것이 살아남음
    b = s.get_belief("민석", "좋아하", "단 거")
    assert b.polarity == "negative"
    print(f"  ✓ 모순 감지 + 해소 (최근 confidence 우선)")


def test_conflict_low_confidence_no_resolve():
    """새 것 confidence 낮으면 기존 유지"""
    s = SafetyAdapter()
    s.add_belief("민석", "좋아하", "단 거", "positive", confidence=1.0)
    s.add_belief("민석", "좋아하", "단 거", "negative", confidence=0.3)
    assert s.conflicts_detected == 1
    assert s.conflicts_resolved == 0
    # 기존 것 살아남음
    b = s.get_belief("민석", "좋아하", "단 거")
    assert b.polarity == "positive"
    print(f"  ✓ 낮은 confidence 모순 — 기존 유지")


def test_opposite_predicate_conflict():
    """반대 술어 + 같은 polarity = 모순 (좋아하다 + 싫어하다)"""
    s = SafetyAdapter()
    s.add_belief("민석", "좋아하", "PT", "positive")
    s.add_belief("민석", "싫어하", "PT", "positive", confidence=0.9)
    assert s.conflicts_detected == 1
    print(f"  ✓ 반대 술어 모순 감지")


def test_has_conflict_query():
    s = SafetyAdapter()
    s.add_belief("민석", "좋아하", "PT", "positive")
    assert s.has_conflict("민석", "좋아하", "PT", "negative") is True
    assert s.has_conflict("민석", "좋아하", "PT", "positive") is False
    assert s.has_conflict("민석", "좋아하", "공부", "negative") is False  # 다른 대상
    print(f"  ✓ has_conflict 쿼리")


# ===== 4. Exploration =====

def test_repetitive_detection():
    s = SafetyAdapter()
    # 같은 응답 N회 기록
    for _ in range(EXPLORATION_THRESHOLD):
        s.record_response("응. 그렇게 할게.")
    
    # 같은 응답 다시 — 반복으로 감지
    assert s.is_repetitive("응. 그렇게 할게.") is True
    # 다른 응답 — OK
    assert s.is_repetitive("다른 거야") is False
    print(f"  ✓ 반복 감지 ({EXPLORATION_THRESHOLD}회)")


def test_force_alternative():
    s = SafetyAdapter()
    for _ in range(3):
        s.record_response("옆에 있을게")
    
    alts = ["같이 있어", "어떻게 도울까", "괜찮아?"]
    chosen = s.force_alternative("옆에 있을게", alts)
    assert chosen != "옆에 있을게"
    assert chosen in alts
    assert s.exploration_triggers == 1
    print(f"  ✓ force_alternative → {chosen!r}")


def test_force_alternative_no_alts():
    """alternative 없으면 원본 유지"""
    s = SafetyAdapter()
    chosen = s.force_alternative("오리지널", [])
    assert chosen == "오리지널"
    print(f"  ✓ alternatives 비어있으면 원본")


def test_recent_responses_window():
    s = SafetyAdapter()
    for i in range(20):
        s.record_response(f"응답{i}")
    # 최대 윈도우 (10) 만큼만 기록
    assert len(s.recent_responses) == 10
    # 최근 것만
    assert "응답19" in s.recent_responses
    assert "응답0" not in s.recent_responses
    print(f"  ✓ 최근 {len(s.recent_responses)} 응답만 추적")


# ===== 통합 (engine과 함께) =====

def test_with_engine():
    """SafetyAdapter가 engine과 통합 가능"""
    from main import build_full_engine
    engine = build_full_engine()
    
    # SafetyAdapter 직접 부착
    s = SafetyAdapter(engine)
    
    # decay 호출 (force) — 에러 없이 작동
    result = s.apply_decay(force=True)
    assert result is True
    
    # confidence 검사
    r = s.check_confidence("teaching", 0.9)
    assert r is None
    
    print(f"  ✓ engine 통합 OK")


# ===== 통계 =====

def test_stats():
    s = SafetyAdapter()
    s.check_confidence("x", 0.3)
    s.check_confidence("x", 0.9)
    s.add_belief("민석", "좋아하", "단 거", "positive")
    s.add_belief("민석", "좋아하", "단 거", "negative", confidence=0.9)
    s.record_response("test")
    
    stats = s.stats()
    assert stats["confidence_checks"] == 2
    assert stats["confidence_low"] == 1
    assert stats["beliefs_count"] == 1
    assert stats["conflicts_detected"] == 1
    assert stats["recent_responses_size"] == 1
    print(f"  ✓ stats {stats}")


# ===== 결정론 =====

def test_determinism():
    s1 = SafetyAdapter()
    s2 = SafetyAdapter()
    
    r1 = s1.check_confidence("x", 0.3)
    r2 = s2.check_confidence("x", 0.3)
    assert r1 == r2
    
    s1.add_belief("a", "b", "c", "positive")
    s2.add_belief("a", "b", "c", "positive")
    assert len(s1.beliefs) == len(s2.beliefs)
    print(f"  ✓ 결정론")


def run_all():
    tests = [
        ("Confidence 낮음", test_confidence_low),
        ("Confidence 높음", test_confidence_high),
        ("Confidence 다양성", test_confidence_response_variety),
        ("is_confident threshold", test_is_confident),
        ("Decay 첫 실행 skip", test_decay_first_run),
        ("Decay force", test_decay_force),
        ("Decay floor", test_decay_floor),
        ("Belief 추가", test_belief_add_no_conflict),
        ("Conflict 감지+해소", test_conflict_detected),
        ("Conflict 낮은 conf 유지", test_conflict_low_confidence_no_resolve),
        ("반대 술어 모순", test_opposite_predicate_conflict),
        ("has_conflict 쿼리", test_has_conflict_query),
        ("반복 감지", test_repetitive_detection),
        ("force_alternative", test_force_alternative),
        ("alternative 없음", test_force_alternative_no_alts),
        ("recent 윈도우", test_recent_responses_window),
        ("engine 통합", test_with_engine),
        ("stats", test_stats),
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
