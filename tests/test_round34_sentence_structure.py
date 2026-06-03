"""
라운드 34 (C-2): SentenceStructureAnalyzer
의도 분류 — 너 진짜 입력 + 학계 케이스
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.korean.sentence_structure import (
    SentenceStructureAnalyzer, IntentStructure,
)


def _check(text, expected_intent, expected_meta=None, min_conf=0.5):
    a = SentenceStructureAnalyzer()
    s = a.analyze(text)
    assert s.intent == expected_intent, \
        f"{text!r} → intent={s.intent} (기대: {expected_intent}), why={s.why}"
    if expected_meta is not None:
        assert s.meta_target == expected_meta, \
            f"{text!r} → meta_target={s.meta_target} (기대: {expected_meta})"
    assert s.intent_confidence >= min_conf, \
        f"{text!r} → confidence={s.intent_confidence} (기대 ≥ {min_conf})"


# ===== 의도 분류 =====

def test_meta_self():
    """EVE 자체 질문"""
    cases = [
        "너는 누구야?",
        "넌 뭐야?",
        "너 살아있어?",
        "너는 뭘 좋아해?",
        "넌 기분이 어때?",
    ]
    for text in cases:
        _check(text, "meta_self")
    print(f"  ✓ meta_self {len(cases)}개")


def test_meta_user():
    """사용자 질문"""
    cases = [
        "내가 누구야?",
        "내가 누군지 알아?",
        "내가 뭘 좋아해?",
    ]
    for text in cases:
        _check(text, "meta_user")
    print(f"  ✓ meta_user {len(cases)}개")


def test_question_general():
    """일반 질문"""
    cases = [
        "오늘 날씨 어때?",
        "PT 언제 끝나?",
        "어디 가니?",
    ]
    for text in cases:
        a = SentenceStructureAnalyzer()
        s = a.analyze(text)
        assert s.intent == "question", f"{text!r} → {s.intent}"
    print(f"  ✓ 일반 질문 {len(cases)}개")


def test_emotional_negative():
    """부정 감정"""
    cases = [
        "민석이 너무 힘들어",
        "PT 빡셌어",
        "오늘 진짜 우울해",
    ]
    for text in cases:
        a = SentenceStructureAnalyzer()
        s = a.analyze(text)
        assert s.intent == "emotional", f"{text!r} → {s.intent}, why: {s.why}"
        assert s.emotion_polarity == "negative"
    print(f"  ✓ 부정 감정 {len(cases)}개")


def test_emotional_positive():
    cases = [
        "오늘 진짜 기뻐",
        "행복해",
        "신나!",
    ]
    for text in cases:
        a = SentenceStructureAnalyzer()
        s = a.analyze(text)
        assert s.intent == "emotional", f"{text!r} → {s.intent}"
        assert s.emotion_polarity == "positive"
    print(f"  ✓ 긍정 감정 {len(cases)}개")


def test_teaching():
    """가르침 (인용 + 명령형 발화)"""
    cases = [
        "응 이라고 말해",
        "이럴때 좋다고 해",
        "민석이라고 부르면 돼",
        "가자고 해",
    ]
    for text in cases:
        a = SentenceStructureAnalyzer()
        s = a.analyze(text)
        assert s.intent == "teaching", f"{text!r} → {s.intent}, why: {s.why}"
    print(f"  ✓ 가르침 {len(cases)}개")


def test_reported_speech():
    """보고 (인용 + 비명령 발화)"""
    cases = [
        "그가 좋다고 했었다",
        "민석이 PT 빡셌다고 했었어",
    ]
    for text in cases:
        a = SentenceStructureAnalyzer()
        s = a.analyze(text)
        assert s.intent in ("reported_speech", "emotional"), \
            f"{text!r} → {s.intent} (둘 중 하나여야)"
    print(f"  ✓ 보고 {len(cases)}개")


def test_definition():
    """정의문 (X는 Y야)"""
    cases = [
        ("좋아하는건 너가 하고싶은거야", "좋아하는건"),
        ("진짜 친구는 가까운 사람이야", "진짜"),
        ("PT는 운동이야", "PT"),
    ]
    for text, expected_concept in cases:
        a = SentenceStructureAnalyzer()
        s = a.analyze(text)
        assert s.intent == "definition", f"{text!r} → {s.intent}, why: {s.why}"
        assert s.defined_concept is not None
        print(f"  {text!r} → defined={s.defined_concept!r}, def={s.definition!r}")
    print(f"  ✓ 정의문 {len(cases)}개")


def test_request():
    """명령/청유/요청"""
    cases = [
        "도와줘",
        "같이 가자",
    ]
    for text in cases:
        a = SentenceStructureAnalyzer()
        s = a.analyze(text)
        assert s.intent == "request", f"{text!r} → {s.intent}"
    print(f"  ✓ 요청 {len(cases)}개")


def test_statement_default():
    """디폴트 평서"""
    cases = [
        "오늘 날씨가 그렇네",
        "지금 11시야",
    ]
    for text in cases:
        a = SentenceStructureAnalyzer()
        s = a.analyze(text)
        # statement 또는 emotional/definition 등 다른 분류
        print(f"  {text!r} → {s.intent}")


# ===== 너 진짜 입력 =====

def test_real_user_input():
    """너가 친 실제 입력 9개 — 의도 분류 확인"""
    a = SentenceStructureAnalyzer()
    cases = [
        ("안녕?", ["question", "ambiguous", "statement"]),
        ("내가 누구야?", ["meta_user"]),
        ("너는 살아있는거같아?", ["meta_self"]),
        ("하고싶은거 있어?", ["meta_self", "question"]),
        ("너는 뭘 좋아해?", ["meta_self"]),
        ("좋아하는건 너가 하고싶은거야 너가 그걸 할때 기쁜거 그걸 좋아한다고해", ["definition", "teaching"]),
        ("왜 몰라?", ["meta_self", "question"]),
        ("민석이 PT 빡셌어", ["emotional"]),
        ("이럴때 응 이라고 말해", ["teaching"]),
    ]
    print()
    for text, allowed in cases:
        s = a.analyze(text)
        match = "✓" if s.intent in allowed else "✗"
        print(f"  {match} {text!r:60s} → {s.intent} (conf={s.intent_confidence:.2f})")
        assert s.intent in allowed, \
            f"{text!r} → {s.intent}, 허용: {allowed}, why: {s.why}"
    print(f"\n  ✓ 모든 진짜 입력 분류")


# ===== 결정론 =====

def test_determinism():
    a1 = SentenceStructureAnalyzer()
    a2 = SentenceStructureAnalyzer()
    text = "민석이가 PT 빡셌다고 했어"
    s1 = a1.analyze(text)
    s2 = a2.analyze(text)
    assert s1.intent == s2.intent
    assert s1.intent_confidence == s2.intent_confidence
    print(f"  ✓ 결정론")


def test_confidence_range():
    a = SentenceStructureAnalyzer()
    cases = ["안녕?", "민석이 누구야?", "PT 빡셌어"]
    for text in cases:
        s = a.analyze(text)
        assert 0.0 <= s.intent_confidence <= 1.0
    print(f"  ✓ confidence 0~1 범위")


def run_all():
    tests = [
        ("meta_self", test_meta_self),
        ("meta_user", test_meta_user),
        ("일반 질문", test_question_general),
        ("부정 감정", test_emotional_negative),
        ("긍정 감정", test_emotional_positive),
        ("가르침", test_teaching),
        ("보고 (reported)", test_reported_speech),
        ("정의문", test_definition),
        ("요청", test_request),
        ("statement 디폴트", test_statement_default),
        ("실제 사용자 입력", test_real_user_input),
        ("결정론", test_determinism),
        ("confidence 범위", test_confidence_range),
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
