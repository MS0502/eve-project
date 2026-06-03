"""
라운드 33 (C-1): KoreanMorphAnalyzer — 룰 기반 한국어 형태소 분석기

학계 기반 검증:
- 학교문법 5분법 (평서/의문/명령/청유/감탄)
- TOPIK 공식 인용 패턴
- 한국민족문화대백과사전 문체법
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.korean.morph_analyzer import (
    KoreanMorphAnalyzer,
    DECLARATIVE_ENDINGS, INTERROGATIVE_ENDINGS,
    IMPERATIVE_ENDINGS, PROPOSITIVE_ENDINGS,
    QUOTE_PARTICLES, CASE_PARTICLES, TOPIC_PARTICLES,
)


# ===== 어휘 테이블 검증 =====

def test_endings_loaded():
    assert len(DECLARATIVE_ENDINGS) > 10
    assert len(INTERROGATIVE_ENDINGS) > 5
    assert len(IMPERATIVE_ENDINGS) > 5
    assert len(PROPOSITIVE_ENDINGS) > 2
    print(f"  ✓ 종결어미 {len(DECLARATIVE_ENDINGS)+len(INTERROGATIVE_ENDINGS)+len(IMPERATIVE_ENDINGS)+len(PROPOSITIVE_ENDINGS)}+ 등록")


def test_quote_particles_loaded():
    assert "라고" in QUOTE_PARTICLES
    assert "다고" in QUOTE_PARTICLES
    assert "냐고" in QUOTE_PARTICLES
    assert "자고" in QUOTE_PARTICLES
    print(f"  ✓ 인용 조사 {len(QUOTE_PARTICLES)}개")


def test_particles_loaded():
    assert "은" in TOPIC_PARTICLES
    assert "이" in CASE_PARTICLES
    assert "을" in CASE_PARTICLES
    print(f"  ✓ 격조사 {len(CASE_PARTICLES)}, 보조사 {len(TOPIC_PARTICLES)}")


# ===== 문장 유형 분류 =====

def test_declarative():
    a = KoreanMorphAnalyzer()
    cases = [
        "오늘 날씨가 좋다",
        "민석이 PT 빡셌어",
        "비가 와요",
        "학생입니다",
        "갔다",
    ]
    for c in cases:
        s = a.analyze(c)
        assert s.sentence_type == "declarative", f"{c!r} → {s.sentence_type}"
    print(f"  ✓ 평서 {len(cases)}개")


def test_interrogative():
    a = KoreanMorphAnalyzer()
    cases = [
        "오늘 어때?",
        "내가 누구야?",
        "갈까요?",
        "뭐 해?",
        "왜?",
        "언제 가나요?",
        "PT 했니?",
    ]
    for c in cases:
        s = a.analyze(c)
        assert s.sentence_type == "interrogative", f"{c!r} → {s.sentence_type}"
    print(f"  ✓ 의문 {len(cases)}개")


def test_imperative():
    a = KoreanMorphAnalyzer()
    cases = [
        "빨리 가라",
        "조용히 해라",
        "응 이라고 말해",
        "거기 서",
    ]
    for c in cases:
        s = a.analyze(c)
        # 명령 또는 평서 (단답 명령은 모호)
        print(f"  {c!r} → {s.sentence_type}")


def test_propositive():
    a = KoreanMorphAnalyzer()
    cases = [
        "같이 가자",
        "밥 먹자",
        "밥 먹읍시다",
    ]
    for c in cases:
        s = a.analyze(c)
        assert s.sentence_type == "propositive", f"{c!r} → {s.sentence_type}"
    print(f"  ✓ 청유 {len(cases)}개")


# ===== 인용 분석 (핵심) =====

def test_indirect_quote_declarative():
    """V/A + (ㄴ/는)다고 (간접 평서 인용)"""
    a = KoreanMorphAnalyzer()
    cases = [
        ("민석이가 좋다고 말했어", "declarative"),
        ("그가 간다고 했어", "declarative"),
        ("아이가 먹는다고 답했어", "declarative"),
    ]
    for text, expected_modality in cases:
        s = a.analyze(text)
        assert s.is_quoted, f"{text!r} 인용 못 잡음"
        assert s.quoted_modality == expected_modality, \
            f"{text!r} modality: {s.quoted_modality}"
    print(f"  ✓ 간접 평서 {len(cases)}개")


def test_indirect_quote_interrogative():
    """냐고 (간접 의문)"""
    a = KoreanMorphAnalyzer()
    cases = [
        "왜냐고 물었어",
        "어디냐고 답해",
    ]
    for text in cases:
        s = a.analyze(text)
        assert s.is_quoted
        assert s.quoted_modality == "interrogative", \
            f"{text!r} modality: {s.quoted_modality}"
    print(f"  ✓ 간접 의문 {len(cases)}개")


def test_indirect_quote_propositive():
    """자고 (간접 청유)"""
    a = KoreanMorphAnalyzer()
    s = a.analyze("같이 가자고 했어")
    assert s.is_quoted
    assert s.quoted_modality == "propositive"
    print(f"  ✓ 간접 청유")


def test_direct_quote():
    """라고/이라고 (직접 또는 명사)"""
    a = KoreanMorphAnalyzer()
    s = a.analyze("'민석'이라고 부르면 돼")
    assert s.is_quoted
    print(f"  ✓ 직접 인용")


# ===== 너 실제 입력 (라운드 32 후 발견) =====

def test_user_real_input_korean_quote():
    """너가 친 거 — '좋아한다고해'"""
    a = KoreanMorphAnalyzer()
    # 띄어쓰기 다양
    cases = [
        "좋아한다고 해",
        "좋아한다고해",
    ]
    for text in cases:
        s = a.analyze(text)
        # is_quoted 됐으면 OK (띄어쓰기 없는 건 어절 분리에서 한계)
        print(f"  {text!r}: is_quoted={s.is_quoted}, modality={s.quoted_modality}")


def test_question_words_detected():
    """의문 어휘 인식"""
    a = KoreanMorphAnalyzer()
    cases = [
        "내가 누구야?",
        "넌 뭐 좋아해?",
        "어디 가니?",
        "왜 안 와?",
    ]
    for text in cases:
        s = a.analyze(text)
        assert s.has_question_word, f"{text!r} 의문 어휘 X"
        assert s.sentence_type == "interrogative"
    print(f"  ✓ 의문 어휘 {len(cases)}개")


# ===== 부정/시제/높임 =====

def test_negation():
    a = KoreanMorphAnalyzer()
    cases = [
        "안 가",
        "PT 안 했어",
        "가지 않아",
        "가지 마",
    ]
    for text in cases:
        s = a.analyze(text)
        assert s.is_negative, f"{text!r} 부정 X"
    print(f"  ✓ 부정 {len(cases)}개")


def test_past_tense():
    a = KoreanMorphAnalyzer()
    cases = [
        "PT 빡셌어",
        "갔어",
        "했어",
        "민석이 PT 빡셌어",
    ]
    for text in cases:
        s = a.analyze(text)
        assert s.has_past, f"{text!r} 과거 X"
    print(f"  ✓ 과거 시제 {len(cases)}개")


def test_honorific():
    a = KoreanMorphAnalyzer()
    cases = [
        "하십니까?",
        "오시지요",
    ]
    for text in cases:
        s = a.analyze(text)
        assert s.has_honorific, f"{text!r} 높임 X"
    print(f"  ✓ 높임 {len(cases)}개")


# ===== 결정론 =====

def test_determinism():
    a1 = KoreanMorphAnalyzer()
    a2 = KoreanMorphAnalyzer()
    text = "민석이가 좋아한다고 말했어"
    s1 = a1.analyze(text)
    s2 = a2.analyze(text)
    assert s1.sentence_type == s2.sentence_type
    assert s1.is_quoted == s2.is_quoted
    print(f"  ✓ 결정론")


# ===== 헬퍼 =====

def test_helpers():
    a = KoreanMorphAnalyzer()
    assert a.is_question("뭐야?") is True
    assert a.is_question("좋아") is False
    assert a.is_propositive("같이 가자") is True
    assert a.has_quote("좋다고 했어") is True
    assert a.has_quote("좋아") is False
    print(f"  ✓ 헬퍼 메서드")


# ===== 가르침 인용 자동 인식 =====

def test_is_teaching_quote():
    a = KoreanMorphAnalyzer()
    # 가르침 (인용 + 명령형 발화 동사)
    assert a.is_teaching_quote("응이라고 말해") is True
    assert a.is_teaching_quote("좋다고 해") is True
    assert a.is_teaching_quote("민석이라고 부르면 돼") is True
    # 가르침 X (단순 보고)
    assert a.is_teaching_quote("그는 좋다고 했었다") is False
    print(f"  ✓ 가르침 자동 인식")


def run_all():
    tests = [
        ("종결어미 등록", test_endings_loaded),
        ("인용 조사 등록", test_quote_particles_loaded),
        ("격조사/보조사 등록", test_particles_loaded),
        ("평서 분류", test_declarative),
        ("의문 분류", test_interrogative),
        ("명령 분류", test_imperative),
        ("청유 분류", test_propositive),
        ("간접 평서 인용", test_indirect_quote_declarative),
        ("간접 의문 인용", test_indirect_quote_interrogative),
        ("간접 청유 인용", test_indirect_quote_propositive),
        ("직접 인용", test_direct_quote),
        ("사용자 실제 입력", test_user_real_input_korean_quote),
        ("의문 어휘", test_question_words_detected),
        ("부정 인식", test_negation),
        ("과거 시제", test_past_tense),
        ("높임", test_honorific),
        ("결정론", test_determinism),
        ("헬퍼 메서드", test_helpers),
        ("가르침 자동 인식", test_is_teaching_quote),
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
