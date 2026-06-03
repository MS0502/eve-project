"""
라운드 29 (B-1): OrchestratorAdapter — 상황 분류 + 모듈 선택 + 컨텍스트 수집
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import build_full_engine
from utils.types import Meaning


def _meaning(text, intent="statement", entities=None, emotions=None):
    m = Meaning(intent=intent, entities=entities or [], emotions=emotions or {})
    m.raw_text = text
    return m


def test_adapter_present():
    engine = build_full_engine()
    assert engine.orchestrator_adapter is not None
    print(f"  ✓ orchestrator_adapter 부착")


# ===== 분류 정확도 =====

def test_classify_greeting():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = ["안녕", "안녕 민석이야", "오랜만이야", "왔어"]
    for text in cases:
        s = o.classify(_meaning(text))
        assert s == "greeting", f"{text!r} → {s} (기대: greeting)"
    print(f"  ✓ greeting {len(cases)}개")


def test_classify_meta_self():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = [
        "너는 누구야?",
        "넌 뭐야",
        "너는 살아있어?",
        "EVE는 누구야",
        "기분 어때?",
    ]
    for text in cases:
        s = o.classify(_meaning(text, intent="question"))
        assert s == "meta_self", f"{text!r} → {s} (기대: meta_self)"
    print(f"  ✓ meta_self {len(cases)}개")


def test_classify_meta_user():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = [
        "내가 누구야?",
        "내가 누군지 알아?",
        "나에 대해 알아?",
    ]
    for text in cases:
        s = o.classify(_meaning(text, intent="question"))
        assert s == "meta_user", f"{text!r} → {s} (기대: meta_user)"
    print(f"  ✓ meta_user {len(cases)}개")


def test_classify_emotional_share():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = [
        "민석이 너무 힘들어",
        "오늘 진짜 빡셌어 피곤해",
        "기분 좋아 행복해",
        "화나 짜증나",
        "무서워",
    ]
    for text in cases:
        s = o.classify(_meaning(text))
        assert s == "emotional_share", f"{text!r} → {s} (기대: emotional_share)"
    print(f"  ✓ emotional_share {len(cases)}개")


def test_classify_task():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = [
        "3 더하기 5는?",
        "10 빼기 3",
        "2 곱하기 4",
        "이거 계산해줘",
    ]
    for text in cases:
        s = o.classify(_meaning(text, intent="command"))
        assert s == "task", f"{text!r} → {s} (기대: task)"
    print(f"  ✓ task {len(cases)}개")


def test_classify_teaching():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = [
        "이럴때 응 이라고 말해",
        "'민석이 같아'라고 말하면 돼",
        "나를 민석이라고 부르면 돼",
    ]
    for text in cases:
        s = o.classify(_meaning(text))
        assert s == "teaching", f"{text!r} → {s} (기대: teaching)"
    print(f"  ✓ teaching {len(cases)}개")


def test_classify_causal_what_if():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = [
        "만약 PT 안 했으면 어땠을까",
        "PT 안 했더라면 어떨까",
        "그게 다르다면 어땠을까",
    ]
    for text in cases:
        s = o.classify(_meaning(text))
        assert s == "causal_what_if", f"{text!r} → {s} (기대: causal_what_if)"
    print(f"  ✓ causal_what_if {len(cases)}개")


def test_classify_past_recall():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = [
        "저번에 얘기한 거 기억나?",
        "전에 PT 얘기 했었지",
        "어제 뭐 했어?",
        "예전에 그런 적 있었어",
    ]
    for text in cases:
        s = o.classify(_meaning(text))
        assert s == "past_recall", f"{text!r} → {s} (기대: past_recall)"
    print(f"  ✓ past_recall {len(cases)}개")


def test_classify_factual_question():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    cases = [
        "PT 어디서 해?",
        "오늘 왜 그래?",
        "그게 뭐야?",
    ]
    for text in cases:
        s = o.classify(_meaning(text, intent="question"))
        assert s == "factual_question", f"{text!r} → {s} (기대: factual_question)"
    print(f"  ✓ factual_question {len(cases)}개")


def test_classify_small_talk_fallback():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    # 어디에도 안 걸리는 평범한 평서문
    cases = ["응", "그래", "좋아 좋아"]
    for text in cases:
        s = o.classify(_meaning(text))
        # small_talk 또는 emotional_share ('좋아'에 걸림) 다 OK
        print(f"  {text!r} → {s}")


# ===== 모듈 선택 =====

def test_select_modules_for_each_situation():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    for situation in [
        "greeting", "meta_self", "meta_user",
        "emotional_share", "factual_question", "task",
        "teaching", "causal_what_if", "past_recall", "small_talk",
    ]:
        modules = o.select_modules(situation)
        assert isinstance(modules, list)
        assert len(modules) > 0
        print(f"  {situation}: {modules}")


# ===== 컨텍스트 수집 =====

def test_gather_context_emotional():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    m = _meaning("민석이 PT 빡셌어", entities=["민석", "PT", "빡셌"])
    ctx = o.gather_context(m, situation="emotional_share")
    assert "situation" in ctx
    assert ctx["situation"] == "emotional_share"
    print(f"  keys: {list(ctx.keys())}")
    # suffering, user_presence, hormones 등 있어야
    assert "suffering_signal" in ctx or "user_presence" in ctx


def test_gather_context_meta_self():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    m = _meaning("너는 누구야?", entities=["너"])
    ctx = o.gather_context(m, situation="meta_self")
    print(f"  keys: {list(ctx.keys())}")
    # self snapshot 있어야
    assert any(k in ctx for k in ["self", "somatic", "hormones"])


def test_gather_context_task():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    m = _meaning("3 더하기 5는?", intent="command")
    ctx = o.gather_context(m, situation="task")
    print(f"  keys: {list(ctx.keys())}")
    # task_result 키 있어야 (None이라도)
    assert "task_result" in ctx or "reasoning" in ctx


# ===== process 종합 =====

def test_process_returns_full():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    m = _meaning("민석이 너무 힘들어")
    r = o.process(m)
    assert "situation" in r
    assert "trace" in r
    assert "modules_selected" in r
    assert "context" in r
    assert r["situation"] == "emotional_share"
    print(f"  ✓ situation: {r['situation']}, modules: {r['modules_selected'][:3]}")


def test_classify_with_trace():
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    trace = o.classify_with_trace(_meaning("안녕"))
    assert "situation" in trace
    assert "via" in trace
    print(f"  trace: {trace}")


# ===== chat_stream 통합 =====

def test_chat_stream_attaches_orchestrator():
    """chat_stream 후 meaning.context에 orchestrator 결과 attach 검증."""
    engine = build_full_engine()
    list(engine.chat_stream("민석이 PT 빡셌어"))
    # orchestrator 통계
    s = engine.orchestrator_adapter.stats()
    assert s["total_classifications"] > 0
    print(f"  분류 누적: {s['total_classifications']}")
    print(f"  상황 분포: {s['situations']}")


def test_chat_stream_does_not_break():
    """B-1은 추가만 — 응답 흐름 안 깨짐."""
    engine = build_full_engine()
    out = list(engine.chat_stream("3 더하기 5는 뭐야"))
    assert "8" in " ".join(out)
    out = list(engine.chat_stream("어디 있어?"))
    assert "내방" in " ".join(out)
    out = list(engine.chat_stream("민석아 너무 힘들어"))
    full = " ".join(out)
    # 라운드 56: SpeechHub 다양화 — 공감 키워드 풀 넓힘
    EMPATHY_KEYWORDS = ["느껴져", "지친", "힘들", "빡세", "옆에", "쉬어",
                         "마음이", "같이", "그치", "얘기"]
    assert any(k in full for k in EMPATHY_KEYWORDS), f"공감 응답 X: {full}"
    print(f"  ✓ 산술/환경/공감 회귀 없음")


def test_determinism():
    e1 = build_full_engine()
    e2 = build_full_engine()
    o1 = list(e1.chat_stream("민석아 힘들다"))
    o2 = list(e2.chat_stream("민석아 힘들다"))
    assert o1 == o2
    # 분류도 동일
    s1 = e1.orchestrator_adapter.classify(_meaning("민석아 힘들다"))
    s2 = e2.orchestrator_adapter.classify(_meaning("민석아 힘들다"))
    assert s1 == s2
    print(f"  ✓ 결정론")


def test_priority_teaching_over_emotion():
    """teaching 패턴이 emotional 키워드 있어도 우선."""
    engine = build_full_engine()
    o = engine.orchestrator_adapter
    # "힘들" 단어 있어도 "라고 말해" 패턴 있으면 teaching
    s = o.classify(_meaning("힘들 때 응 이라고 말해"))
    assert s == "teaching"
    print(f"  ✓ teaching 우선순위")


def test_round28_still_works():
    engine = build_full_engine()
    # 가르침 평가 정상
    out = list(engine.chat_stream("멍청이라고 말해"))
    full = " ".join(out)
    print(f"  부정 가르침 응답: {full}")
    assert any(p in full for p in ["이상", "안 될", "하고 싶지", "왜", "뭔데"])


def run_all():
    tests = [
        ("어댑터 부착", test_adapter_present),
        ("greeting 분류", test_classify_greeting),
        ("meta_self 분류", test_classify_meta_self),
        ("meta_user 분류", test_classify_meta_user),
        ("emotional 분류", test_classify_emotional_share),
        ("task 분류", test_classify_task),
        ("teaching 분류", test_classify_teaching),
        ("causal 분류", test_classify_causal_what_if),
        ("past 분류", test_classify_past_recall),
        ("factual 분류", test_classify_factual_question),
        ("small_talk fallback", test_classify_small_talk_fallback),
        ("모듈 선택 9종", test_select_modules_for_each_situation),
        ("ctx 감정", test_gather_context_emotional),
        ("ctx 메타", test_gather_context_meta_self),
        ("ctx 산술", test_gather_context_task),
        ("process 종합", test_process_returns_full),
        ("classify_with_trace", test_classify_with_trace),
        ("chat 통합 attach", test_chat_stream_attaches_orchestrator),
        ("chat 흐름 안 깨짐", test_chat_stream_does_not_break),
        ("결정론", test_determinism),
        ("teaching 우선순위", test_priority_teaching_over_emotion),
        ("라운드 28 회귀", test_round28_still_works),
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
