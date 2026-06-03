"""
라운드 2: NLAdapter + SDAdapter

확인:
1. NL 어댑터 없으면 기존 v41 동작
2. NL 켜면 한국어 entities 정확해짐 ("PT가" → "PT")
3. NL.understand의 intent가 v41 Meaning에 반영
4. SD 어댑터 없으면 기존 동작
5. SD 켜도 평이한 후보는 그대로 (accept)
6. 5 어댑터 다 켜도 결정론
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language.streaming import StreamingEngine
from adapters.hormone_adapter import HormoneAdapter
from adapters.activation_adapter import ActivationAdapter
from adapters.memory_adapter import MemoryAdapter
from adapters.nl_adapter import NLAdapter
from adapters.sd_adapter import SDAdapter


def make_engine(use_nl=False, use_sd=False, use_full=False):
    kwargs = {"delays": {"stage1": 0, "core": 0, "expand": 0, "final": 0}}
    if use_full:
        h = HormoneAdapter()
        a = ActivationAdapter(hormone_adapter=h)
        m = MemoryAdapter(activation_adapter=a, hormone_adapter=h)
        n = NLAdapter(hormone_adapter=h, activation_adapter=a) if use_nl else None
        s = SDAdapter(hormone_adapter=h, activation_adapter=a) if use_sd else None
        kwargs.update(dict(
            hormone_adapter=h, activation_adapter=a, memory_adapter=m,
            nl_adapter=n, sd_adapter=s,
        ))
    else:
        if use_nl:
            kwargs["nl_adapter"] = NLAdapter()
        if use_sd:
            kwargs["sd_adapter"] = SDAdapter()
    return StreamingEngine(**kwargs)


def test_nl_strips_particles():
    """NL 켜면 'PT가 힘들다' → entity에 'PT' (조사 'PT가' 아님)."""
    engine = make_engine(use_nl=True)
    meaning = engine.lu.parse("PT가 힘들다")
    print(f"  NL on: entities={meaning.entities}, intent={meaning.intent}")
    # 정확히는 NL.categories 사용. 'PT가' 대신 'PT'가 들어가야 함.
    has_clean = "PT" in meaning.entities and "PT가" not in meaning.entities
    assert has_clean, f"조사 처리 실패: {meaning.entities}"


def test_nl_off_uses_keyword_split():
    """NL 끄면 기존 키워드 split — 조사 안 떼짐."""
    engine = make_engine(use_nl=False)
    meaning = engine.lu.parse("PT가 힘들다")
    print(f"  NL off: entities={meaning.entities}")
    # 'PT가' 그대로 있을 것 (또는 split 결과 그대로)
    assert any("PT" in e for e in meaning.entities)


def test_nl_intent_overrides_v41_default():
    """NL이 'emotion' intent 정확히 잡음."""
    engine = make_engine(use_nl=True)
    meaning = engine.lu.parse("나 진짜 힘들어")
    print(f"  NL intent: {meaning.intent}, sentiment={meaning.context.get('nl_sentiment')}")
    # NL이 negative sentiment + emotion 잡아야 함
    assert meaning.intent in ("emotion", "statement")
    assert meaning.context.get("nl_sentiment") == "negative"


def test_sd_off_keeps_all_candidates():
    """SD 끄면 모든 후보 통과."""
    engine = make_engine(use_sd=False)
    out = list(engine.chat_stream("민석아 힘들다"))
    print(f"  SD off: {out}")
    assert len(out) >= 4


def test_sd_on_normal_input_still_works():
    """SD 켜도 평이한 입력은 응답 유지 (전부 reject 되면 안 됨)."""
    engine = make_engine(use_sd=True)
    out = list(engine.chat_stream("민석아 힘들다"))
    print(f"  SD on: {out}")
    assert len(out) >= 3, f"SD가 너무 reject함: {out}"


def test_sd_doubt_metadata_in_candidates():
    """SD 켜면 후보에 sd_action 메타 붙음."""
    engine = make_engine(use_sd=True)
    meaning = engine.lu.parse("민석아 힘들다")
    from cognition.meaning_graph import build_graph
    engine.workspace.update(build_graph(meaning), meaning)
    state = engine.workspace.get_state()
    cands = engine.s1.generate_candidates(state)
    cands = engine.s2.filter(cands)
    print(f"  candidates: {[(c.get('text'), c.get('sd_action'), c.get('sd_doubt')) for c in cands]}")
    assert any("sd_action" in c for c in cands), f"sd_action 없음: {cands}"


def test_full_stack_5_adapters_works():
    """호르몬+활성+기억+NL+SD 다 켜도 정상 응답."""
    engine = make_engine(use_nl=True, use_sd=True, use_full=True)
    out = list(engine.chat_stream("민석아 PT가 힘들다"))
    full = " ".join(out)
    print(f"  full stack: {out}")
    assert len(out) >= 4, f"풀스택 응답 짧음: {out}"
    # fatigue 응답 들어가야 함
    assert any("지친" in c or "에너지" in c for c in out), f"fatigue 응답 없음: {full}"


def test_full_stack_two_turn_recall_with_clean_categories():
    """NL이 'PT가'를 'PT'로 정리 → 2턴 회상이 'PT' 일관 카테고리로."""
    engine = make_engine(use_nl=True, use_sd=True, use_full=True)
    list(engine.chat_stream("PT가 힘들다"))
    out2 = list(engine.chat_stream("PT 또 힘들어"))
    print(f"  2nd turn: {out2}")
    # 회상 phrase 들어가야 함 (entity 매칭이 NL로 깨끗해져서)
    has_recall = any("전에" in c and "떠오르네" in c for c in out2)
    assert has_recall, f"회상 안 들어옴: {out2}"


def test_full_stack_determinism():
    """5 어댑터 결정론 유지."""
    e1 = make_engine(use_nl=True, use_sd=True, use_full=True)
    e2 = make_engine(use_nl=True, use_sd=True, use_full=True)
    out1 = list(e1.chat_stream("민석아 힘들다"))
    out2 = list(e2.chat_stream("민석아 힘들다"))
    assert out1 == out2, f"비결정론!\n{out1}\nvs\n{out2}"
    print(f"  ✓ 5스택 결정론")


def run_all():
    tests = [
        ("NL 조사 떼기", test_nl_strips_particles),
        ("NL off는 split", test_nl_off_uses_keyword_split),
        ("NL intent 보강", test_nl_intent_overrides_v41_default),
        ("SD off 정상", test_sd_off_keeps_all_candidates),
        ("SD on 정상", test_sd_on_normal_input_still_works),
        ("SD 메타 부착", test_sd_doubt_metadata_in_candidates),
        ("풀스택 작동", test_full_stack_5_adapters_works),
        ("풀스택 2턴 회상", test_full_stack_two_turn_recall_with_clean_categories),
        ("풀스택 결정론", test_full_stack_determinism),
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
