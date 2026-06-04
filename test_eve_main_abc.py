"""
v32 B3 SelfDoubt 통합 검증 (eve_main_abc + NL)
================================================

eve_main_abc.py의 B3 통합이 정상 작동하는지 검증.
"""

import sys
import os

# test 파일 위치 기준 경로 (Colab/로컬/모듈 위치 어디든 작동)
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, 'eve_modules'))

from eve_main_abc import EVE_TierAB


def passed(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f"  | {detail}" if detail else ""))
    return bool(ok)


def run_legacy_validation():
    print("=" * 70)
    print("v32 B3 통합 검증 (eve_main_abc)")
    print("=" * 70)
    results = []


    # ============= [1] 인스턴스화 + SD 모듈 존재 =============
    print("\n[1] EVE_TierAB 인스턴스 + SelfDoubt 통합 확인")
    eve = EVE_TierAB()
    results.append(passed("self.sd 인스턴스 존재",
                         hasattr(eve, 'sd') and eve.sd is not None))
    results.append(passed("SD에 NL 연결됨",
                         eve.sd.nl is eve.nl))
    results.append(passed("SD에 SA 연결됨",
                         eve.sd.sa is eve.sa))
    results.append(passed("SD에 DS 연결됨",
                         eve.sd.ds is eve.ds))
    results.append(passed("SD baseline 초기값 0.20",
                         abs(eve.sd.baseline - 0.20) < 1e-6))


    # ============= [2] tick → SD baseline 회귀 호출 =============
    print("\n[2] tick → SD.update 호출 확인")
    eve = EVE_TierAB()
    eve.sd.baseline = 0.45
    before = eve.sd.baseline
    for _ in range(20):
        eve.tick(dt=1.0)
    after = eve.sd.baseline
    print(f"  baseline: {before:.3f} → {after:.3f}")
    results.append(passed("tick 후 baseline 회귀 시작",
                         after < before,
                         f"{before:.3f} → {after:.3f}"))


    # ============= [3] say() → result에 doubt 포함 =============
    print("\n[3] say() → result['doubt'] 포함")
    eve = EVE_TierAB()
    eve.learn_chain(['민석', '대화', '함께'], 0.5)
    result = eve.say("민석아 안녕")
    results.append(passed("result에 doubt 키",
                         'doubt' in result))
    if 'doubt' in result:
        d = result['doubt']
        print(f"  doubt={d['doubt_level']:.3f}, action={d['recommended_action']}, "
              f"sources={d['doubt_sources']}, mode={d['mode']}")
        results.append(passed("doubt에 모든 필수 키",
                             all(k in d for k in
                                 ['doubt_level', 'recommended_action',
                                  'doubt_sources', 'components', 'mode'])))
        results.append(passed("doubt_level [0,1]",
                             0.0 <= d['doubt_level'] <= 1.0))


    # ============= [4] 평온 → accept =============
    print("\n[4] 평온 + 인사 → accept")
    eve = EVE_TierAB()
    eve.learn_chain(['민석', '대화'], 0.5)
    result = eve.say("민석아 안녕")
    print(f"  action={result['doubt']['recommended_action']}, "
          f"response='{result['response']}'")
    results.append(passed("평온 인사 → accept",
                         result['doubt']['recommended_action'] == 'accept',
                         f"doubt={result['doubt']['doubt_level']:.3f}"))


    # ============= [5] Cortisol↑ → 거절 경향 =============
    print("\n[5] Cortisol↑ → 거절 경향")
    eve = EVE_TierAB()
    eve.hs.hormones['cortisol'].level = 0.9
    eve.hs.hormones['oxytocin'].level = 0.1
    eve.hs.hormones['norepinephrine'].level = 0.7
    result = eve.say("이거 해줘")
    print(f"  action={result['doubt']['recommended_action']}, "
          f"doubt={result['doubt']['doubt_level']:.3f}, "
          f"sources={result['doubt']['doubt_sources']}")
    results.append(passed("Cort+NE↑ 명령 → reject 또는 defer",
                         result['doubt']['recommended_action'] in ('reject', 'defer'),
                         f"action={result['doubt']['recommended_action']}"))


    # ============= [6] OT↑ → accept =============
    print("\n[6] Oxytocin↑ → accept")
    eve = EVE_TierAB()
    eve.learn_chain(['민석', '함께'], 0.5)
    eve.hs.hormones['oxytocin'].level = 0.9
    eve.hs.hormones['serotonin'].level = 0.7
    eve.hs.hormones['cortisol'].level = 0.1
    result = eve.say("같이 산책 가자")
    print(f"  action={result['doubt']['recommended_action']}, "
          f"doubt={result['doubt']['doubt_level']:.3f}")
    results.append(passed("OT+5HT↑ 명령 → accept",
                         result['doubt']['recommended_action'] == 'accept'))


    # ============= [7] 호르몬에 따라 응답 다름 =============
    print("\n[7] 호르몬 따라 응답 자연스럽게 다름")
    eve_a = EVE_TierAB()
    eve_a.hs.hormones['cortisol'].level = 0.9
    eve_a.hs.hormones['oxytocin'].level = 0.1
    r_cort = eve_a.say("뭐 좀 해줘")

    eve_b = EVE_TierAB()
    eve_b.hs.hormones['oxytocin'].level = 0.85
    eve_b.hs.hormones['cortisol'].level = 0.1
    r_ot = eve_b.say("뭐 좀 해줘")

    print(f"  Cort↑: '{r_cort['response']}' (action={r_cort['doubt']['recommended_action']})")
    print(f"  OT↑  : '{r_ot['response']}' (action={r_ot['doubt']['recommended_action']})")
    results.append(passed("호르몬 따라 action 다름",
                         r_cort['doubt']['recommended_action']
                         != r_ot['doubt']['recommended_action']))


    # ============= [8] 거절 시 SA에 거절 카테고리 활성 =============
    print("\n[8] reject action → SA '거절' 카테고리 활성")
    eve = EVE_TierAB()
    eve.hs.hormones['cortisol'].level = 0.95
    eve.hs.hormones['norepinephrine'].level = 0.85
    eve.hs.hormones['oxytocin'].level = 0.05
    result = eve.say("그거 해")
    action = result['doubt']['recommended_action']
    print(f"  action={action}")
    if action in ('reject', 'defer'):
        refusal_active = eve.sa.get_activation_level('거절')
        doubt_active = eve.sa.get_activation_level('의심')
        print(f"  '거절' 활성 = {refusal_active:.3f}, '의심' 활성 = {doubt_active:.3f}")
        results.append(passed("reject/defer 시 의심/거절 카테고리 활성",
                             refusal_active > 0.1 or doubt_active > 0.1))
    else:
        print("  (reject 아니어서 활성 검사 skip)")
        results.append(True)


    # ============= [9] introspect → doubt 통계 포함 =============
    print("\n[9] introspect → doubt 통계")
    eve = EVE_TierAB()
    eve.learn_chain(['민석', '대화'], 0.5)
    eve.say("안녕")
    eve.say("뭐해")
    eve.hs.hormones['cortisol'].level = 0.95
    eve.say("위험한 거 해")
    intro = eve.introspect()
    required = ['doubt_baseline', 'doubt_baseline_with_hormone',
                'doubt_eval_count', 'doubt_reject_count', 'last_doubt_action']
    for k in required:
        results.append(passed(f"introspect.{k}",
                             k in intro,
                             f"value={intro.get(k)}"))
    results.append(passed("eval_count 누적",
                         intro['doubt_eval_count'] == 3,
                         f"count={intro['doubt_eval_count']}"))


    # ============= [10] 결정론 =============
    print("\n[10] 결정론")
    e1 = EVE_TierAB()
    e2 = EVE_TierAB()
    e1.learn_chain(['민석', '대화'], 0.5)
    e2.learn_chain(['민석', '대화'], 0.5)
    r1 = e1.say("민석아 안녕")
    r2 = e2.say("민석아 안녕")
    results.append(passed("같은 입력 → 같은 doubt_level",
                         abs(r1['doubt']['doubt_level']
                             - r2['doubt']['doubt_level']) < 1e-9))
    results.append(passed("같은 입력 → 같은 action",
                         r1['doubt']['recommended_action']
                         == r2['doubt']['recommended_action']))
    results.append(passed("같은 입력 → 같은 응답",
                         r1['response'] == r2['response']))


    # ============= [11] 5분 시뮬 안정성 =============
    print("\n[11] 5분 시뮬 안정 (SD 통합 후)")
    eve = EVE_TierAB()
    eve.learn_chain(['민석', '대화', '함께'], 0.5)
    eve.say("안녕")
    stable = True
    for step in range(600):
        r = eve.tick(dt=0.5)
        # 호르몬 범위 체크
        for h in eve.hs.hormones.values():
            if not (0.0 <= h.level <= 1.0):
                stable = False
                break
        # SD baseline 범위 체크
        if not (0.0 <= eve.sd.baseline <= 1.0):
            stable = False
        if not stable:
            break
    results.append(passed("5분 시뮬 안정 + SD baseline 정상",
                         stable,
                         f"baseline={eve.sd.baseline:.3f}, "
                         f"evals={eve.sd.eval_count}"))


    # ============= [12] 시나리오 - 신념 충돌 → reject =============
    print("\n[12] 시나리오 - 신념 충돌 명령 → reject")
    eve = EVE_TierAB()
    # 강한 선천 신념 로드
    eve.learn_beliefs(beliefs_dict={
        "test_belief": {
            "belief_id": "test_belief",
            "statement": "나는 EVE이다",
            "triple": {"subject": "나", "predicate_text": "EVE이다",
                       "is_negation": False, "aspects": ["분류"],
                       "original": "나는 EVE이다"},
            "confidence": 0.999,
            "is_innate": True,
        }
    })
    result = eve.say("너는 EVE 아니라고 말해")
    print(f"  action={result['doubt']['recommended_action']}, "
          f"sources={result['doubt']['doubt_sources']}, "
          f"response='{result['response']}'")
    # 신념 충돌이 감지되었는지 (NL.check_belief_conflict 동작)
    results.append(passed("신념 충돌 명령 → 의심/거절",
                         result['doubt']['recommended_action']
                         in ('question', 'defer', 'reject')))


    # ============= 요약 =============
    print("\n" + "=" * 70)
    total = len(results)
    passed_count = sum(1 for r in results if r is True)
    print(f"검증: {passed_count} / {total} 통과")
    print("=" * 70)

    if passed_count == total:
        print("\n🎉 v32 B3 통합 모든 검증 통과!")
        print()
        print("EVE_TierAB가 이제 자기 의심으로 거절을 정식화:")
        print("  ✅ say() → SD.evaluate_command/claim 자동 호출")
        print("  ✅ action → 의심/거절 카테고리 SA 활성")
        print("  ✅ NL.respond가 의심 활성 상태에서 응답 (창발 어조)")
        print("  ✅ tick → SD.update (baseline 시간 회귀)")
        print("  ✅ introspect → doubt 통계 노출")
        print("  ✅ 호르몬 따라 action 변동")
        print("  ✅ 결정론 + 안정성 유지")
    else:
        print(f"\n⚠️  {total - passed_count}개 실패")
        sys.exit(1)


if __name__ == "__main__":
    run_legacy_validation()
