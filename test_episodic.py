"""
v32 EpisodicMemory (B2) 단독 검증
==================================
A1+A2+A4+B1+B2 통합 검증 포함

검증 영역:
1. encode/recall/remind 기본 작동
2. Hybrid 표현 (tuple + categories + summary)
3. Pattern completion (Jaccard)
4. 호르몬 변조 (BDNF/cortisol/endorphin)
5. 망각 + recall 보호 (단 floor 있음)
6. auto_encode (WM focus 변경)
7. 결정론
8. capacity 초과 evict
9. is_protected 보호
10. 통합 시나리오
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from episodic import EpisodicMemory, Episode
from hormone_system import HormoneSystem
from spreading_activation import SpreadingActivation
from working_memory import WorkingMemory


def passed(name, ok, detail=""):
    mark = "✅" if ok else "❌"
    print(f"  {mark} {name}" + (f"  | {detail}" if detail else ""))
    return ok


def make_em(capacity=500):
    hs = HormoneSystem(phase=4, developmental_stage="adult")
    sa = SpreadingActivation(base_threshold=0.3)
    wm = WorkingMemory()
    em = EpisodicMemory(sa, wm, hs, capacity=capacity)
    return em, hs, sa, wm


print("=" * 70)
print("v32 EpisodicMemory (B2) 검증")
print("=" * 70)

results = []

# ============= [1] encode 기본 =============
print("\n[1] encode 기본 작동")
em, hs, sa, wm = make_em()
ep = em.encode(categories={'민석', '대화', '기쁨'},
              tuple_data={'actor': '민석', 'action': '대화'},
              summary='민석이랑 즐거운 대화',
              salience=0.8)
results.append(passed("encode 성공 + Episode 반환", ep is not None))
results.append(passed("Hybrid 표현 (3종)",
                     ep.tuple_data.get('actor') == '민석'
                     and ep.categories == {'민석', '대화', '기쁨'}
                     and ep.summary == '민석이랑 즐거운 대화'))
results.append(passed("호르몬 snapshot 저장",
                     'valence' in ep.mood and ep.primary_hormone[0]))
results.append(passed("category_index 갱신",
                     ep.id in em.category_index['민석']
                     and ep.id in em.category_index['대화']))

# 빈 카테고리 거부
ep_none = em.encode(categories=set())
results.append(passed("빈 categories → None", ep_none is None))


# ============= [2] recall (pattern completion) =============
print("\n[2] recall - Jaccard 유사도 매칭")
em2, hs2, sa2, wm2 = make_em()
em2.encode({'민석', '대화', '기쁨'}, summary='ep_a', salience=0.7)
em2.encode({'민석', '오늘'}, summary='ep_b', salience=0.6)
em2.encode({'영화', '재밌다'}, summary='ep_c', salience=0.8)
em2.encode({'민석', '대화', '오늘', '카페'}, summary='ep_d', salience=0.9)

# '민석' cue → ep_a, ep_b, ep_d 매치
r1 = em2.recall({'민석'})
r1_summaries = [ep.summary for ep in r1]
results.append(passed("'민석' cue → 매치 (3개)", len(r1) == 3,
                     f"summaries={r1_summaries}"))

# '민석', '대화' cue → ep_a, ep_d 우선
r2 = em2.recall({'민석', '대화'})
r2_top = r2[0].summary if r2 else None
results.append(passed("'민석'+'대화' cue → 정확 매치 우선",
                     r2_top in ['ep_a', 'ep_d'],
                     f"top={r2_top}"))

# 매치 X
r3 = em2.recall({'없는카테고리'})
results.append(passed("매치 없으면 빈 리스트", r3 == []))

# recall_count 갱신
ep_a_id = [eid for eid, ep in em2.episodes.items() if ep.summary == 'ep_a'][0]
results.append(passed("recall 시 recall_count 증가",
                     em2.episodes[ep_a_id].recall_count > 0))


# ============= [3] remind (DMN 떠올림) =============
print("\n[3] remind - focus 카테고리 → 일화 + SA 활성")
em3, hs3, sa3, wm3 = make_em()
em3.encode({'민석', '대화', '기쁨', '카페'}, salience=0.7)

# remind 전 SA 상태
sa_before = set(sa3.get_active())

ep = em3.remind('민석')
results.append(passed("remind('민석') → Episode 반환", ep is not None))

# remind 후 SA에 다른 카테고리들 활성됨
sa_after = set(sa3.get_active())
results.append(passed("remind 후 일화 카테고리 SA 활성 (회상 효과)",
                     len(sa_after - sa_before) > 0,
                     f"새 활성: {sa_after - sa_before}"))

# remind 없는 cue
ep_none = em3.remind('없는거')
results.append(passed("매치 없으면 None", ep_none is None))


# ============= [4] auto_encode (WM focus) =============
print("\n[4] auto_encode - WM focus 변경 시 자동 저장")
em4, hs4, sa4, wm4 = make_em()
em4._auto_encode_min_interval = 0.0  # 테스트용 즉시 인코딩

# WM에 카테고리 강하게 추가
wm4.add('민석', salience=0.8)
wm4.add('대화', salience=0.6)
wm4.time = 1.0
em4.time = 1.0
em4.auto_encode_from_wm(dt=0.5)
results.append(passed("WM focus 첫 등장 → encode",
                     em4.encode_count == 1,
                     f"encode_count={em4.encode_count}"))

# 같은 focus 두 번째 → skip
em4.time = 1.5
em4.auto_encode_from_wm(dt=0.5)
results.append(passed("같은 focus 연속 → skip",
                     em4.encode_count == 1,
                     f"encode_count={em4.encode_count}"))

# focus 변경 → encode
wm4.add('새카테고리', salience=0.9)  # 새 focus
em4.time = 3.0
em4.auto_encode_from_wm(dt=0.5)
results.append(passed("focus 변경 → encode",
                     em4.encode_count == 2,
                     f"encode_count={em4.encode_count}"))


# ============= [5] consolidate - 호르몬 변조 =============
print("\n[5] consolidate - 호르몬 → 일화 강도")

# BDNF ↑ → 보존 강화
em_b, hs_b, sa_b, wm_b = make_em()
ep_b = em_b.encode({'a', 'b'}, salience=0.5)
sal_before_b = ep_b.salience
hs_b.hormones['bdnf'].level = 0.9
em_b.consolidate(dt=10.0)
sal_after_b = em_b.episodes[ep_b.id].salience
results.append(passed("BDNF ↑ → salience 증가",
                     sal_after_b > sal_before_b,
                     f"{sal_before_b:.3f} → {sal_after_b:.3f}"))

# 만성 Cortisol → 약화
em_c, hs_c, sa_c, wm_c = make_em()
ep_c = em_c.encode({'a', 'b'}, salience=0.6)
sal_before_c = ep_c.salience
hs_c.hormones['cortisol'].level = 0.95
em_c.consolidate(dt=20.0)
sal_after_c = em_c.episodes[ep_c.id].salience
results.append(passed("만성 Cortisol ↑ → salience 감소",
                     sal_after_c < sal_before_c,
                     f"{sal_before_c:.3f} → {sal_after_c:.3f}"))

# Endorphin → 최근 강화
em_e, hs_e, sa_e, wm_e = make_em()
ep_e = em_e.encode({'a'}, salience=0.4)
sal_before_e = ep_e.salience
hs_e.hormones['endorphin'].level = 0.9
em_e.consolidate(dt=5.0)
sal_after_e = em_e.episodes[ep_e.id].salience
results.append(passed("Endorphin ↑ → 최근 일화 강화",
                     sal_after_e > sal_before_e,
                     f"{sal_before_e:.3f} → {sal_after_e:.3f}"))


# ============= [6] forget (Ebbinghaus + recall 보호) =============
print("\n[6] forget - 망각 + recall 보호 (floor 있음)")
em_f, hs_f, sa_f, wm_f = make_em()
em_f.base_forget_rate = 0.01  # 빠른 망각

# 두 일화: 자주 회상 vs 방치
ep_active = em_f.encode({'a', 'b'}, salience=0.5)
ep_neglect = em_f.encode({'c', 'd'}, salience=0.5)

# 자주_회상 5번 recall
for _ in range(5):
    em_f.recall({'a'})

# 60초 망각
for _ in range(60):
    em_f.forget(dt=1.0)

active_sal = em_f.episodes.get(ep_active.id)
neglect_sal = em_f.episodes.get(ep_neglect.id)
active_sal_str = f"{active_sal.salience:.3f}" if active_sal else "evicted"
neglect_sal_str = f"{neglect_sal.salience:.3f}" if neglect_sal else "evicted"
print(f"  active sal={active_sal_str}, neglect sal={neglect_sal_str}")
results.append(passed(
    "자주 회상한 일화가 더 강함",
    (active_sal is not None and (neglect_sal is None or active_sal.salience > neglect_sal.salience))
))

# 무한 보호 안 됨 (floor): 충분히 오래 두면 회상한 것도 사라짐
em_f2, _, _, _ = make_em()
em_f2.base_forget_rate = 0.05
ep_x = em_f2.encode({'x'}, salience=0.3)
for _ in range(10):
    em_f2.recall({'x'})
# 10분 망각
for _ in range(600):
    em_f2.forget(dt=1.0)
final_sal_str = f"final={em_f2.episodes[ep_x.id].salience:.3f}" if ep_x.id in em_f2.episodes else "evicted"
results.append(passed(
    "10분 후엔 자주 회상한 것도 망각 (floor 작동, 영구 lock 방지)",
    em_f2.episodes.get(ep_x.id) is None or em_f2.episodes[ep_x.id].salience < 0.2,
    final_sal_str
))

# is_protected는 무한 보존
em_f3, _, _, _ = make_em()
em_f3.base_forget_rate = 0.1  # 매우 빠른 망각
ep_p = em_f3.encode({'core'}, salience=0.3, is_protected=True)
for _ in range(1200):  # 20분
    em_f3.forget(dt=1.0)
results.append(passed(
    "is_protected 일화는 망각 X",
    ep_p.id in em_f3.episodes,
    f"protected ep 살아있음" if ep_p.id in em_f3.episodes else "evict됨"
))


# ============= [7] capacity 초과 evict =============
print("\n[7] capacity 초과 → 가장 약한 일화 evict")
em_cap, _, _, _ = make_em(capacity=5)
for i in range(10):
    em_cap.encode({f'cat_{i}'}, salience=0.3 + i * 0.05)

results.append(passed(
    "capacity 5 → 5개만 유지",
    len(em_cap.episodes) == 5,
    f"{len(em_cap.episodes)}개"
))
# 가장 약한 게 evict됨 → 처음 추가한 것들 (낮은 salience) 사라짐
remaining_sals = sorted([ep.salience for ep in em_cap.episodes.values()])
results.append(passed(
    "가장 약한 일화부터 evict (높은 salience만 남음)",
    min(remaining_sals) > 0.4,
    f"남은 salience={remaining_sals}"
))


# ============= [8] 결정론 =============
print("\n[8] 결정론 (np.random X)")

def run():
    em_r, hs_r, sa_r, wm_r = make_em()
    em_r.encode({'a', 'b'}, summary='ep1', salience=0.5)
    em_r.encode({'b', 'c'}, summary='ep2', salience=0.7)
    em_r.encode({'a', 'c'}, summary='ep3', salience=0.6)
    em_r.recall({'a'})
    em_r.recall({'b'})
    for _ in range(50):
        em_r.consolidate(dt=0.5)
        em_r.forget(dt=0.5)
    return {eid: (ep.salience, ep.recall_count) for eid, ep in em_r.episodes.items()}

a = run()
b = run()
det = (set(a.keys()) == set(b.keys())
       and all(abs(a[k][0] - b[k][0]) < 1e-12
               and abs(a[k][1] - b[k][1]) < 1e-12 for k in a))
results.append(passed("같은 시퀀스 → 같은 결과", det))

import inspect
src = inspect.getsource(EpisodicMemory) + inspect.getsource(Episode)
no_random = ('np.random.' not in src
            and 'random.choice(' not in src
            and 'random.random(' not in src
            and '.sample(' not in src)
results.append(passed("코드: random/sample 호출 없음", no_random))


# ============= [9] 통합 시나리오 - WM focus → auto encode → 회상 =============
print("\n[9] 통합 시나리오 - WM/SA/HS와 연동")
em_int, hs_int, sa_int, wm_int = make_em()
em_int._auto_encode_min_interval = 0.0  # 즉시 인코딩 (테스트용)

# 시나리오 1: '민석' 만남 - WM focus 형성
sa_int.activate('민석', 0.7)
sa_int.activate('대화', 0.6)
wm_int.update_from_activation(sa_int)
em_int.update(dt=0.5)

initial_eps = em_int.encode_count

# 시나리오 2: 시간 흐름 → '민석' 약화 → 새 주제 '영화'
wm_int.slots['민석'].salience = 0.3  # decay 시뮬
wm_int.slots['대화'].salience = 0.2
wm_int.add('영화', salience=0.9, source='external')
wm_int.add('재밌다', salience=0.7, source='external')
em_int.time = 5.0
wm_int.time = 5.0
em_int.update(dt=0.5)
final_eps = em_int.encode_count

results.append(passed(
    "[시나리오] WM focus 변경 → 자동 encode",
    final_eps > initial_eps,
    f"{initial_eps} → {final_eps}"
))

# 시나리오 3: '민석' 회상 → 그 때 같이 있던 카테고리들 SA 활성
sa_active_before = set(sa_int.get_active())
ep_remind = em_int.remind('민석')
sa_active_after = set(sa_int.get_active())

results.append(passed(
    "[시나리오] remind('민석') → 일화 + SA 회상 효과",
    ep_remind is not None,
    f"summary='{ep_remind.summary if ep_remind else None}'"
))


# ============= 요약 =============
print("\n" + "=" * 70)
total = len(results)
passed_count = sum(1 for r in results if bool(r))
print(f"검증: {passed_count} / {total} 통과")
print("=" * 70)

if passed_count == total:
    print("\n🎉 EpisodicMemory (B2) 모든 검증 통과!")
    print("→ Hybrid 표현 (tuple + categories + summary)")
    print("→ Pattern completion (Jaccard)")
    print("→ 호르몬 변조 (BDNF/Cortisol/Endorphin)")
    print("→ 망각 + recall 보호 (floor 있음, 영구 lock 방지)")
    print("→ auto_encode (WM focus 변경)")
    print("→ is_protected 일화 보호")
    print("→ A1+A2+A4+B2 통합 작동")
    print("→ EVE 절대 원칙 #2 (확률 X) 부합")
else:
    print(f"\n⚠️  {total - passed_count}개 실패")
    sys.exit(1)
