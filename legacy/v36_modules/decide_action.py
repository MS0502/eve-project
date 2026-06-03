"""
EVE v37.5 — 자율 의사결정 (Action Decision)
============================================

핵심 원칙: EVE가 "위로해야 함"을 강제하지 않는다.
대신 후보를 생성하고, 호르몬/욕구/학습/SelfDoubt가 점수를 매기고,
EVE가 가장 점수 높은 걸 선택. 점수 임계 미달 → "안 함" 도 옵션.

이전 v37 `suggest_action`의 문제:
  if 슬픈친구: return 위로  ← 강제
이것이 EVE 절대 원칙 #2 ("자기 판단해서 명령도 만들고") 위반.

새 구조:
  1) propose_candidates() — 가능한 행동 모두 나열 (강제 X)
  2) score(candidate) — 호르몬+욕구+학습+SelfDoubt+신념
  3) decide() — argmax. 임계 미달이면 None ("그냥 있기" = 능동 선택)

학계:
- Botvinick et al. 2019 — RL fast and slow (가치 함수 = 호르몬 가중)
- Friston 2010 — Active Inference (예측 자유 에너지 최소화)
- Frankfurt 1971 — 2차 욕구 (어떤 욕구를 따를지 결정)
- Damasio 1994 — Somatic Marker (몸 신호로 결정)

Author: 김민석 + Claude v37.5
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ActionCandidate:
    """행동 후보"""
    action: str
    target: Optional[str]
    base_hint: float = 0.0  # 휴리스틱 힌트 (참고용)
    score_breakdown: Dict[str, float] = None  # 점수 출처별 (디버그)
    total_score: float = 0.0


class ActionDecisionMaker:
    """
    EVE의 자율 행동 결정.
    
    환경 인스턴스를 받아 모든 가능 행동 나열,
    각각에 점수 매기고, 가장 좋은 걸 선택.
    """

    # ===== 점수 가중치 (튜닝 가능) =====
    W_HORMONE_NEED = 0.50
    W_PHYSICAL_NEED = 0.50
    W_LEARNED = 0.10
    W_DOUBT = -0.15
    W_FATIGUE = -0.30
    W_DARK_MODE = -0.40
    W_AVOIDANCE = -0.30

    # ★ v37.14 — Goal/Self 강화 (Active Inference prior preferences 기반)
    W_GOAL_ALIGNMENT = 0.50      # was 0.25 — 학계 부합 강화 (EFE prior)
    W_THOUGHT_BIAS = 0.20         # 최근 thought stream 영향
    W_IDENTITY_COHERENCE = 0.40   # 정체성 일치 (긍정), 위반 시 negative
    IDENTITY_VIOLATION_THRESHOLD = -0.50  # 강한 위반은 사실상 veto

    # 휴식 행동은 피로 신호 있을 때만 점수 — 인플레이션 방지
    REST_BASE_BONUS = 0.0  # 기본 보너스 X

    # 점수 임계 — 이 미만이면 "안함"
    MIN_ACTION_SCORE = 0.15

    # 사회 신호 boost (슬픈 친구 + OT 결핍 등)
    SOCIAL_TRIGGER_BOOST = 0.30

    # 행동 카테고리 분류
    SOCIAL_ACTIONS = {'인사하다', '대화하다', '위로하다', '함께있다', '나누다'}
    INGEST_ACTIONS = {'먹다', '마시다'}
    REST_ACTIONS = {'쉬다', '잠자다', '관찰하다', '있다', '눕다'}
    ROOM_ACTIONS = {'눕다', '앉다', '읽다', '쓰다', '창밖_보다'}
    REFLECTION_ACTIONS = {'쓰다', '창밖_보다'}
    TRAVEL_ACTIONS = {'외출', '복귀'}  # ★ v37.10 — 환경 전환

    # 외출 가능한 목적지
    OUTING_DESTINATIONS = ['거실', '주방']

    # 점수 임계 — 이 미만이면 "안함"
    MIN_ACTION_SCORE = 0.15

    # 사회 신호 boost
    SOCIAL_TRIGGER_BOOST = 0.30

    # 일주기 시간대 → 행동 카테고리 적합도 (0=안 맞음, 1=맞음)
    # sim_hour 기반 (0-24)
    @staticmethod
    def _circadian_fit(action: str, sim_hour: float) -> float:
        """
        시간대별 행동 적합도.
        새벽 (0-6): 휴식 강함
        아침 (6-10): 깨어남, 활동 시작
        낮 (10-18): 활동/사회 강함
        저녁 (18-22): 정리/감정 (일기)
        밤 (22-24): 휴식 시작
        """
        h = sim_hour % 24
        # 휴식 행동
        if action in {'눕다', '쉬다', '잠자다'}:
            if h < 6:   return 0.9   # 새벽 — 강하게 적합
            if h < 10:  return 0.2   # 아침 — 일어날 때
            if h < 22:  return 0.0   # 낮/저녁 — 부적합
            return 0.7              # 밤 — 적합
        # 정리/감정 행동
        if action in {'쓰다', '창밖_보다'}:
            if 18 <= h < 23: return 0.6  # 저녁 — 일기 적합
            if 6 <= h < 10:  return 0.3  # 아침 — 약하게
            return 0.2
        # 사회 행동
        if action in {'인사하다', '대화하다', '위로하다', '함께있다', '나누다'}:
            if 9 <= h < 22: return 0.5
            return 0.1   # 새벽/심야 — 부적합
        # 활동 행동 (먹다/읽다/앉다/이동)
        if 8 <= h < 22: return 0.4
        return 0.1

    def __init__(self, eve):
        self.eve = eve
        self.last_decision: Optional[ActionCandidate] = None
        self.decision_history: List[ActionCandidate] = []

    # ============= 1. 후보 생성 =============
    def propose_candidates(self, env) -> List[ActionCandidate]:
        """
        환경의 모든 가능한 행동 나열. 강제 X.
        '안함'(None)도 후보에 포함.
        v37.10 — 자기 방이면 외출 후보, 시뮬 환경이면 복귀 후보 자동 추가.
        """
        candidates: List[ActionCandidate] = []

        # 1) 객체 대상 행동
        objs_here = env.get_objects_here()
        for obj in objs_here:
            for action in obj.available_actions:
                candidates.append(ActionCandidate(
                    action=action,
                    target=obj.name,
                    base_hint=0.1,
                ))

        # 2) NPC 대상 행동 (사회적 환경일 때)
        if hasattr(env, 'get_npcs_here'):
            npcs_here = env.get_npcs_here()
            for npc in npcs_here:
                for action in self.SOCIAL_ACTIONS:
                    candidates.append(ActionCandidate(
                        action=action,
                        target=npc.name,
                        base_hint=0.1,
                    ))

        # 3) 환경 내 위치 이동
        for loc in env.locations:
            if loc != env.current_location:
                candidates.append(ActionCandidate(
                    action='가다',
                    target=loc,
                    base_hint=0.05,
                ))

        # 4) ★ v37.10 — 외출/복귀 후보 (환경 전환)
        is_sim = getattr(env, 'is_simulation', True)
        if not is_sim and env.name == '내방':
            # 자기 방 → 외출 후보
            for dest in self.OUTING_DESTINATIONS:
                candidates.append(ActionCandidate(
                    action='외출',
                    target=dest,
                    base_hint=0.05,
                ))
        elif is_sim:
            # 시뮬 환경 → 복귀 후보
            candidates.append(ActionCandidate(
                action='복귀',
                target='내방',
                base_hint=0.05,
            ))

        # 5) 무행동 옵션
        candidates.append(ActionCandidate(
            action='있다',
            target=None,
            base_hint=0.10,
        ))

        # 6) 휴식 옵션
        candidates.append(ActionCandidate(
            action='쉬다',
            target=None,
            base_hint=0.10,
        ))

        return candidates

    # ============= 2. 점수 매기기 =============
    def score(self, cand: ActionCandidate, env) -> float:
        """
        후보 행동에 점수 부여. 결정론.
        - 휴식: 피로 신호 있을 때만 점수
        - 사회: OT 결핍 + 사회 신호 (슬픈친구 등) 동시 → 큰 점수
        - 학습: 약하게 (시드 인플레이션 방지)
        """
        breakdown = {}
        action = cand.action
        target = cand.target

        # 호르몬 상태
        ot = self.eve.hs.hormones['oxytocin'].level
        da = self.eve.hs.hormones['dopamine'].level
        cort = self.eve.hs.hormones['cortisol'].level
        mel = self.eve.hs.hormones.get('melatonin')
        ade = self.eve.hs.hormones.get('adenosine')
        mel_lvl = mel.level if mel else 0.1
        ade_lvl = ade.level if ade else 0.1

        # 피로 시그널 — 휴식이 의미 있는지 판단용
        fatigue_signal = max(0, mel_lvl - 0.4) + max(0, ade_lvl - 0.4) + max(0, cort - 0.6)

        # ----- (1) 휴식 행동 점수 (피로 의존적) -----
        if action in self.REST_ACTIONS:
            # 피로 없으면 휴식 무의미 — 점수 없음
            if fatigue_signal > 0.05:
                breakdown['rest_need'] = fatigue_signal * 0.6
            else:
                breakdown['rest_idle'] = 0  # 피로 없으면 0
            # 학습 가중치는 휴식 행동에 적용 X (시드 인플레이션 방지)

        # ----- (2) 사회 행동 점수 -----
        if action in self.SOCIAL_ACTIONS:
            ot_need = max(0, 0.5 - ot)
            npc = self._get_npc(env, target)
            if npc:
                if action == '위로하다':
                    if npc.mood in ('슬픔', '아픔') and npc.relationship in ('친구', '가족'):
                        # ★ 강한 trigger — 슬픈 친구 + OT 결핍
                        breakdown['social_trigger'] = self.SOCIAL_TRIGGER_BOOST
                        breakdown['ot_need'] = ot_need * self.W_HORMONE_NEED
                    elif npc.relationship == '낯선' or npc.mood not in ('슬픔', '아픔'):
                        breakdown['social_mismatch'] = -0.30  # 어색
                elif action == '대화하다':
                    if npc.relationship == '낯선' and npc.intimacy < 0.1:
                        breakdown['social_mismatch'] = -0.25
                    else:
                        # 친밀도 따라 reward
                        intimacy_factor = 0.3 + npc.intimacy * 0.7
                        breakdown['ot_need'] = ot_need * intimacy_factor * self.W_HORMONE_NEED
                elif action == '인사하다':
                    # 가벼운 행동 — 거의 항상 OK
                    breakdown['ot_need'] = ot_need * 0.4 * self.W_HORMONE_NEED
                elif action == '나누다':
                    breakdown['ot_need'] = ot_need * 0.6 * self.W_HORMONE_NEED
                elif action == '함께있다':
                    if npc.relationship in ('친구', '가족'):
                        breakdown['ot_need'] = ot_need * 0.8 * self.W_HORMONE_NEED
                        # 가족이면 추가 — 깊은 유대
                        if npc.relationship == '가족':
                            breakdown['family_bond'] = 0.10
                    else:
                        breakdown['social_mismatch'] = -0.15

        # ----- (3) 신체 욕구 (배고픔/목마름) -----
        if action == '먹다' and target:
            obj = self._get_obj(env, target)
            if obj and obj.properties.get('edible'):
                hunger = env.state.get('배고픔', 0)
                breakdown['physical'] = hunger * self.W_PHYSICAL_NEED
            elif obj:
                breakdown['inedible'] = -0.30  # 못 먹는 거 먹기 시도 회피
        elif action == '마시다' and target:
            obj = self._get_obj(env, target)
            if obj and obj.properties.get('drinkable'):
                thirst = env.state.get('목마름', 0)
                breakdown['physical'] = thirst * self.W_PHYSICAL_NEED
            elif obj:
                breakdown['undrinkable'] = -0.30

        # ----- (4) 학습된 가중치 (SA Hebbian) — 약하게만 -----
        # 휴식이 아닌 행동에만, 그것도 약하게 (시드 인플레이션 방지)
        if action not in self.REST_ACTIONS:
            positive_outcomes = ['만족', '회복', '뿌듯', '재밌음', '맛있다']
            if hasattr(self.eve.sa, 'get_weight'):
                sum_pos = 0.0
                for pos in positive_outcomes:
                    sum_pos += self.eve.sa.get_weight(action, pos)
                breakdown['learned_pos'] = sum_pos * self.W_LEARNED

                # 회피 — '실패' 연결 강하면 ↓
                fail_w = self.eve.sa.get_weight(action, '실패')
                if target:
                    fail_w += self.eve.sa.get_weight(target, '실패') * 0.5
                if fail_w > 0:
                    breakdown['avoidance'] = fail_w * self.W_AVOIDANCE

        # ----- (5) SelfDoubt — 능동 행동 위축 -----
        if (hasattr(self.eve, 'sd') and self.eve.sd is not None
                and action not in self.REST_ACTIONS):
            try:
                doubt = getattr(self.eve.sd, 'baseline_doubt', 0)
                if doubt > 0.3:
                    breakdown['doubt'] = doubt * self.W_DOUBT
            except Exception:
                pass

        # ----- (5.4) ★ v37.14 Goal alignment — 강화 (Active Inference prior preferences) -----
        if hasattr(self.eve, 'self_view') and self.eve.self_view is not None:
            try:
                goal_score = self.eve.self_view.goal_alignment_score(action, target)
                if goal_score > 0:
                    breakdown['goal_alignment'] = goal_score * self.W_GOAL_ALIGNMENT
            except Exception:
                pass

        # ----- (5.4.5) ★ v37.14 ThoughtStream bias — 최근 사고가 행동 영향 -----
        # "방금 생각한 것"이 다음 행동을 끌어들이는 자연스러운 효과
        if hasattr(self.eve, 'self_view') and self.eve.self_view is not None:
            try:
                stream = self.eve.self_view.thought_stream(n=3)
                # 최근 thought들의 content가 action/target과 매칭되면 점수 ↑
                action_terms = {action}
                if target:
                    action_terms.add(target)
                bias = 0.0
                for thought in stream[:5]:  # 가장 최근 5개
                    content = thought.get('content', '')
                    if content and content in action_terms:
                        bias += 0.3
                    elif content and any(t in content or content in t for t in action_terms):
                        bias += 0.15
                if bias > 0:
                    breakdown['thought_bias'] = min(0.5, bias) * self.W_THOUGHT_BIAS
            except Exception:
                pass

        # ----- (5.4.6) ★ v37.14 Identity coherence — 정체성 일치 -----
        # 보호 카테고리 (선천 신념, 사용자 정체성)와 정렬되면 점수 ↑
        # 위반하면 강한 negative (사실상 veto)
        protected = getattr(self.eve, 'protected_categories', set())
        if protected:
            # 1) 정렬: action/target이 보호 카테고리거나 SA에서 강하게 연결
            coherence = 0.0
            if action in protected:
                coherence += 0.3
            if target and target in protected:
                coherence += 0.3
            # SA에서 보호 카테고리와의 연결
            for pcat in list(protected)[:10]:  # 성능 — 상위 10개만
                w = self.eve.sa.get_weight(action, pcat)
                if w > 0.4:
                    coherence += 0.1
                if target:
                    w2 = self.eve.sa.get_weight(target, pcat)
                    if w2 > 0.4:
                        coherence += 0.1
            if coherence > 0:
                breakdown['identity_coherence'] = min(1.0, coherence) * self.W_IDENTITY_COHERENCE

            # 2) 위반: '실패' / '거절' / '아니' 와 강하게 연결된 보호 카테고리 행동
            # 즉 EVE의 정체성 위반 → 사실상 veto
            # (지금은 단순 — '나는 EVE 아니다' 류는 SD가 처리. 여기선 강한 negative만)
            violation_signals = ['실패', '아니', '거절']
            for vs in violation_signals:
                w = self.eve.sa.get_weight(action, vs)
                if w > 0.6 and action in protected:
                    # 보호 행동 + 강한 부정 시그널 → veto-level negative
                    breakdown['identity_violation'] = self.IDENTITY_VIOLATION_THRESHOLD
                    break

        # ----- (5.5) 일주기 시간대 적합도 -----
        sim_hour = getattr(self.eve.hs, 'sim_hour', 12.0)
        circadian = self._circadian_fit(action, sim_hour)
        if circadian > 0:
            breakdown['circadian'] = circadian * 0.20

        # ----- (5.6) 방 행동 — 거주지 자율 활동 -----
        # 자기 방에서만 의미 있음. 시뮬 환경엔 침대/책상 없음.
        if action in self.ROOM_ACTIONS:
            # 정리 행동 (쓰다/창밖) — 코르티솔 높을 때 매력 ↑
            if action in self.REFLECTION_ACTIONS and cort > 0.4:
                breakdown['reflection_need'] = (cort - 0.4) * 0.40
            # 읽다 — 도파민 결핍 + 멜라토닌 낮을 때
            if action == '읽다':
                da_low = max(0, 0.4 - da)
                if mel_lvl < 0.5:
                    breakdown['curiosity'] = da_low * 0.30
            # 앉다 — 약한 보너스 (활동 준비)
            if action == '앉다':
                if 6 <= sim_hour < 22:
                    breakdown['posture'] = 0.05

        # ----- (5.6.5) ★ v37.10 — 외출/복귀 점수 -----
        if action in self.TRAVEL_ACTIONS:
            # 체류 시간 (env_adapter)
            adapter = getattr(self.eve, '_env_adapter', None)
            dwell = adapter.get_dwell_time() if adapter else 0.0

            if action == '외출':
                # 외출 = 자기 방 → 시뮬 환경
                # 점수 요소:
                #   - 옥시토신 결핍 (사회 욕구) — 거실 외출
                #   - 도파민 결핍 (자극 욕구) — 어디든
                #   - 자기 방 체류 시간 (오래 있을수록 변화 욕구)
                #   - 일주기 (낮 시간에 활발)
                #   - 코르티솔 ↑ → 외출 X (위축)
                #   - 멜라토닌 ↑ → 외출 X (밤)

                # 사회 욕구 (거실로)
                if target == '거실':
                    ot_need = max(0, 0.4 - ot)
                    breakdown['outing_social'] = ot_need * 0.30
                # 보상 욕구 (주방으로 — 먹기 학습됨)
                if target == '주방':
                    da_need = max(0, 0.4 - da)
                    breakdown['outing_reward'] = da_need * 0.20

                # 체류 변화 욕구 — 자기 방에 너무 오래
                # dwell > 5분 (300초)부터 약하게, 30분 (1800초)에 강하게
                if dwell > 300:
                    dwell_factor = min(1.0, (dwell - 300) / 1500)
                    breakdown['dwell_change'] = dwell_factor * 0.20

                # 일주기 — 낮(9-20시)에 외출 적합
                if 9 <= sim_hour < 20:
                    breakdown['outing_circadian'] = 0.10
                elif sim_hour < 7 or sim_hour >= 22:
                    breakdown['outing_dark'] = -0.30  # 밤엔 외출 X

                # 스트레스 → 위축
                if cort > 0.6:
                    breakdown['outing_stress'] = -(cort - 0.6) * 0.30
                # 피곤 → 외출 X
                if ade_lvl > 0.6:
                    breakdown['outing_tired'] = -(ade_lvl - 0.6) * 0.40

            elif action == '복귀':
                # 복귀 = 시뮬 환경 → 자기 방
                # 점수 요소:
                #   - 시뮬 체류 시간 (오래 있을수록 안정 욕구)
                #   - 코르티솔 ↑ → 안전 욕구 ↑
                #   - 멜라토닌 ↑ → 잠 자고 싶음
                #   - 옥시토신 충족됨 → 사회 욕구 ↓ → 복귀 OK
                #   - 도파민 충족됨 → 자극 욕구 ↓ → 복귀 OK

                # 시뮬 체류 시간 (오래 있을수록 복귀 욕구)
                if dwell > 180:  # 3분 후부터
                    dwell_factor = min(1.0, (dwell - 180) / 1200)
                    breakdown['return_dwell'] = dwell_factor * 0.30

                # 코르티솔 ↑ → 안전한 곳 (자기 방)으로
                if cort > 0.5:
                    breakdown['return_safety'] = (cort - 0.5) * 0.30

                # 멜라토닌 ↑ → 자기 방으로 (잠)
                if mel_lvl > 0.4:
                    breakdown['return_dark'] = (mel_lvl - 0.4) * 0.40

                # 일주기 밤 → 복귀 강함
                if sim_hour >= 22 or sim_hour < 6:
                    breakdown['return_night'] = 0.20

                # 사회/보상 충족 → 복귀 OK
                if ot > 0.65:
                    breakdown['return_satisfied_ot'] = 0.10

        # ----- (5.7) 최근 반복 페널티 — 단조로움 회피 -----
        # 인간은 같은 행동 연속 X. 최근 행동이면 점수 ↓
        if self.decision_history:
            recent_actions = [d.action for d in self.decision_history[-3:]]
            same_count = recent_actions.count(action)
            if same_count > 0:
                breakdown['recency_penalty'] = -0.08 * same_count

        # ----- (6) 일주기/스트레스 — 능동 행동 위축 -----
        if action not in self.REST_ACTIONS:
            # 코르티솔 매우 높음 → 위축
            if cort > 0.7:
                breakdown['cort_stress'] = (cort - 0.7) * self.W_FATIGUE
            # 멜라토닌 ↑ → 활동 ↓
            if mel_lvl > 0.5:
                breakdown['dark_mode'] = (mel_lvl - 0.5) * self.W_DARK_MODE
            # 아데노신 ↑ → 활동 ↓
            if ade_lvl > 0.6:
                breakdown['adenosine_active'] = (ade_lvl - 0.6) * self.W_FATIGUE

        # ----- (7) base hint -----
        breakdown['base'] = cand.base_hint

        # 합산
        total = sum(breakdown.values())
        cand.score_breakdown = breakdown
        cand.total_score = total
        return total

    # ============= 3. 결정 =============
    def decide(self, env, verbose: bool = False) -> Optional[ActionCandidate]:
        """
        후보 생성 → 점수 → 가장 좋은 거 (또는 None).
        결정론적: 동점 시 알파벳 순.
        """
        candidates = self.propose_candidates(env)
        if not candidates:
            return None

        for cand in candidates:
            self.score(cand, env)

        # 정렬: 점수 ↓, 그 다음 결정론 동점 처리
        candidates.sort(key=lambda c: (-c.total_score,
                                       c.action,
                                       c.target or ''))

        best = candidates[0]

        if verbose:
            print(f"  [decide] top 5:")
            for c in candidates[:5]:
                print(f"    {c.action:8s}({c.target or '-':10s}) "
                      f"score={c.total_score:+.3f} "
                      f"breakdown={ {k: round(v,2) for k,v in c.score_breakdown.items()} }")

        # 임계 미달 → 명시적으로 None ("아무것도 안 함" 능동 선택)
        if best.total_score < self.MIN_ACTION_SCORE:
            if verbose:
                print(f"  [decide] 최고 점수 {best.total_score:.3f} < 임계 "
                      f"{self.MIN_ACTION_SCORE} → 안함")
            return None

        self.last_decision = best
        self.decision_history.append(best)
        return best

    # ============= 헬퍼 =============
    def _get_obj(self, env, name: str):
        return env.objects.get(name) if name else None

    def _get_npc(self, env, name: str):
        if not hasattr(env, 'npcs'):
            return None
        return env.npcs.get(name)

    def get_stats(self) -> Dict[str, Any]:
        return {
            'decisions_made': len(self.decision_history),
            'last': (self.last_decision.action, self.last_decision.target,
                    round(self.last_decision.total_score, 3))
                   if self.last_decision else None,
        }
