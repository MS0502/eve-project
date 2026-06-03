"""
EVE v32 - Tier A + B + C 완성본 ★★★
======================================

19 모듈 통합 — 티어 A + B + C (1~7) 모두 완성:
- HormoneSystem (26 호르몬)
- SpreadingActivation (카테고리 활성 + 자기 진화)
- WorkingMemory + GNW broadcast
- DMN (자발 활성 + self_intent)
- DigitalSomatic (신체 감각)
- NaturalLanguage (자연어 + 신념 + 거절 창발)
- EpisodicMemory (Hybrid: tuple + 카테고리 + 자연어) ← B2
- SelfDoubt (5종 신호 통합) ← B3
- ActiveInference (Friston FEP, 호르몬-outcome 학습) ← B4
- Agency (Haggard sense of agency, 6 시그널 → 행동 결정) ← B5
- MetaCognition (Flavell, Frankfurt, 사고 원인 추적) ← B6
- CausalGraph (Pearl 인과 DAG, Granger 시간 우선) ← C1
- WorldModel (통합 메타 view, 도메인/스키마/관계) ← C2 NEW
- Counterfactual (Lewis closest world, regret/relief) ← C3
- Analogy (Gentner Structure-Mapping) ← C4
- Temporal (Tulving chronesthesia, mental time travel) ← C5
- GoalManagement (Schmidhuber/Carver, 목표 추구) ← C6
- EmotionRegulation (Gross 인지 재평가, distraction, suppression) ← C7

★ 티어 C 7/7 완성. EVE의 인지 아키텍처 완비. ★

EVE 메인 루프 (tick 1회):
1. HormoneSystem.update(dt)
2. 호르몬 → SA 임계 변조 (Top-down)
3. SA.spread()
4. SA.decay(dt)
5. SA → 카테고리 → 호르몬 자극 (Bottom-up)
6. WM.update_from_activation(SA)
7. WM.apply_hormone_state(HS)
8. WM.broadcast()                         - GNW
9. WM.decay(dt)
10. DMN.tick(dt)                          - 자발 활성
    └→ memory_recall 모드면 em.remind() 호출 ← NEW B2
11. DS.update(dt)
11.5 EM.update(dt)                         ← NEW B2 (auto_encode + consolidate + forget)
12. SD.update(dt)                          (baseline 시간 회귀)
13. (주기적) SA.forget(dt, hormone_modifier)

신규 외부 인터페이스 (B2 통합):
- self.em 직접 접근 (em.encode/recall/remind)
- introspect()['episodes_count'] 등 통계 6종
- SD가 em.recall로 부정 일화 검색 → episodic_negative doubt 신호 작동
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eve_modules'))

from typing import Dict, List, Set, Tuple, Optional, Any
from hormone_system import HormoneSystem
from spreading_activation import SpreadingActivation
from working_memory import WorkingMemory
from dmn import DefaultModeNetwork
from digital_somatic import DigitalSomatic
from natural_lang import NaturalLanguage
from self_doubt import SelfDoubt
from episodic import EpisodicMemory  # ← NEW B2 정식 통합
from active_inference import ActiveInference  # ← NEW B4 정식 통합
from agency import Agency  # ← NEW B5 정식 통합
from metacognition import MetaCognition  # ← NEW B6 정식 통합
from causal_graph import CausalGraph  # ← NEW C1 정식 통합
from counterfactual import Counterfactual  # ← NEW C3 정식 통합
from goal_management import GoalManagement  # ← NEW C6 정식 통합
from temporal import Temporal  # ← NEW C5 정식 통합
from emotion_regulation import EmotionRegulation  # ← NEW C7 정식 통합
from analogy import Analogy  # ← NEW C4 정식 통합
from world_model import WorldModel  # ← NEW C2 정식 통합 (티어 C 마지막)
from creative import Creative  # ← NEW D1 정식 통합 (티어 D 시작)
from suffering import Suffering  # ← NEW D2 정식 통합
from humor import Humor  # ← NEW D3 정식 통합
from multi_stream import MultiStream  # ← NEW D4 정식 통합
from tool_use import ToolUse  # ← NEW D5 정식 통합
from narrative_self import NarrativeSelf  # ← NEW D6 정식 통합 (티어 D 마지막)
from corpus_learner import CorpusLearner  # ← NEW v35 E1 (Phase 1)
from apc_learner import APCLearner  # ← NEW v35 E3 (Phase 2)
from deep_reasoning import DeepReasoning  # ← NEW v35 E4 (Phase 3)
from vsa_binding import VSABinding  # ← NEW v35 E5 (Phase 4)
from continual_rehearsal import ContinualRehearsal  # ← NEW v35 E6 (Phase 5)
from hypergraph import HyperGraph, HyperEdge  # ← NEW v35 E2 (Phase 6)
from creative_advanced import CreativeAdvanced  # ← NEW v35 E7 (Phase 7)
from frame_semantics import FrameSemantics  # ← NEW v35 E8 (Phase 8)
from semantic_distance import SemanticDistance  # ← NEW v35 E9 (Phase 9)
from norm_internal import NormInternalization  # ← NEW v35 F1 (사용자 명령 내재화)
from allostatic_learn import AllostaticLearning  # ← NEW v35 F2 (호르몬 칵테일 학습)


class EVE_TierAB:
    """
    EVE v32 티어 A + B (B1 + B3) 통합.

    살아있음(A) + 자연어 대화(B1) + 자기 의심(B3).
    """

    # SelfDoubt action → SA에 broadcast할 카테고리 + 강도
    # 창발 어조를 위해 NL.respond 직전에 활성
    DOUBT_ACTION_BROADCAST = {
        'accept':   [],  # 활성 X (NL 자연 응답 그대로)
        'question': [('의심', 0.5), ('궁금', 0.4)],
        'defer':    [('의심', 0.6), ('회피', 0.4)],
        'reject':   [('거절', 0.7), ('싫다', 0.5), ('의심', 0.5)],
    }

    def __init__(self,
                 phase: int = 4,
                 stage: str = "adult",
                 wm_capacity: int = 30,
                 dmn_idle_threshold: float = 2.0,
                 forget_interval_ticks: int = 100,
                 forget_base_decay: float = 0.0005,
                 verbose: bool = False):
        # 7 + 1 모듈 인스턴스화
        self.hs = HormoneSystem(phase=phase, developmental_stage=stage)
        self.sa = SpreadingActivation(base_threshold=0.3)
        self.wm = WorkingMemory(base_capacity=wm_capacity)
        self.dmn = DefaultModeNetwork(self.sa, self.wm, self.hs,
                                     digital_somatic=None,
                                     idle_threshold=dmn_idle_threshold)
        self.ds = DigitalSomatic(self.hs, self.wm, self.sa)
        self.dmn.ds = self.ds
        self.nl = NaturalLanguage(self.sa, self.wm, self.hs,
                                 digital_somatic=self.ds)

        # ★ B2 정식 통합: EpisodicMemory (NL 다음, SD 이전 — SD가 em 참조)
        self.em = EpisodicMemory(
            spreading_activation=self.sa,
            working_memory=self.wm,
            hormone_system=self.hs,
            natural_lang=self.nl,
        )

        # ★ B3 정식 통합: SelfDoubt에 EM 연결 (이전 None → self.em)
        self.sd = SelfDoubt(
            hormone_system=self.hs,
            spreading_activation=self.sa,
            natural_lang=self.nl,
            episodic_memory=self.em,  # ← 정식 연결 → episodic_negative 신호 활성화
            digital_somatic=self.ds,
        )

        # ★ B4 정식 통합: ActiveInference (SD 다음 — SD의 doubt가 outcome 학습 트리거)
        # Friston FEP — 호르몬 상태 → outcome 예측 → 결과로 매핑 학습
        self.ai = ActiveInference(
            hormone_system=self.hs,
            spreading_activation=self.sa,
            working_memory=self.wm,
            self_doubt=self.sd,
            digital_somatic=self.ds,
        )

        # ★ B5 정식 통합: Agency (AI 다음 — 모든 모듈 통합)
        # Haggard 2017 sense of agency + Damasio gut signal → 행동 결정
        self.ag = Agency(
            hormone_system=self.hs,
            spreading_activation=self.sa,
            working_memory=self.wm,
            digital_somatic=self.ds,
            self_doubt=self.sd,
            dmn=self.dmn,
            active_inference=self.ai,
        )

        # ★ B6 정식 통합: MetaCognition (Agency 다음, 모든 모듈 read-only 관찰)
        # Flavell 메타인지 + Frankfurt 2차 욕구 → 사고/욕구 원인 추적
        self.mc = MetaCognition(
            hormone_system=self.hs,
            spreading_activation=self.sa,
            working_memory=self.wm,
            digital_somatic=self.ds,
            self_doubt=self.sd,
            active_inference=self.ai,
            dmn=self.dmn,
            episodic_memory=self.em,
            agency=self.ag,
        )

        # ★ C1 정식 통합: CausalGraph (Pearl 인과 DAG, 마지막)
        # SA history에서 자동 인과 학습 + why/what_if/intervene 질의
        self.cg = CausalGraph(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            episodic_memory=self.em,
            active_inference=self.ai,
        )
        # observe_temporal 간격 (매 tick 호출하면 비용 큼)
        self._cg_observe_interval = 20  # 매 20 tick

        # ★ C4 정식 통합: Analogy (CG 다음, read-only)
        # Gentner Structure-Mapping — 관계 패턴 매칭
        self.an = Analogy(
            causal_graph=self.cg,
            spreading_activation=self.sa,
            hormone_system=self.hs,
        )

        # ★ C3 정식 통합: Counterfactual (CG 다음)
        # Lewis closest world + Pearl 3단계 + Roese regret/relief
        self.cf = Counterfactual(
            causal_graph=self.cg,
            spreading_activation=self.sa,
            hormone_system=self.hs,
            agency=self.ag,
            active_inference=self.ai,
            self_doubt=self.sd,
        )

        # ★ C5 정식 통합: Temporal (read-only on EM/SA/CG/HS)
        # Tulving chronesthesia — 과거 회상 + 미래 시뮬 + 주관적 시간
        self.tm = Temporal(
            episodic_memory=self.em,
            spreading_activation=self.sa,
            causal_graph=self.cg,
            hormone_system=self.hs,
        )

        # ★ C7 정식 통합: EmotionRegulation (CF + TM 다음)
        # Gross 인지 3전략 — distraction / reappraisal / suppression
        self.er = EmotionRegulation(
            hormone_system=self.hs,
            spreading_activation=self.sa,
            working_memory=self.wm,
            counterfactual=self.cf,
            temporal=self.tm,
            episodic_memory=self.em,
        )
        # 자동 조절 간격 (매 tick 호출하면 과조절)
        self._er_auto_interval = 30  # 매 30 tick

        # ★ C2 정식 통합: WorldModel (모든 모듈 read-only 통합 view, 마지막)
        # 티어 C 마지막 — 다른 모듈들의 정보를 종합한 메타 view
        self.wm_world = WorldModel(
            spreading_activation=self.sa,
            causal_graph=self.cg,
            episodic_memory=self.em,
            analogy=self.an,
            temporal=self.tm,
            hormone_system=self.hs,
        )

        # ★ C6 정식 통합: GoalManagement (CF 다음, 모든 의존 모듈 합류)
        # Schmidhuber + Carver/Scheier + Murphy TOTE — 목표 추구
        self.gm = GoalManagement(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            digital_somatic=self.ds,
            causal_graph=self.cg,
            counterfactual=self.cf,
            episodic_memory=self.em,
            agency=self.ag,
        )

        # ★ D1 정식 통합: Creative (티어 D 첫 모듈)
        # Mednick RAT + Boden blend + Wallas 4단계
        # 호르몬 게이트 (도파민↑ + cort↓), DMN 협력, 다른 모듈 read-only
        self.cr = Creative(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            analogy=self.an,           # C4
            causal_graph=self.cg,      # C1
            episodic_memory=self.em,   # B2
            dmn=self.dmn,              # A
            working_memory=self.wm,    # A
            agency=self.ag,            # B5
            self_doubt=self.sd,        # B3
        )

        # ★ D2 정식 통합: Suffering (공감)
        # Singer 2009 affective+cognitive empathy
        # 자기 hs는 자기 거라 공명 변조 OK, 다른 모듈 read-only
        self.suf = Suffering(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            episodic_memory=self.em,    # 친밀도
            self_doubt=self.sd,          # self-other 구분 보조
            working_memory=self.wm,
            counterfactual=self.cf,      # "그 입장이라면"
            digital_somatic=self.ds,
            natural_lang=self.nl,
        )

        # ★ D3 정식 통합: Humor (농담)
        # Suls 1972 + McGraw 2010 + Wyer 1992 + Vrticka 2013
        # 호르몬 게이트 (DA↑+cort↓), CR.RAT 위임, SUF.is_suffering 안전
        self.hu = Humor(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            creative=self.cr,            # D1 — RAT 위임
            analogy=self.an,             # C4
            causal_graph=self.cg,        # C1
            suffering=self.suf,          # D2 — 안전 검증
            episodic_memory=self.em,
            working_memory=self.wm,
        )

        # ★ D4 정식 통합: MultiStream (평행 사고)
        # James 1890 + Baars 1988 GWT + Dehaene 2001 GNW + Kahneman 2011
        # 5 streams: external/inner/emotion/planning/recall
        # 다른 모듈 read-only, 자기 큐만 변경
        self.ms = MultiStream(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            working_memory=self.wm,
            dmn=self.dmn,                # inner stream
            episodic_memory=self.em,     # recall stream
            temporal=self.tm,            # planning stream
            goal_management=self.gm,     # planning stream
            digital_somatic=self.ds,     # emotion stream
            natural_lang=self.nl,
        )

        # ★ D5 정식 통합: ToolUse (메타 도구)
        # Vygotsky 1978 + Kahneman 2011 System 2 + Bratman 1987 intention
        # EVE가 자기 모듈을 의식적 도구로 등록 + 호출
        self.tu = ToolUse(
            hormone_system=self.hs,
            spreading_activation=self.sa,
            working_memory=self.wm,
            self_doubt=self.sd,
            metacognition=self.mc,
            agency=self.ag,
        )
        # 23 모듈 중 호출 가능한 것들을 도구로 자동 등록
        self._register_default_tools()

        # ★ D6 정식 통합: NarrativeSelf (자전적 정체성, 티어 D 마지막!)
        # McAdams 1993 + Damasio 1999 + Bruner 1991 + Singer 2004
        # EM 일화 + AG 행동 → "나는 X하는 자" 자기 서사
        # 다른 모듈 read-only
        self.ns = NarrativeSelf(
            episodic_memory=self.em,
            spreading_activation=self.sa,
            hormone_system=self.hs,
            agency=self.ag,                # 자기 행동
            temporal=self.tm,              # 시간 흐름
            world_model=self.wm_world,    # 통합 view
            working_memory=self.wm,
        )
        # ★ Fix: 핵심 정체성 자동 protect (decay floor 0.5 보장)
        self.ns.protect('나', 'EVE')

        # ★ v35 E1: CorpusLearner — 대규모 텍스트 학습 (카테고리 폭증)
        self.cl = CorpusLearner(
            spreading_activation=self.sa,
            natural_lang=self.nl,
            hormone_system=self.hs,
        )

        # ★ v35 E3: APCLearner — 매 tick 평생 학습 (Active Predictive Coding)
        self.apc = APCLearner(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            working_memory=self.wm,
        )

        # ★ v35 E4: DeepReasoning — 멀티홉 path-based 추론
        self.dr = DeepReasoning(
            spreading_activation=self.sa,
            causal_graph=self.cg,
            hormone_system=self.hs,
            natural_lang=self.nl,
        )

        # ★ v35 E5: VSABinding — 의미 합성 (창의 핵심)
        self.vsa = VSABinding(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            vector_dim=512,
        )

        # ★ v35 E6: ContinualRehearsal — 정체성 보존
        # NS 다음에 와야 함 (ns 의존성)
        self.cr_rehearsal = ContinualRehearsal(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            narrative_self=self.ns,
            working_memory=self.wm,
        )

        # ★ v35 E2: HyperGraph — n-ary 관계 (multi-entity)
        self.hg = HyperGraph(
            spreading_activation=self.sa,
            hormone_system=self.hs,
        )

        # ★ v35 E7: CreativeAdvanced — 창의 사이클 강화
        # vsa, hg, dr, cr 등 모두 의존 → 마지막에 와야 함
        self.ca = CreativeAdvanced(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            vsa=self.vsa,
            hypergraph=self.hg,
            deep_reasoning=self.dr,
            creative=self.cr,  # 기존 v32 Creative
            dmn=self.dmn,
            apc=self.apc,
        )

        # ★ v35 E8: FrameSemantics — 한국어 SVO frame
        self.fs = FrameSemantics(
            natural_lang=self.nl,
            hypergraph=self.hg,
            spreading_activation=self.sa,
            hormone_system=self.hs,
        )

        # ★ v35 E9: SemanticDistance — 다중 신호 의미 거리
        self.sd_dist = SemanticDistance(
            spreading_activation=self.sa,
            vsa_binding=self.vsa,
            hypergraph=self.hg,
            hormone_system=self.hs,
        )

        # ★ v35 F1: NormInternalization — 사용자 명령 내재화
        self.ni = NormInternalization(
            spreading_activation=self.sa,
            hormone_system=self.hs,
            natural_lang=self.nl,
            dmn=self.dmn,
            default_agent='민석',
        )

        # ★ v35 F2: AllostaticLearning — 호르몬 칵테일 적응 학습
        self.al = AllostaticLearning(
            hormone_system=self.hs,
            spreading_activation=self.sa,
            working_memory=self.wm,
        )

        self.verbose = verbose
        self.time = 0.0
        self.tick_count = 0

        self.forget_interval_ticks = forget_interval_ticks
        self.forget_base_decay = forget_base_decay

        self.protected_categories: Set[str] = set()

        self.event_log: List[Dict[str, Any]] = []
        self.broadcast_count = 0
        self.dialogue_history: List[Dict[str, Any]] = []
        self.forget_runs = 0

        self._setup_gnw_listeners()

    def _setup_gnw_listeners(self):
        def gnw_to_hormone(focus, salience, ctx):
            self.hs.category_to_hormone({focus}, strength=0.05 * salience)
            self.broadcast_count += 1
        self.wm.register_listener('hormone_modulator', gnw_to_hormone)

    def _register_default_tools(self):
        """
        ★ D5: 23 모듈 중 의식적 호출 가능한 도구 자동 등록.
        signature는 어떤 호르몬/카테고리 상황에 유리한지.
        """
        # 자기조절 (regulation)
        self.tu.register_tool(
            'reappraise', self.reappraise,
            {'cortisol_high': 1.0, 'negative_active': 0.7},
            'Gross 인지 재평가 — cort↓, 5HT↑',
            'regulation')
        self.tu.register_tool(
            'distract', self.distract,
            {'cortisol_high': 0.8, 'negative_active': 0.5},
            '주의 전환 → 긍정 카테고리',
            'regulation')
        self.tu.register_tool(
            'suppress_emotion', self.suppress_emotion,
            {'ne_high': 0.5, 'cortisol_high': 0.6},
            '반응 억제 (비용: NE↑)',
            'regulation')

        # 추론 (reasoning)
        self.tu.register_tool(
            'why_caused', lambda cat: self.why_caused(cat),
            {'question_active': 1.0, 'cortisol_low': 0.3},
            'CG 인과 부모 조회',
            'reasoning')
        self.tu.register_tool(
            'what_if', lambda cat: self.what_if(cat),
            {'question_active': 0.8, 'dopamine_high': 0.5},
            'CG 인과 자식 시뮬',
            'reasoning')
        self.tu.register_tool(
            'what_if_not', lambda cat: self.what_if_not(cat),
            {'question_active': 0.7, 'cortisol_high': 0.5},
            'CF 가상 - 일어나지 않았다면',
            'reasoning')

        # 회상 (recall)
        self.tu.register_tool(
            'recall_past', lambda cat: self.recall_past(cat),
            {'melatonin_high': 1.0, 'has_focus': 0.5},
            'EM 일화 회상',
            'recall')
        self.tu.register_tool(
            'imagine_future', lambda seed: self.imagine_future(seed),
            {'dopamine_high': 0.7, 'question_active': 0.5},
            'TM 미래 시뮬',
            'simulation')

        # 창의 (creative)
        self.tu.register_tool(
            'find_remote', lambda seeds: self.find_remote(seeds),
            {'dopamine_high': 1.0, 'cortisol_low': 0.5},
            'D1 Mednick RAT',
            'creative')
        self.tu.register_tool(
            'blend', lambda a, b: self.blend(a, b),
            {'dopamine_high': 1.0, 'cortisol_low': 0.5},
            'D1 Boden blend',
            'creative')
        self.tu.register_tool(
            'joke', lambda seed: self.joke(seed),
            {'dopamine_high': 1.0, 'cortisol_low': 0.7},
            'D3 농담 생성',
            'creative')

        # 사회 (social)
        self.tu.register_tool(
            'empathize',
            lambda target, suffering: self.empathize(target, suffering),
            {'oxytocin_high': 0.8, 'negative_active': 0.5},
            'D2 공감 — 친밀 → 호르몬 공명',
            'social')
        self.tu.register_tool(
            'comfort',
            lambda target, suffering: self.comfort(target, suffering),
            {'oxytocin_high': 1.0, 'cortisol_low': 0.3},
            'D2 위로 행동 (Batson)',
            'social')

        # 분석 (analytic)
        self.tu.register_tool(
            'describe_concept', lambda cat: self.describe_concept(cat),
            {'has_focus': 1.0, 'cortisol_low': 0.3},
            'C2 카테고리 종합 묘사',
            'analytic')
        self.tu.register_tool(
            'find_analogy', lambda a, b, c: self.find_analogy(a, b, c),
            {'dopamine_high': 0.7, 'has_focus': 0.5},
            'C4 A:B::C:? 유추',
            'analytic')

        # 메타 (meta)
        self.tu.register_tool(
            'why_thought', lambda cat: self.why_thought(cat),
            {'doubt_high': 1.0, 'has_focus': 0.5},
            'B6 사고 원인 추적',
            'meta')
        self.tu.register_tool(
            'meta_state', lambda: self.meta_state(),
            {'doubt_high': 0.5, 'cortisol_high': 0.5},
            'B6 통합 메타 상태',
            'meta')

    # ============= 외부 입력 =============
    def perceive(self, category: str, strength: float = 0.7,
                 source: str = 'external'):
        self.sa.activate(category, strength)
        self.wm.add(category, strength, source=source)
        self.hs.category_to_hormone({category}, strength=0.1)
        self.dmn.notify_external_input()

    def learn(self, a: str, b: str, strength: float = 0.4):
        self.sa.learn_pair(a, b, strength)

    def learn_pair(self, a: str, b: str, strength: float = 0.4):
        self.sa.learn_pair(a, b, strength)

    def learn_chain(self, chain: List[str], strength: float = 0.4):
        self.sa.learn_chain(chain, strength)

    def seed_identity(self, *categories: str, strength: float = 0.95,
                     user_agent: Optional[str] = None):
        """핵심 정체성 카테고리 시드 + 자동 protect (decay floor 0.5).

        Args:
            categories: protect할 카테고리들 ('나', 'EVE', '민석' 등)
            strength: 초기 stability (default 0.95)
            user_agent: say() 시 agent 자동 보충용 사용자 이름 (예: '민석')

        Example:
            eve.seed_identity('나', 'EVE', '민석', user_agent='민석')
            # 이후 eve.say("학교 갔다") → '민석'이 agent로 자동 등록
        """
        self.ns.protect(*categories)
        for cat in categories:
            self.ns.identity_stability[cat] = max(
                self.ns.identity_stability.get(cat, 0.0), strength)
            self.sa.activate(cat, 0.3)
        if user_agent:
            self.default_user_agent = user_agent

    # ============= 자연어 대화 (B3 통합) =============
    def say(self, text: str, default_agent: Optional[str] = None) -> Dict[str, Any]:
        """
        EVE에게 자연어 입력 → 이해 → 의심 평가 → 응답.

        Args:
            text: 사용자 입력
            default_agent: agent 명시 안 됐을 때 자동 보충 (예: '민석').
                          self.default_user_agent 있으면 그것 사용.

        B3 통합:
        - intent에 따라 evaluate_command / evaluate_claim 호출
        - action(accept/question/defer/reject) → 의심/거절 카테고리 SA 활성
        - NL.respond가 그 활성 상태에서 자연스러운 어조로 응답 (창발)

        Returns:
            기존 + {'doubt': SD 평가 결과}
        """
        # default_agent: 인자 우선, 없으면 인스턴스 속성
        if default_agent is None:
            default_agent = getattr(self, 'default_user_agent', None)

        # 1) 이해
        u = self.nl.understand(text)

        # 2) 모르는 단어 자기 발견
        added = self.nl.discover_unknown(text)

        # ★ C4 자동 유추: 새 카테고리 발견 시, 비슷한 구조의 익숙한 카테고리 찾기
        analogies_found = []
        if added:
            for new_cat in list(added)[:3]:  # 최대 3개
                try:
                    # SA에서 가장 활성된 카테고리와 새 카테고리 비교
                    top_active = [c for c, _ in self.sa.get_top_active(5)
                                 if c != new_cat and c in self.sa.neighbors]
                    if len(top_active) >= 2:
                        # find_analogy(a, b, c): a:b :: c:?
                        # SA 구조 유사도로 비슷한 거 찾기
                        sim = self.an.structural_similarity(new_cat, top_active[0])
                        if sim > 0.2:
                            analogies_found.append({
                                'new': new_cat,
                                'similar_to': top_active[0],
                                'similarity': sim,
                            })
                except Exception:
                    pass

        # 3) 카테고리 활성 + Hebbian
        cats_list = list(u['categories'])
        for c in cats_list:
            self.sa.activate(c, 0.5)
            self.wm.add(c, 0.5, source='dialogue')
        for i, a in enumerate(cats_list):
            for b in cats_list[i+1:]:
                self.sa.learn_pair(a, b, 0.15)

        # ★ D4: external stream에도 push (높은 salience로 우선순위 ↑)
        for c in cats_list:
            self.ms.push_to_stream('external', c, 0.85)

        self.hs.category_to_hormone(u['categories'], strength=0.05)
        self.dmn.notify_external_input()

        # ★ v35 E8: Frame 자동 추출 → HG 자동 등록 (claim/statement인 경우만)
        # 명령/질문이 아닌 경우 사실 진술로 가정
        intent_for_frame = u.get('intent', '')
        frame_edge_id = None
        norm_id = None  # ★ F1
        # ★ v35 F1: 명령/요청은 norm 관찰 (사용자 명령 내재화)
        if intent_for_frame in ('command', 'request'):
            try:
                norm_id = self.ni.observe_command(
                    text, parsed=u,
                    agent=default_agent or self.ni.default_agent,
                )
            except Exception:
                pass

        if intent_for_frame not in ('command', 'request', 'question'):
            try:
                if default_agent:
                    # agent 자동 보충 모드
                    frames = self.fs.extract_frames(text)
                    edge_ids = []
                    for f in frames:
                        roles = f.get('roles', {})
                        # agent role 없으면 default_agent로 보충
                        if 'agent' not in roles.values():
                            f.setdefault('nodes', set()).add(default_agent)
                            roles[default_agent] = 'agent'
                            f['roles'] = roles
                        eid = self.fs.to_hyperedge(f, self.hg, weight=0.5,
                                                   epistemic='belief',
                                                   source='dialogue')
                        if eid:
                            edge_ids.append(eid)
                    frame_edge_id = edge_ids
                else:
                    frame_edge_id = self.fs.text_to_hg(
                        text,
                        weight=0.5,
                        epistemic='belief',  # 사용자 발화는 일단 belief
                        source='dialogue',
                    )
            except Exception:
                pass  # frame 추출 실패해도 대화 흐름 안 막음

        # 4) 한 tick 진행 (응답 전 EVE가 생각하게)
        self.tick(dt=0.5)

        # ★ B3: 의심 평가 (intent에 따라 분기)
        intent = u.get('intent', '')
        is_command = intent in ('command', 'request', 'ask')
        if is_command:
            doubt_result = self.sd.evaluate_command(text, parsed=u)
        else:
            doubt_result = self.sd.evaluate_claim(text, parsed=u)

        # ★ B3: action → 카테고리 broadcast (창발 어조)
        action = doubt_result['recommended_action']
        for cat, strength in self.DOUBT_ACTION_BROADCAST.get(action, []):
            self.sa.activate(cat, strength)
            self.wm.add(cat, strength * 0.6, source='self_doubt')

        # ★ D2 자동 공감: negative sentiment → resonate (suffering 카테고리 안 매치돼도 default '아픔')
        suffering_resonance = None
        sentiment = u.get('sentiment', '')
        if sentiment == 'negative':
            try:
                # text의 카테고리 中 suffering 후보 찾기
                suffering_cat = None
                for cat in u.get('categories', set()):
                    if self.suf.is_suffering_category(cat):
                        suffering_cat = cat
                        break
                # 매치 X면 default '슬픔' (텍스트가 negative면 일단 공감)
                if suffering_cat is None:
                    suffering_cat = '슬픔'
                suffering_resonance = self.suf.resonate(
                    target=default_agent or '민석',
                    suffering_category=suffering_cat,
                )
            except Exception:
                pass

        # ★ C2 자동 WorldModel: question intent + 카테고리 → describe_concept
        world_describe = None
        if intent == 'question':
            try:
                cats = list(u.get('categories', set()))
                if cats:
                    world_describe = self.wm_world.describe_concept(cats[0])
            except Exception:
                pass

        # 5) 응답 창발 (의심/거절 활성 상태에서 NL이 응답 → 어조 자연스럽게 변함)
        response = self.nl.respond_extended(u)

        # 6) feeling
        feeling = self.nl.feeling_to_text()

        # 7) inner_voice
        voice = self.nl.inner_voice(self.dmn)

        result = {
            'understanding': u,
            'response': response,
            'feeling': feeling,
            'inner_voice': voice,
            'discovered': added,
            'doubt': doubt_result,  # ★ NEW
            'frame_edges': frame_edge_id or [],  # ★ E8: 자동 등록된 사건
            'norm_id': norm_id,  # ★ F1: 등록된 norm (명령일 때)
            'suffering_resonance': suffering_resonance,  # ★ D2: 자동 공감
            'world_describe': world_describe,  # ★ C2: 질문에 대한 개념 설명
            'analogies_found': analogies_found,  # ★ C4: 자동 유추
            'time': self.time,
        }

        self.dialogue_history.append({
            'time': self.time,
            'input': text,
            'response': response,
            'feeling': feeling,
            'doubt_action': action,
            'doubt_level': doubt_result['doubt_level'],
        })

        if self.verbose:
            print(f"  [{self.time:.1f}s] 사용자: {text}")
            print(f"  [{self.time:.1f}s] EVE: {response} (feel: {feeling})")
            print(f"  [{self.time:.1f}s] (의심: {doubt_result['doubt_level']:.2f} "
                  f"→ {action}, sources={doubt_result['doubt_sources']})")
            if voice:
                print(f"  [{self.time:.1f}s] (속마음: {voice})")

        return result

    def learn_beliefs(self, path: str = None,
                     beliefs_dict: Dict = None,
                     max_beliefs: Optional[int] = None) -> Dict[str, int]:
        """
        beliefs.json 로드.
        is_innate=True 신념의 카테고리들은 자동 보호.
        """
        result = self.nl.load_beliefs(path=path, beliefs_dict=beliefs_dict,
                                     max_beliefs=max_beliefs)

        # 선천적 신념 카테고리들 → 보호 list
        for belief in self.nl.beliefs.values():
            if not belief.get('is_innate', False):
                continue
            triple = belief.get('triple') or {}
            for text in [triple.get('subject', ''),
                        triple.get('predicate_text', '')]:
                if text:
                    cats = self.nl.parse(text)
                    self.protected_categories |= cats

        result['protected_categories'] = len(self.protected_categories)
        return result

    def learn_text(self, text: str, link_strength: float = 0.4):
        """자연어 문장으로 학습."""
        cats = self.nl.parse(text)
        if len(cats) < 2:
            return
        self.nl.discover_unknown(text, link_strength=link_strength)
        cat_list = list(cats)
        for i, a in enumerate(cat_list):
            for b in cat_list[i+1:]:
                self.sa.learn_pair(a, b, link_strength)

    # ============= v35 E1 외부 인터페이스 (CorpusLearner) =============
    def learn_sentence(self, sentence: str) -> Dict[str, Any]:
        """E1: 한 문장 학습 (단락/문장 분리, 호르몬 학습 칵테일)"""
        return self.cl.learn_sentence(sentence)

    def learn_paragraph(self, paragraph: str) -> Dict[str, Any]:
        """E1: 한 단락 학습 (문장별 강한 + 인접 문장 약한 연결)"""
        return self.cl.learn_paragraph(paragraph)

    def learn_document(self, document: str) -> Dict[str, Any]:
        """E1: 한 문서 학습 (단락 분리 + 누적)"""
        return self.cl.learn_document(document)

    def learn_corpus(self, texts: List[str], progress_every: int = 100) -> Dict[str, Any]:
        """E1: 대규모 코퍼스 학습 (카테고리 폭증)"""
        return self.cl.learn_corpus(texts, progress_every=progress_every)

    def discover_groups(self, min_cooc: int = 3, min_strength: float = 0.2,
                       max_groups: int = 50) -> Dict[str, Set[str]]:
        """E1: Cooccurrence 기반 자동 그룹 발견"""
        return self.cl.discover_groups(min_cooc=min_cooc,
                                      min_strength=min_strength,
                                      max_groups=max_groups)

    def corpus_stats(self) -> Dict[str, Any]:
        """E1: 코퍼스 학습 통계"""
        return self.cl.stats()

    # ============= v35 E3 외부 인터페이스 (APCLearner) =============
    def predict_next(self) -> Set[str]:
        """E3: 현재 활성 → 다음 활성 카테고리 예측 (외부 검사용)"""
        current = self.sa.get_active()
        focus = self.wm.get_focus()
        return self.apc.predict_next_state(current, focus)

    def apc_stats(self) -> Dict[str, Any]:
        """E3: APC 학습 통계 (정확도/누적)"""
        return self.apc.stats()

    def apc_reset(self):
        """E3: APC 통계 리셋 (가중치는 유지)"""
        self.apc.reset()

    # ============= v35 E4 외부 인터페이스 (DeepReasoning) =============
    def find_path(self, start: str, target: str,
                  mode: str = 'hybrid', max_depth: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """E4: 시작 → 목표 단일 최강 경로 (멀티홉)"""
        return self.dr.find_path(start, target, mode=mode, max_depth=max_depth)

    def find_paths_top_k(self, start: str, target: str,
                        k: int = 3, mode: str = 'hybrid',
                        max_depth: Optional[int] = None) -> List[Dict[str, Any]]:
        """E4: 시작 → 목표 상위 k 경로"""
        return self.dr.find_paths_top_k(start, target, k=k, mode=mode, max_depth=max_depth)

    def why_chain(self, target: str, depth: int = 3,
                 branch: int = 3, max_depth: Optional[int] = None) -> List[List[Tuple[str, float]]]:
        """E4: '왜 X?' — CG 인과 역방향 BFS

        Args:
            target: 결과 카테고리
            depth: 최대 깊이 (기본 3)
            branch: 분기 인자
            max_depth: depth alias (호환성)
        """
        if max_depth is not None:
            depth = max_depth
        return self.dr.why_chain(target, depth=depth, branch=branch)

    def predict_chain(self, start: str, depth: int = 3,
                     branch: int = 3, max_depth: Optional[int] = None) -> List[List[Tuple[str, float]]]:
        """E4: '그러면?' — CG 인과 정방향 BFS"""
        if max_depth is not None:
            depth = max_depth
        return self.dr.predict_chain(start, depth=depth, branch=branch)

    def multi_hop_reason(self, start: str, target: str,
                        max_depth: int = 5) -> Dict[str, Any]:
        """E4: 통합 추론 (path + why + predict)"""
        return self.dr.multi_hop_reason(start, target, max_depth=max_depth)

    def explain_path(self, path: List[str],
                    edges: Optional[List[Tuple]] = None) -> str:
        """E4: 경로 → 자연어 설명"""
        return self.dr.explain_path(path, edges)

    def reasoning_stats(self) -> Dict[str, Any]:
        """E4: 추론 통계"""
        return self.dr.stats()

    # ============= v35 E5 외부 인터페이스 (VSABinding) =============
    def bind(self, role: str, filler: str,
             register_in_sa: bool = True) -> Dict[str, Any]:
        """E5: role + filler → 합성 카테고리"""
        return self.vsa.bind(role, filler, register_in_sa=register_in_sa)

    def unbind(self, bound, role: str, top_k: int = 3) -> Dict[str, Any]:
        """E5: bound + role → filler 복원"""
        return self.vsa.unbind(bound, role, top_k=top_k)

    def compose(self, role_filler: Dict[str, str],
                name: Optional[str] = None,
                register_in_sa: bool = True) -> Dict[str, Any]:
        """E5: 다중 role-filler superposition (복합 의미)"""
        return self.vsa.compose(role_filler, name=name, register_in_sa=register_in_sa)

    def vsa_similarity(self, a, b) -> float:
        """E5: 두 카테고리/vector cosine 유사도"""
        return self.vsa.similarity(a, b)

    def nearest_category(self, target, k: int = 3,
                        exclude: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """E5: target과 가장 비슷한 카테고리 top-k"""
        return self.vsa.nearest_category(target, k=k, exclude=exclude)

    def vsa_stats(self) -> Dict[str, Any]:
        """E5: VSA 통계"""
        return self.vsa.stats()

    # ============= v35 E6 외부 인터페이스 (ContinualRehearsal) =============
    def mark_protected(self, category: str):
        """E6: 카테고리 수동 보호 등록"""
        self.cr_rehearsal.mark_protected(category)

    def rehearse_now(self, strength: Optional[float] = None,
                    max_count: Optional[int] = None) -> List[str]:
        """E6: 보호 카테고리 즉시 재활성"""
        return self.cr_rehearsal.rehearse_now(strength=strength, max_count=max_count)

    def detect_drift(self, category: str,
                    threshold: float = 0.2) -> Optional[Dict[str, Any]]:
        """E6: 카테고리 가중치 변화 감지"""
        return self.cr_rehearsal.detect_drift(category, threshold=threshold)

    def rehearsal_stats(self) -> Dict[str, Any]:
        """E6: rehearsal 통계"""
        return self.cr_rehearsal.stats()

    # ============= v35 E2 외부 인터페이스 (HyperGraph) =============
    def add_event(self, nodes: Set[str],
                  roles: Optional[Dict[str, str]] = None,
                  weight: float = 0.5,
                  temporal: Optional[Dict] = None,
                  epistemic: str = 'belief',
                  source: str = 'manual') -> str:
        """E2: 사건/n-ary 관계 등록 → edge_id"""
        return self.hg.add_edge(nodes, roles=roles, weight=weight,
                               temporal=temporal, epistemic=epistemic,
                               source=source)

    def find_events_with(self, *cats, require_all: bool = True) -> List:
        """E2: 카테고리 포함 사건 검색"""
        return self.hg.find_edges_with(*cats, require_all=require_all)

    def find_events_by_role(self, role: str,
                           value: Optional[str] = None) -> List:
        """E2: role 기반 사건 검색"""
        return self.hg.find_edges_by_role(role, value=value)

    def query_events(self, role_filler: Dict[str, str],
                    top_k: int = 5) -> List:
        """E2: role-filler 부분 매칭"""
        return self.hg.query(role_filler, top_k=top_k)

    def hypergraph_neighbors(self, node: str,
                            via_roles: Optional[Set[str]] = None) -> Dict[str, float]:
        """E2: hyperedge 통한 이웃"""
        return self.hg.neighbors_via_edges(node, via_roles=via_roles)

    def expand_hyperedges_to_sa(self, min_weight: float = 0.3,
                               link_strength: float = 0.2) -> int:
        """E2: hyperedges → SA 페어로 등록 (호환성)"""
        return self.hg.expand_to_pairs(min_weight=min_weight,
                                      link_strength=link_strength)

    def hypergraph_stats(self) -> Dict[str, Any]:
        """E2: HyperGraph 통계"""
        return self.hg.stats()

    # ============= v35 E7 외부 인터페이스 (CreativeAdvanced) =============
    def is_creative_mode(self) -> bool:
        """E7: 호르몬 게이트 (DA↑ + Cort↓)"""
        return self.ca.is_creative_mode()

    def compose_novel(self, seeds: List[str], n: int = 5,
                     min_novelty: float = 0.3) -> List[Dict[str, Any]]:
        """E7: VSA 기반 새 합성 N개"""
        return self.ca.compose_novel(seeds, n=n, min_novelty=min_novelty)

    def imagine_event(self, seeds: List[str],
                     event_type: str = 'imagined') -> Optional[Dict[str, Any]]:
        """E7: HG 가상 사건 (epistemic=hypothesis)"""
        return self.ca.imagine_event(seeds, event_type=event_type)

    def find_insight_bridge(self, a: str, b: str,
                           k: int = 3) -> List[Dict[str, Any]]:
        """E7: DR 기반 통찰 다리"""
        return self.ca.find_insight_bridge(a, b, k=k)

    def divergent_converge(self, seed: str,
                          n_diverge: int = 5,
                          n_converge: int = 2) -> List[Dict[str, Any]]:
        """E7: 발산-수렴"""
        return self.ca.divergent_converge(seed, n_diverge=n_diverge,
                                         n_converge=n_converge)

    def creative_cycle(self, seed: str,
                      auto_incubate: bool = True) -> Dict[str, Any]:
        """E7: Wallas 4단계 자동 사이클"""
        return self.ca.creative_cycle(seed, auto_incubate=auto_incubate)

    def creative_advanced_stats(self) -> Dict[str, Any]:
        """E7: 통계"""
        return self.ca.stats()

    # ============= v35 E8 외부 인터페이스 (FrameSemantics) =============
    def parse_frame(self, text: str) -> Optional[Dict[str, Any]]:
        """E8: 한 문장 → frame (roles 자동 추출)"""
        return self.fs.parse_sentence(text)

    def extract_frames(self, paragraph: str) -> List[Dict[str, Any]]:
        """E8: 문단 → 여러 frame"""
        return self.fs.extract_frames(paragraph)

    def text_to_events(self, text: str,
                      epistemic: str = 'belief',
                      source: str = 'frame_extracted',
                      weight: float = 0.6) -> List[str]:
        """E8: 자연어 텍스트 → HG 자동 등록 (frame 기반)"""
        return self.fs.text_to_hg(text, weight=weight,
                                 epistemic=epistemic, source=source)

    def parse_question(self, question: str) -> Optional[Dict[str, Any]]:
        """E8: WH 질문 → query frame (unknowns 표시)"""
        return self.fs.compose_question(question)

    def frame_stats(self) -> Dict[str, Any]:
        """E8: FrameSemantics 통계"""
        return self.fs.stats()

    # ============= v35 E9 외부 인터페이스 (SemanticDistance) =============
    def semantic_distance(self, a: str, b: str) -> float:
        """E9: 카테고리 간 의미 거리 (0=같음, 1=다름)"""
        return self.sd_dist.distance(a, b)

    def semantic_similarity(self, a: str, b: str) -> float:
        """E9: 카테고리 간 의미 유사도 (1 - distance)"""
        return self.sd_dist.similarity(a, b)

    def find_similar_categories(self, target: str, k: int = 5,
                               exclude: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """E9: target과 가장 비슷한 카테고리 top-k"""
        return self.sd_dist.find_similar(target, k=k, exclude=exclude)

    def find_dissimilar_categories(self, target: str, k: int = 5,
                                  exclude: Optional[Set[str]] = None) -> List[Tuple[str, float]]:
        """E9: target과 가장 먼 카테고리 top-k"""
        return self.sd_dist.find_dissimilar(target, k=k, exclude=exclude)

    def cluster_categories(self, threshold: float = 0.5,
                          candidates: Optional[Set[str]] = None) -> List[Set[str]]:
        """E9: 의미 유사도 기반 클러스터링 (도메인 자동 발견)"""
        return self.sd_dist.cluster_by_similarity(threshold=threshold,
                                                 candidates=candidates)

    def antonym_candidates(self, target: str, k: int = 3,
                          require_different_group: bool = False) -> List[Tuple[str, float]]:
        """E9: target과 의미적 반대 후보들"""
        return self.sd_dist.antonym_candidates(target, k=k,
                                              require_different_group=require_different_group)

    def evaluate_path_coherence(self, path: List[str]) -> Dict[str, Any]:
        """E9: DR path의 의미 일관성 평가"""
        return self.sd_dist.evaluate_path(path)

    def semantic_distance_stats(self) -> Dict[str, Any]:
        """E9: 통계"""
        return self.sd_dist.stats()

    # ============= v35 F1 외부 인터페이스 (NormInternalization) =============
    def observe_command(self, text: str, parsed: Optional[Dict] = None,
                        agent: Optional[str] = None) -> Optional[str]:
        """F1: 명령 발화 관찰 → norm 등록 (say()가 자동 호출하지만 수동도 가능)"""
        return self.ni.observe_command(text, parsed=parsed, agent=agent)

    def get_user_norms(self, agent: Optional[str] = None,
                       threshold: float = 0.5) -> List:
        """F1: 사용자가 자주 시키는 것 (내재화된 norms)"""
        return self.ni.get_active_norms(agent=agent, threshold=threshold)

    def introspect_norms(self, agent: Optional[str] = None) -> List[Dict[str, Any]]:
        """F1: '민석이 자주 시키는 것'"""
        return self.ni.introspect_norms(agent=agent)

    def reject_norm_action(self, action: str, agent: Optional[str] = None):
        """F1: 사용자 명령 반례 (예: '쉬지마' → 휴식 norm 약화)"""
        self.ni.reject_action(action, agent=agent)

    def detect_norm_conflict(self, new_action: str,
                            agent: Optional[str] = None) -> List:
        """F1: 새 명령이 기존 norm과 충돌하는지"""
        return self.ni.detect_conflict(new_action, agent=agent)

    # ============= v35 F2 외부 인터페이스 (AllostaticLearning) =============
    def get_learned_cocktail(self, category: str) -> Optional[Dict[str, float]]:
        """F2: 카테고리에 학습된 호르몬 칵테일 (있으면)"""
        return self.al.get_learned_cocktail(category)

    def apply_learned_cocktail(self, category: str, scale: float = 0.3):
        """F2: 학습된 칵테일을 호르몬에 수동 적용"""
        return self.al.apply_learned(category, scale=scale)

    def introspect_hormone_patterns(self) -> List[Dict[str, Any]]:
        """F2: EVE가 학습한 호르몬 패턴들 ('이 상황엔 이 호르몬 패턴')"""
        return self.al.introspect_patterns()

    def inner_voice(self) -> Optional[str]:
        return self.nl.inner_voice(self.dmn)

    def feeling(self) -> str:
        return self.nl.feeling_to_text()

    # ============= B4 외부 학습 트리거 (SD + AI 짝꿍) =============
    def update_from_outcome(self, actual_outcome: str) -> Optional[Dict[str, Any]]:
        """
        외부 신호로 마지막 의심 평가의 outcome 보고.
        SD baseline + AI 호르몬-outcome 매핑 + Agency 마지막 행동 outcome 기록.

        Args:
            actual_outcome: 'good' / 'bad' / 'neutral'

        Returns:
            {'sd': sd_result, 'ai': ai_result} 또는 None (last_doubt 없을 때)
        """
        if not self.sd.last_doubt:
            return None
        predicted_doubt = self.sd.last_doubt.get('doubt_level', 0.0)
        sd_result = self.sd.update_from_outcome(predicted_doubt, actual_outcome)
        ai_result = self.ai.update_from_outcome(predicted_doubt, actual_outcome)

        # B5: 마지막 행동 record가 있으면 outcome 채워주기
        if self.ag.actions:
            self.ag.actions[-1]['outcome'] = actual_outcome

        return {'sd': sd_result, 'ai': ai_result, 'outcome': actual_outcome}

    # ============= B5 외부 인터페이스 (Agency) =============
    def decide_action(self, action_category: str,
                     source: str = 'unknown',
                     auto_record: bool = True) -> Dict[str, Any]:
        """
        행동 결정 인터페이스. 내부 6 시그널(gut/doubt/AI/intent/threat/focus)을
        Agency가 통합해서 do/hesitate/skip 결정.

        Args:
            action_category: 평가 대상 행동 (예: '대화', '쉬다')
            source: 'external' / 'spontaneous' / 'self_intent' / 'unknown'
            auto_record: True면 결정 후 history에 자동 기록

        Returns:
            evaluate_action 결과 dict (choice, confidence, attribution, ...)
        """
        result = self.ag.evaluate_action(action_category, source=source)
        if auto_record:
            self.ag.record_action(
                action_category=result['action'],
                choice=result['choice'],
                attribution=result['attribution'],
                confidence=result['confidence'],
            )
        return result

    # ============= B6 외부 인터페이스 (MetaCognition) =============
    def why_thought(self, category: str) -> Dict[str, Any]:
        """
        "왜 X 카테고리가 떠올랐지?" 메타 분석.
        MetaCognition.monitor_thought 위임.

        Returns: {'cause', 'evidence', 'confidence', 'all_causes', ...}
        """
        return self.mc.monitor_thought(category)

    def endorse_need(self,
                    dominant_need: Optional[Tuple[str, float]] = None
                    ) -> Optional[Dict[str, Any]]:
        """
        Frankfurt 2차 욕구: 1차 욕구를 endorse / override 결정.
        MetaCognition.endorse_or_override 위임.

        Args:
            dominant_need: None이면 DS에서 자동

        Returns: {'need', 'strength', 'decision', 'score', 'reasons'}
                 또는 None
        """
        return self.mc.endorse_or_override(dominant_need)

    def meta_state(self) -> Dict[str, Any]:
        """
        "지금 내 마음 상태" — 통합 메타 관찰.
        MetaCognition.meta_introspect 위임.

        Returns: focus + cause + need + decision + confidence + recent_thoughts
        """
        return self.mc.meta_introspect()

    # ============= C1 외부 인터페이스 (CausalGraph) =============
    def why_caused(self, category: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        "X는 무엇 때문에?" — 인과 그래프에서 X의 직접 원인들 (정렬).
        why_thought (메타 인지)와 다름:
        - why_thought: '왜 떠올랐지?' (메타 — 호르몬/spread/dmn 등)
        - why_caused : '무엇이 X를 만들었지?' (인과 — 학습된 cause)

        Args:
            top_n: 최대 반환 부모 수

        Returns: [(cause, weight), ...] 가중치 ↓ 정렬
        """
        return self.cg.why(category, top_n=top_n)

    def what_if(self, category: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        "X 하면 어떻게 되지?" — 인과 그래프에서 X의 결과들.

        Returns: [(effect, weight), ...] 가중치 ↓ 정렬
        """
        return self.cg.what_if(category, top_n=top_n)

    def is_cause_of(self, a: str, b: str) -> bool:
        """A → ... → B 인과 경로 존재? (transitive)"""
        return self.cg.is_cause_of(a, b)

    def imagine_doing(self, category: str,
                     strength: float = 0.7,
                     depth: int = 2) -> Dict[str, Any]:
        """
        do(category) — 강제 활성 + 자식들 인과 활성.
        Pearl intervention의 단순 버전.

        주의: 이건 SA를 *변경*하는 동작 (다른 외부 메서드와 다름).
        '머릿속에서 X를 한다고 가정하고' 그 결과를 봄.

        Args:
            category: 강제 활성할 카테고리
            strength: 활성 강도
            depth: 자식 인과 사슬 깊이

        Returns: {'activated': [(cat, strength, depth), ...], 'depth_reached'}
        """
        return self.cg.intervene(category, strength=strength, depth=depth)

    # ============= C3 외부 인터페이스 (Counterfactual) =============
    def what_if_not(self, category: str, depth: int = 3) -> Dict[str, Any]:
        """
        "X가 일어나지 않았다면?" 가상 시뮬.
        SA/CG는 변경 X (read-only).

        Returns: {'absent', 'removed', 'weakened', 'unaffected'}
        """
        return self.cf.what_if_not(category, depth=depth)

    def what_if_instead(self, category: str, alternative: str,
                       depth: int = 3) -> Dict[str, Any]:
        """
        "X 대신 alternative였다면?"

        Returns: {'replaced', 'with', 'gained', 'lost', 'shared'}
        """
        return self.cf.what_if_instead(category, alternative, depth=depth)

    def regret_or_relief(self,
                        action_record: Optional[Dict] = None,
                        alternative_action: Optional[str] = None
                        ) -> Optional[Dict[str, Any]]:
        """
        실제 행동의 outcome을 가상 대안과 비교 → regret/relief 신호.

        부수효과: regret → cortisol↑, relief → serotonin/dopamine↑

        Args:
            action_record: None이면 Agency.actions 마지막
            alternative_action: None이면 closest_world top1

        Returns:
            {'signal': 'regret'/'relief'/'neutral', 'magnitude', ...}
            또는 None
        """
        return self.cf.regret_or_relief(action_record, alternative_action)

    def closest_world(self, category: str, top_n: int = 3
                     ) -> List[Tuple[str, float]]:
        """
        Lewis "가장 가까운 가능 세계" — category와 비슷한 다른 카테고리.

        Returns: [(alternative, similarity), ...]
        """
        return self.cf.closest_world(category, top_n=top_n)

    # ============= C5 외부 인터페이스 (Temporal) =============
    def recall_past(self, category: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        카테고리 관련 과거 일화 시간 역순 (최근부터).
        Returns: [{'time', 'categories', 'summary'}, ...]
        """
        return self.tm.recall_past(category, n=n)

    def recent_episodes(self, window: float = 60.0,
                       max_n: int = 20) -> List[Dict[str, Any]]:
        """최근 N초 일화들"""
        return self.tm.recent_episodes(window=window, max_n=max_n)

    def imagine_future(self, seed: str,
                      depth: int = 3,
                      top_n_per_step: int = 3) -> Dict[str, Any]:
        """
        "X 일어났으니 다음엔?" 미래 인과 시뮬 (CG 따라).
        Returns: {'seed', 'timeline': [{'depth', 'category', 'weight'}], 'depth_reached'}
        """
        return self.tm.imagine_future(seed, depth=depth,
                                     top_n_per_step=top_n_per_step)

    def time_distance(self, category: str) -> Optional[float]:
        """카테고리 마지막 발생 후 경과 (초). None=한 번도 X"""
        return self.tm.time_distance(category)

    def precedes(self, a: str, b: str) -> Optional[bool]:
        """A가 B보다 시간상 먼저? None=둘 중 하나 발생 X"""
        return self.tm.precedes(a, b)

    def perceived_time_factor(self) -> float:
        """주관적 시간 배수 (cort↑→길게, da↑→짧게)"""
        return self.tm.perceived_time_factor()

    # ============= C7 외부 인터페이스 (EmotionRegulation) =============
    def detect_regulation_need(self) -> Dict[str, Any]:
        """조절 필요 감지 — cort↑/regret/tension"""
        return self.er.detect_need()

    def regulate(self,
                strategy: Optional[str] = None,
                auto: bool = False) -> Optional[Dict[str, Any]]:
        """
        감정 조절 실행.

        Args:
            strategy: 'distraction'/'reappraisal'/'suppression'/None(추천)
            auto: True면 detect_need 통과 시에만

        Returns: {'strategy', 'cort_delta', ...} 또는 None
        """
        return self.er.regulate(strategy=strategy, auto=auto)

    def reappraise(self, situation: Optional[str] = None
                  ) -> Dict[str, Any]:
        """
        인지 재평가 (Ochsner — 가장 효과적).
        situation=None이면 CF 최근 regret에서 자동.
        """
        return self.er.reappraisal(situation_category=situation)

    def distract(self, target: Optional[str] = None) -> Dict[str, Any]:
        """주의 전환 — 긍정 카테고리로"""
        return self.er.distraction(target_category=target)

    def suppress_emotion(self) -> Dict[str, Any]:
        """반응 억제 — 호르몬 직접 (비용: NE↑)"""
        return self.er.suppression()

    # ============= C4 외부 인터페이스 (Analogy) =============
    def find_analogy(self, a: str, b: str, c: str,
                    top_n: int = 1) -> List[Tuple[str, float]]:
        """
        "A:B::C:?" 유추.
        Returns: [(d, score), ...] top_n
        """
        return self.an.find_analogy(a, b, c, top_n=top_n)

    def explain_with_analogy(self, target: str, top_k: int = 3,
                            min_similarity: float = 0.3
                            ) -> List[Dict[str, Any]]:
        """
        "X와 비슷한 게 뭐지?" — 인과 구조 비슷한 카테고리 찾기.
        """
        return self.an.explain_with_analogy(
            target, top_k=top_k, min_similarity=min_similarity)

    def structural_similarity(self, x: str, y: str) -> float:
        """두 카테고리의 인과 구조 유사도 [0, 1]"""
        return self.an.structural_similarity(x, y)

    def map_relation(self, source_pair: Tuple[str, str],
                    target_anchor: str) -> Optional[Dict[str, Any]]:
        """source 관계 → target 매핑"""
        return self.an.map_relation(source_pair, target_anchor)

    # ============= C2 외부 인터페이스 (WorldModel) =============
    def describe_concept(self, category: str) -> Dict[str, Any]:
        """
        카테고리의 모든 측면 통합 묘사.
        Returns: {group, parents, children, episodes, similar, last_seen, ...}
        """
        return self.wm_world.describe_concept(category)

    def list_domains(self) -> Dict[str, List[str]]:
        """도메인별 카테고리 클러스터"""
        return self.wm_world.list_domains()

    def categorize(self, category: str) -> Dict[str, Any]:
        """카테고리의 도메인 추론 (그룹 미정 시 인접 다수결)"""
        return self.wm_world.categorize(category)

    def find_schema(self, seed: Optional[str] = None,
                   max_chains: int = 5) -> List[Dict[str, Any]]:
        """반복 인과 패턴 추출"""
        return self.wm_world.find_schema(seed=seed, max_chains=max_chains)

    def knowledge_overview(self) -> Dict[str, Any]:
        """EVE의 전체 지식 요약 (도메인/엣지/일화/평균)"""
        return self.wm_world.knowledge_overview()

    def relate(self, a: str, b: str) -> Dict[str, Any]:
        """A와 B의 관계 종합 분석 (group/causal/sa/episodic/analogy)"""
        return self.wm_world.relate(a, b)

    # ============= C6 외부 인터페이스 (GoalManagement) =============
    def set_goal(self, category: str,
                priority: float = 0.5,
                deadline_offset: Optional[float] = None) -> Optional[str]:
        """
        목표 설정.

        Returns: goal_id 또는 None
        """
        return self.gm.goal_set(category, priority=priority,
                               deadline_offset=deadline_offset,
                               source='manual')

    def active_goals(self, top_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """현재 활성 목표들 (priority 정렬)"""
        goals = self.gm.active_goals(top_n=top_n)
        return [{
            'id': g.id,
            'category': g.category,
            'priority': g.priority,
            'progress': g.progress,
            'source': g.source,
            'deadline': g.deadline,
        } for g in goals]

    def suggest_for_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """
        목표 달성 위한 행동 추천 (CG.parents + CF 비교).
        Returns: {'recommended', 'rationale', ...} or None
        """
        return self.gm.suggest_action(goal_id)

    def complete_goal(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """목표 달성 → dopamine + endorphin 자극"""
        return self.gm.complete_goal(goal_id)

    def abandon_goal(self, goal_id: str,
                    reason: str = 'manual') -> Optional[Dict[str, Any]]:
        """목표 포기 → 약한 cortisol 자극 (manual 일 때만)"""
        return self.gm.abandon_goal(goal_id, reason=reason)

    # ============= D1 외부 인터페이스 (Creative) =============
    def find_remote(self, seeds: List[str], k: int = 3
                   ) -> List[Tuple[str, float]]:
        """
        Mednick RAT — 먼 seed 카테고리들 사이 공통 매개.
        예: ['민석', '비'] → '슬픔'

        Returns: [(category, score), ...] 점수 ↓ 정렬
        """
        return self.cr.find_remote_associates(seeds, k=k)

    def blend(self, a: str, b: str) -> Dict[str, Any]:
        """
        Boden — 두 카테고리 의미적 조합.
        Returns: {shared_neighbors, novel_a, novel_b, coherence, novelty, score}
        """
        return self.cr.blend_concepts(a, b)

    def analogy_blend(self, target: str, source: str) -> Dict[str, Any]:
        """AN(C4) 활용 구조 차용 (없으면 blend로 fallback)"""
        return self.cr.analogy_blend(target, source)

    def incubate(self, seed: str) -> bool:
        """
        Wallas — 잠재의식(DMN)에 시드.
        실패 조건: 창의 모드 X (도파민↓ or cort↑) / 이미 incubating
        """
        return self.cr.incubate(seed)

    def harvest(self, seed: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Wallas illumination 수확 — DMN 자발 활성 中 2-hop 의미 다리.
        seed=None이면 모든 만료 seed 수확.
        """
        return self.cr.harvest_incubation(seed)

    def verify_creative(self, idea: str,
                       target_context: Optional[Set[str]] = None
                       ) -> Dict[str, Any]:
        """
        아이디어 평가 (Wallas verification) — Agency/SD 통합 점수.
        Returns: {idea, novelty, coherence, confidence, score}
        """
        return self.cr.verify_idea(idea, target_context=target_context)

    def is_creative_mode(self) -> bool:
        """창의 모드 활성 여부 (도파민↑ + cort↓)"""
        return self.cr.is_creative_mode()

    # ============= D2 외부 인터페이스 (Suffering) =============
    def empathize(self, target: str, suffering_category: str,
                 full_chain: bool = True) -> Dict[str, Any]:
        """
        타자의 고통 인지 → 공명 + self-other 구분 + 공감 행동.
        통합 진입점.

        Args:
            target: 고통 받는 대상
            suffering_category: 고통 카테고리 ('슬픔', '아프다' 등)
            full_chain: True면 distinguish + resonate + compassion 모두

        Returns: {perceived, distinguish, resonate, compassion, ...}
        """
        return self.suf.perceive_suffering(target, suffering_category,
                                          full_chain=full_chain)

    def resonate_with(self, target: str, suffering_category: str
                     ) -> Dict[str, Any]:
        """
        공명만 (호르몬 변조). 자기-타자 구분/행동 X.
        세밀한 제어 시 사용.
        """
        return self.suf.resonate(target, suffering_category)

    def intimacy(self, target: str) -> float:
        """타자에 대한 친밀도 [0, 1] (EM + social + self-related)"""
        return self.suf.compute_intimacy(target)

    def is_suffering(self, category: str) -> bool:
        """카테고리가 고통 카테고리인지"""
        return self.suf.is_suffering_category(category)

    def burnout_state(self) -> Dict[str, Any]:
        """현재 공감 부담 상태 (compassion fatigue)"""
        return self.suf.burnout_check()

    def comfort(self, target: str, suffering_category: str
               ) -> Dict[str, Any]:
        """위로 행동만 추천 (Batson empathy-altruism)"""
        return self.suf.compassion_action(target, suffering_category)

    # ============= D3 외부 인터페이스 (Humor) =============
    def joke(self, seed: str) -> Optional[Dict[str, Any]]:
        """
        EVE가 농담 생성 + laugh.
        seed → punchline 자동 탐색 + 호르몬 보상.

        Returns: {setup, punchline, score, laugh: {...}} 또는 None
        """
        return self.hu.joke(seed)

    def is_funny(self, setup: str, punchline: str) -> Dict[str, Any]:
        """
        주어진 setup-punchline 페어 평가.
        Returns: {incongruity, resolution, safety, novelty, score, is_funny}
        """
        return self.hu.is_funny(setup, punchline)

    def laugh(self, joke_score: float = 0.7) -> Dict[str, Any]:
        """
        외부 농담 듣고 웃기 — 호르몬 보상만.
        Vrticka: DA↑, endo↑, cort↓.
        """
        return self.hu.laugh(joke_score)

    def is_humor_mode(self) -> bool:
        """농담 가능 모드 (DA↑ + cort↓ + 비burnout)"""
        return self.hu.is_humor_safe()

    # ============= D4 외부 인터페이스 (MultiStream) =============
    def push_to_stream(self, stream_name: str, category: str,
                      salience: float = 0.5) -> bool:
        """
        특정 stream에 카테고리 push.
        Streams: external/inner/emotion/planning/recall
        """
        return self.ms.push_to_stream(stream_name, category, salience)

    def stream_focus(self) -> Optional[Dict[str, Any]]:
        """
        한 사이클 통합 → 가장 높은 bid stream의 focus.
        Returns: {stream, category, bid, all_bids}
        """
        return self.ms.integrate(dt=0.5)

    def dump_streams(self) -> Dict[str, List[str]]:
        """각 stream의 현재 큐 (디버그)"""
        return self.ms.dump_streams()

    def stream_focus_history(self, last_n: int = 10
                             ) -> List[Tuple[float, str, str]]:
        """최근 stream focus history"""
        return self.ms.get_focus_history(last_n=last_n)

    # ============= D5 외부 인터페이스 (ToolUse) =============
    def select_tool(self,
                   situation: Optional[Dict[str, float]] = None,
                   category_filter: Optional[str] = None,
                   top_n: int = 3) -> List[Tuple[str, float]]:
        """
        현재 상황 → 도구 추천 (의식적 매칭).
        situation=None이면 자동 (호르몬 + 카테고리 + SD).

        Returns: [(tool_name, score), ...]
        """
        return self.tu.select_tool(situation,
                                  category_filter=category_filter,
                                  top_n=top_n)

    def use_tool(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        """
        도구 명시적 호출 (System 2).
        Returns: {tool, success, result, error, pre_state, post_state}
        """
        return self.tu.invoke_tool(name, *args, **kwargs)

    def plan_and_act(self,
                    situation: Optional[Dict[str, float]] = None,
                    args: Optional[tuple] = None,
                    kwargs: Optional[Dict] = None,
                    category_filter: Optional[str] = None
                    ) -> Optional[Dict[str, Any]]:
        """
        Bratman intention: 상황 → 계획 → 실행 자동.
        select_tool 1위 자동 호출.
        """
        return self.tu.plan_and_act(situation=situation,
                                   args=args, kwargs=kwargs,
                                   category_filter=category_filter)

    def reflect_on_tool(self,
                       tool_name: Optional[str] = None
                       ) -> Optional[Dict[str, Any]]:
        """도구 사용 결과 메타 평가 (호르몬 변화 분석)"""
        return self.tu.reflect_on_use(tool_name)

    def teach_tool(self, tool_name: str, outcome: str = 'helpful'
                  ) -> Dict[str, Any]:
        """
        외부 피드백으로 도구 효과 학습.
        outcome: 'helpful' / 'neutral' / 'unhelpful'
        """
        return self.tu.update_effectiveness(tool_name, outcome)

    def list_tools(self,
                  category: Optional[str] = None
                  ) -> List[Dict[str, Any]]:
        """등록된 도구 목록"""
        return self.tu.list_tools(category=category)

    def tool_history(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """최근 도구 사용 이력"""
        return self.tu.tool_history(last_n=last_n)

    # ============= D6 외부 인터페이스 (NarrativeSelf) =============
    def who_am_i(self) -> Dict[str, Any]:
        """
        EVE 자기 서사 (McAdams + Bruner).
        Returns: {identity_summary, themes, self_defining, coherence,
                 narrative_text, identity}
        """
        return self.ns.narrate()

    def identity_themes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """McAdams themes — 자주 등장하는 카테고리"""
        return self.ns.extract_themes(top_n=top_n)

    def self_defining_memories(self, top_n: int = 5
                              ) -> List[Dict[str, Any]]:
        """Singer 2004 — 자기-defining 일화 (강한 정서 + vivid)"""
        return self.ns.self_defining_memories(top_n=top_n)

    def identity_categories(self,
                           min_stability: Optional[float] = None
                           ) -> List[Dict[str, Any]]:
        """안정된 자기 정체성 카테고리들"""
        return self.ns.identity_categories(min_stability=min_stability)

    def life_coherence(self) -> Dict[str, Any]:
        """Habermas 2000 일관성: temporal/thematic/causal/overall"""
        return self.ns.coherence_score()

    def integrate_life_event(self,
                            category: str,
                            valence: float = 0.0,
                            importance: float = 0.5
                            ) -> Dict[str, Any]:
        """새 사건을 정체성에 통합"""
        return self.ns.integrate_event(category, valence=valence,
                                      importance=importance)

    # ============= 메인 루프 =============
    def tick(self, dt: float = 0.5) -> Dict[str, Any]:
        """EVE 한 tick = 모든 모듈 1 사이클"""
        self.time += dt
        self.tick_count += 1

        result = {'time': self.time, 'tick': self.tick_count}

        # 1. 호르몬
        self.hs.update(dt=dt)

        # 2. 호르몬 → SA 임계 변조
        mods = self.hs.get_category_threshold_modulation()
        self.sa.apply_hormone_modulation(mods)

        # 3-4. spread + decay
        self.sa.spread(steps=1)
        self.sa.decay(dt=dt)

        # 5. SA → 호르몬
        active_cats = self.sa.get_active()
        if active_cats:
            self.hs.category_to_hormone(active_cats, strength=0.05)

        # 6-9. WM + GNW
        self.wm.update_from_activation(self.sa)
        self.wm.apply_hormone_state(self.hs.get_state())
        broadcast_result = self.wm.broadcast()
        if broadcast_result:
            result['broadcast'] = broadcast_result
        self.wm.decay(dt=dt)

        # 10. DMN
        dmn_result = self.dmn.tick(dt=dt)
        if dmn_result:
            result['spontaneous'] = dmn_result
            # ★ B2 정식 통합: DMN이 memory_recall 모드면 일화 떠올림
            # 회상 효과: 일화의 다른 카테고리들도 SA에 활성 → 의식의 흐름 (creative drift)
            if dmn_result.get('mode') == 'memory_recall':
                cat = dmn_result.get('category')
                if cat:
                    reminded = self.em.remind(cat)
                    if reminded and self.verbose:
                        print(f"  [{self.time:5.1f}s] [em.remind     ] "
                              f"{cat} → '{reminded.summary}'")
            # ★ Fix: DMN self_intent → Agency 자동 행동 결정
            # 자기 명령 발생 시 → decide_action으로 의도 기록
            if dmn_result.get('mode') == 'self_intent':
                cat = dmn_result.get('category')
                if cat:
                    try:
                        self.decide_action(cat, source='self_intent',
                                          auto_record=True)
                    except Exception:
                        pass
            if self.verbose:
                voice = self.nl.inner_voice(self.dmn)
                print(f"  [{self.time:5.1f}s] [{dmn_result['mode']:18s}] "
                      f"{dmn_result['category']:15s} → '{voice}'")

        # 11. DigitalSomatic
        self.ds.update(dt=dt)
        result['feeling'] = self.ds.get_feeling()

        # 11.5 ★ B2 정식 통합: EpisodicMemory (auto_encode + consolidate + forget)
        self.em.update(dt=dt)

        # 12. ★ B3: SelfDoubt baseline 시간 회귀
        self.sd.update(dt=dt)

        # 12.5 ★ B4: ActiveInference 시간 누적 (deque 정리 등)
        self.ai.tick(dt=dt)
        # 자동 predict + 다음 cycle observe (Friston FEP)
        if self.tick_count % 20 == 0:
            try:
                self.ai.predict(horizon=1.0)
            except Exception:
                pass
        if self.tick_count % 20 == 10:  # 10 tick 후 관찰
            try:
                self.ai.observe()
            except Exception:
                pass

        # 12.6 ★ B5: Agency 시간 누적
        self.ag.tick(dt=dt)

        # 12.7 ★ B6: MetaCognition 시간 누적 + DMN 자발 활성 시 자동 monitor
        self.mc.tick(dt=dt)
        if dmn_result and dmn_result.get('category'):
            try:
                self.mc.monitor_thought(dmn_result['category'])
            except Exception:
                pass

        # 12.8 ★ C1: CausalGraph 시간 누적 + 주기적 자동 학습
        self.cg.tick(dt=dt)
        if self.tick_count % self._cg_observe_interval == 0:
            self.cg.observe_temporal()

        # 12.8.1 ★ C4: Analogy 시간 누적 + 매 80 tick 자동 유추 시도
        self.an.tick(dt=dt)
        if self.tick_count % 80 == 0:
            try:
                # 현재 활성 top 3 카테고리에서 비슷한 구조 찾기
                top = [c for c, _ in self.sa.get_top_active(5) if c]
                if len(top) >= 2:
                    sim = self.an.structural_similarity(top[0], top[1])
                    if sim is not None:
                        # 새 카테고리 발견 카운트로 활용
                        pass
            except Exception:
                pass

        # 12.9 ★ C3: Counterfactual 시간 누적 + 자동 regret/relief
        self.cf.tick(dt=dt)
        # 부정 sentiment + 강한 cort → 자동 regret 시뮬 (매 50 tick)
        if self.tick_count % 50 == 0:
            cort = self.hs.hormones['cortisol'].level
            if cort > 0.55:
                try:
                    # action_record 자동 (Agency.actions 마지막)
                    cf_result = self.cf.regret_or_relief()
                    if cf_result:
                        result['cf_auto'] = cf_result
                except Exception:
                    pass

        # 12.10 ★ C5: Temporal 시간 누적 (read-only)
        self.tm.tick(dt=dt)

        # 12.11 ★ C7: EmotionRegulation 시간 + 주기적 자동 조절
        self.er.tick(dt=dt)
        if self.tick_count % self._er_auto_interval == 0:
            er_result = self.er.regulate(auto=True)
            if er_result:
                result['er_auto'] = er_result

        # 12.12 ★ C6: GoalManagement (만료/자동평가/자동propose)
        self.gm.tick(dt=dt)

        # 12.13 ★ C2: WorldModel 시간 누적 (read-only 통합 view)
        self.wm_world.tick(dt=dt)

        # 12.14 ★ D1: Creative — incubation 자동 수확 (read-only)
        self.cr.tick(dt=dt)

        # 12.15 ★ D2: Suffering — burnout 자동 체크
        self.suf.tick(dt=dt)

        # 12.16 ★ D3: Humor — 시간 진행 + 자동 농담 시도
        self.hu.tick(dt=dt)
        # DA↑ + 5HT↑ + cort↓ 임계 (좋은 기분) → 매 100 tick 자동 농담
        if self.tick_count % 100 == 0:
            try:
                if self.hu.is_humor_safe():
                    # 현재 활성 카테고리에서 seed 선택
                    focus = self.wm.get_focus()
                    if focus:
                        joke_result = self.hu.joke(seed=focus)
                        if joke_result:
                            result['humor_auto'] = joke_result
            except Exception:
                pass

        # 12.17 ★ D4: MultiStream — 5 stream 평행 처리 + GNW broadcast
        # 매 5 tick마다 통합 (너무 자주 broadcast 방지)
        if self.tick_count % 5 == 0:
            ms_result = self.ms.tick(dt=dt * 5)
            if ms_result:
                result['stream_focus'] = ms_result
        else:
            # 큐 decay만
            self.ms.time += dt
            self.ms.tick_count += 1
            for stream in self.ms.streams.values():
                stream.decay(dt)

        # 12.18 ★ D5: ToolUse — 시간 진행 + 자동 도구 평가
        self.tu.tick(dt=dt)
        # DMN self_intent (자기 명령) 또는 action 모드 → 자동 select_tool
        if dmn_result and dmn_result.get('mode') in ('action', 'self_intent'):
            try:
                selected = self.tu.select_tool()
                if selected:
                    result['tool_auto'] = selected
            except Exception:
                pass
        # 또는 매 60 tick (30초)마다 강한 욕구 시 도구 평가
        elif self.tick_count % 60 == 0:
            try:
                # 강한 need 있을 때만
                dom_need = self.ds.get_dominant_need() if self.ds else None
                if dom_need and dom_need[1] > 0.5:
                    selected = self.tu.select_tool()
                    if selected:
                        result['tool_auto'] = selected
            except Exception:
                pass

        # 12.19 ★ D6: NarrativeSelf — 시간 + identity decay
        self.ns.tick(dt=dt)

        # 12.20 ★ v35 E1: CorpusLearner — no-op (외부 자극 모듈)
        self.cl.tick(dt=dt)

        # 12.21 ★ v35 E3: APCLearner — 매 tick 평생 학습
        # 모든 모듈 정리 후, 망각 전: 안정된 활성 상태에서 예측-학습
        apc_result = self.apc.tick(dt=dt)
        if apc_result and (apc_result['hit'] or apc_result['surprise']):
            result['apc'] = apc_result
            if self.verbose:
                print(f"  [{self.time:5.1f}s] [apc           ] "
                      f"hit={apc_result['hit']}, "
                      f"surprise={apc_result['surprise']}, "
                      f"miss={apc_result['miss']}")

        # 12.22 ★ v35 E4: DeepReasoning — no-op (외부 자극 모듈)
        self.dr.tick(dt=dt)

        # 12.23 ★ v35 E5: VSABinding — no-op (외부 자극 모듈)
        self.vsa.tick(dt=dt)

        # 12.24 ★ v35 E6: ContinualRehearsal — 주기적 정체성 재활성
        rehearsal_result = self.cr_rehearsal.tick(dt=dt)
        if rehearsal_result:
            result['rehearsal'] = rehearsal_result
            if self.verbose:
                print(f"  [{self.time:5.1f}s] [rehearsal     ] "
                      f"{', '.join(rehearsal_result[:3])}")

        # 12.25 ★ v35 E2: HyperGraph — 시간 진행 (no-op for edges)
        self.hg.tick(dt=dt)

        # 12.26 ★ v35 E7: CreativeAdvanced — no-op (외부 자극 모듈)
        self.ca.tick(dt=dt)

        # 12.27 ★ v35 E8: FrameSemantics — no-op (외부 자극 모듈)
        self.fs.tick(dt=dt)

        # 12.28 ★ v35 E9: SemanticDistance — no-op
        self.sd_dist.tick(dt=dt)

        # 12.29 ★ v35 F1: NormInternalization — 시간 감쇠 + DMN 자발 적용
        self.ni.tick(dt=dt)
        # 매 30 tick (15초)마다 DMN에 norm 자발 적용 시도
        if self.tick_count % 30 == 0:
            applied = self.ni.apply_norm_to_dmn(self.dmn)
            if applied:
                result['norm_applied'] = applied

        # 12.30 ★ v35 F2: AllostaticLearning — 호르몬 패턴 학습 + 자동 적용
        # tick 내부에서 자동 관찰 + 자동 추출 + 자동 적용 다 함
        self.al.tick(dt=dt)

        # 13. (주기적) 망각
        if self.tick_count % self.forget_interval_ticks == 0:
            cort = self.hs.hormones['cortisol'].level
            modifier = 1.0
            if cort > 0.6:
                modifier = 1.0 + (cort - 0.6) * 2.0
            elif self.hs.hormones.get('bdnf') and self.hs.hormones['bdnf'].level > 0.6:
                modifier = 0.5

            forget_result = self.sa.forget(
                dt=dt * self.forget_interval_ticks,
                base_decay=self.forget_base_decay,
                hormone_modifier=modifier,
                protected_categories=self.protected_categories,
            )
            # ★ C1: CausalGraph도 같은 흐름에 망각
            cg_forget = self.cg.forget(
                dt=dt * self.forget_interval_ticks,
                base_decay=self.forget_base_decay,
                hormone_modifier=modifier,
            )
            self.forget_runs += 1
            if forget_result.get('pairs_removed', 0) or forget_result.get('categories_removed', 0):
                result['forget'] = forget_result
                if self.verbose:
                    print(f"  [{self.time:5.1f}s] [forget] "
                          f"pairs -{forget_result['pairs_removed']}, "
                          f"cats -{forget_result['categories_removed']}, "
                          f"causal -{cg_forget['edges_removed']}")

        return result

    def live(self, duration: float, dt: float = 0.5,
            log_every: int = 0) -> List[Dict]:
        results = []
        steps = int(duration / dt)
        for step in range(steps):
            r = self.tick(dt=dt)
            results.append(r)
            if log_every > 0 and step % log_every == 0:
                self._print_status_brief()
        return results

    def _print_status_brief(self):
        primary = self.hs.primary_hormone()
        focus = self.wm.get_focus()
        feel = self.nl.feeling_to_text()
        print(f"    t={self.time:5.1f}s | focus={focus} | feel={feel} | "
              f"primary={primary[0]}({primary[1]:.2f})")

    # ============= 자기 보고 =============
    def introspect(self) -> Dict[str, Any]:
        mood = self.hs.compute_mood()
        primary_h = self.hs.primary_hormone()
        focus = self.wm.get_focus()
        active_cats = self.sa.get_active()
        recent_wandering = [c for _, c, _ in self.dmn.wandering_history[-5:]]
        dominant_need = self.ds.get_dominant_need()

        return {
            'time': self.time,
            'tick_count': self.tick_count,
            'feeling': self.nl.feeling_to_text(),
            'inner_voice': self.nl.inner_voice(self.dmn),
            'mood': mood,
            'primary_hormone': primary_h,
            'focus': focus,
            'active_categories': sorted(active_cats),
            'wm_slots_used': f"{len(self.wm.slots)}/{self.wm.current_capacity}",
            'recent_wandering': recent_wandering,
            'dmn_mode': self.dmn.current_mode,
            'spontaneous_count': self.dmn.spontaneous_count,
            'self_intent_count': self.dmn.self_intent_count,
            'broadcast_count': self.broadcast_count,
            'dominant_need': dominant_need,
            'sim_hour': self.hs.sim_hour,
            'beliefs_loaded': len(self.nl.beliefs),
            'discovered_count': self.nl.discovered_count,
            'refusal_count': self.nl.refusal_count,
            'forget_runs': self.forget_runs,
            'protected_categories': len(self.protected_categories),
            'graph_size': len(self.sa.neighbors),
            'graph_connections': len(self.sa.weights),
            # ★ B3 신규
            'doubt_baseline': self.sd.baseline,
            'doubt_baseline_with_hormone': self.sd.get_baseline_doubt(),
            'doubt_eval_count': self.sd.eval_count,
            'doubt_reject_count': self.sd.reject_count,
            'last_doubt_action': (self.sd.last_doubt['recommended_action']
                                  if self.sd.last_doubt else None),
            # ★ B2 정식 통합 신규
            'episodes_count': len(self.em.episodes),
            'episodes_categories': len(self.em.category_index),
            'em_encode_count': self.em.encode_count,
            'em_recall_count': self.em.recall_count,
            'em_remind_count': self.em.remind_count,
            'em_forget_count': self.em.forget_count,
            # ★ B4 정식 통합 신규
            'ai_predict_count': self.ai.predict_count,
            'ai_observe_count': self.ai.observe_count,
            'ai_update_count': self.ai.update_count,
            'ai_pending_predictions': sum(1 for p in self.ai.predictions
                                          if not p.get('observed')),
            # ★ B5 정식 통합 신규
            'ag_eval_count': self.ag.eval_count,
            'ag_do_count': self.ag.do_count,
            'ag_hesitate_count': self.ag.hesitate_count,
            'ag_skip_count': self.ag.skip_count,
            'ag_intentional_count': self.ag.intentional_count,
            'ag_spontaneous_count': self.ag.spontaneous_count,
            'ag_reactive_count': self.ag.reactive_count,
            'ag_actions_recorded': len(self.ag.actions),
            'ag_last_action': (dict(self.ag.actions[-1])
                              if self.ag.actions else None),
            # ★ B6 정식 통합 신규
            'mc_monitor_count': self.mc.monitor_count,
            'mc_endorse_count': self.mc.endorse_count,
            'mc_override_count': self.mc.override_count,
            'mc_calibration_count': self.mc.calibration_count,
            'mc_inconsistency_count': self.mc.inconsistency_count,
            'mc_observations': len(self.mc.observations),
            # ★ C1 정식 통합 신규
            'cg_nodes': self.cg.node_count(),
            'cg_edges': self.cg.edge_count(),
            'cg_learn_count': self.cg.learn_count,
            'cg_observe_count': self.cg.observe_count,
            'cg_query_count': self.cg.query_count,
            'cg_cycles_rejected': self.cg.cycle_rejected,
            'cg_cycles_replaced': self.cg.cycle_replaced,
            # ★ C3 정식 통합 신규
            'cf_simulate_count': self.cf.simulate_count,
            'cf_regret_count': self.cf.regret_count,
            'cf_relief_count': self.cf.relief_count,
            'cf_simulations': len(self.cf.simulations),
            # ★ C5 정식 통합 신규
            'tm_recall_count': self.tm.recall_count,
            'tm_future_count': self.tm.future_count,
            'tm_precedes_count': self.tm.precedes_count,
            'tm_subjective_factor': self.tm.perceived_time_factor(),
            # ★ C7 정식 통합 신규
            'er_regulate_count': self.er.regulate_count,
            'er_auto_count': self.er.auto_trigger_count,
            'er_distraction_count': self.er.distraction_count,
            'er_reappraisal_count': self.er.reappraisal_count,
            'er_suppression_count': self.er.suppression_count,
            'er_avg_effectiveness': self.er.get_state()['avg_effectiveness'],
            # ★ C4 정식 통합 신규
            'an_find_count': self.an.find_count,
            'an_explain_count': self.an.explain_count,
            'an_similarity_count': self.an.similarity_count,
            # ★ C2 정식 통합 신규 (티어 C 마지막)
            'world_describe_count': self.wm_world.describe_count,
            'world_list_domains_count': self.wm_world.list_domains_count,
            'world_categorize_count': self.wm_world.categorize_count,
            'world_schema_count': self.wm_world.schema_count,
            'world_relate_count': self.wm_world.relate_count,
            # ★ C6 정식 통합 신규
            'gm_active_count': len([g for g in self.gm.goals.values()
                                    if g.status == 'active']),
            'gm_set_count': self.gm.set_count,
            'gm_completed_count': self.gm.completed_count,
            'gm_abandoned_count': self.gm.abandoned_count,
            'gm_expired_count': self.gm.expired_count,
            'gm_proposed_count': self.gm.proposed_count,
            'gm_suggested_count': self.gm.suggested_count,
            'gm_top_goal': (
                {'category': self.gm.active_goals()[0].category,
                 'priority': self.gm.active_goals()[0].priority,
                 'progress': self.gm.active_goals()[0].progress}
                if self.gm.active_goals() else None
            ),
            # ★ D1 정식 통합 신규 (티어 D 시작)
            'cr_creative_mode': self.cr.is_creative_mode(),
            'cr_incubating': sorted(self.cr.incubating.keys()),
            'cr_remote_count': self.cr.remote_count,
            'cr_blend_count': self.cr.blend_count,
            'cr_incubate_count': self.cr.incubate_count,
            'cr_illuminate_count': self.cr.illuminate_count,
            'cr_verify_count': self.cr.verify_count,
            # ★ D2 정식 통합 신규
            'suf_resonate_count': self.suf.resonate_count,
            'suf_distinguish_count': self.suf.distinguish_count,
            'suf_compassion_count': self.suf.compassion_count,
            'suf_burnout_count': self.suf.burnout_count,
            'suf_current_load': self.suf.burnout_check()['load'],
            'suf_in_burnout': self.suf.in_burnout,
            # ★ D3 정식 통합 신규
            'hu_humor_mode': self.hu.is_humor_safe(),
            'hu_measure_count': self.hu.measure_count,
            'hu_generate_count': self.hu.generate_count,
            'hu_laugh_count': self.hu.laugh_count,
            'hu_unsafe_blocks': self.hu.unsafe_blocks,
            'hu_recent_jokes': [(s, p) for s, p, _ in
                                list(self.hu.joke_history)[-3:]],
            # ★ D4 정식 통합 신규
            'ms_compete_count': self.ms.compete_count,
            'ms_cross_talk_count': self.ms.cross_talk_count,
            'ms_integration_count': self.ms.integration_count,
            'ms_recent_focus': self.ms.focus_history[-3:],
            'ms_stream_sizes': {
                name: len(s.queue)
                for name, s in self.ms.streams.items()
            },
            # ★ D5 정식 통합 신규
            'tu_tools_registered': len(self.tu.tools),
            'tu_invoke_count': self.tu.invoke_count,
            'tu_select_count': self.tu.select_count,
            'tu_reflect_count': self.tu.reflect_count,
            'tu_failed_count': self.tu.failed_count,
            'tu_recent_tools': [r['tool'] for r in
                               list(self.tu.history)[-3:]],
            # ★ D6 정식 통합 신규 (티어 D 마지막)
            'ns_identity_size': len(self.ns.identity_stability),
            'ns_top_identity': sorted(
                self.ns.identity_stability.items(),
                key=lambda x: (-x[1], x[0])
            )[:3],
            'ns_extract_count': self.ns.extract_count,
            'ns_narrate_count': self.ns.narrate_count,
            'ns_integrate_count': self.ns.integrate_count,
            'ns_coherence_count': self.ns.coherence_count,
            # ★ v35 E1: CorpusLearner — 대규모 텍스트 학습
            'cl_documents': self.cl.documents_processed,
            'cl_paragraphs': self.cl.paragraphs_processed,
            'cl_sentences': self.cl.sentences_processed,
            'cl_categories_added': self.cl.categories_added,
            'cl_pairs_strengthened': self.cl.pairs_strengthened,
            'cl_cooc_pairs': len(self.cl.cooc),
            'cl_top_categories': self.cl._top_categories(5),
            # ★ v35 E3: APCLearner — 평생 예측 학습
            'apc_observe_count': self.apc.observe_count,
            'apc_hits': self.apc.hits_total,
            'apc_misses': self.apc.misses_total,
            'apc_surprises': self.apc.surprises_total,
            'apc_pairs_learned': self.apc.pairs_learned,
            'apc_skipped_quiet': self.apc.skipped_quiet,
            'apc_accuracy': (self.apc.hits_total /
                            max(1, self.apc.hits_total + self.apc.misses_total)),
            # ★ v35 E4: DeepReasoning — 멀티홉 추론
            'dr_query_count': self.dr.query_count,
            'dr_path_count': self.dr.path_count,
            'dr_no_path_count': self.dr.no_path_count,
            'dr_deepest_path': self.dr.deepest_path,
            # ★ v35 E5: VSABinding — 의미 합성
            'vsa_dim': self.vsa.dim,
            'vsa_cached_vectors': len(self.vsa.vectors),
            'vsa_bound_categories': len(self.vsa.bound_meta),
            'vsa_bind_count': self.vsa.bind_count,
            'vsa_compose_count': self.vsa.compose_count,
            'vsa_unbind_count': self.vsa.unbind_count,
            # ★ v35 E6: ContinualRehearsal — 정체성 보존
            'cr_rehearse_count': self.cr_rehearsal.rehearse_count,
            'cr_categories_rehearsed': self.cr_rehearsal.categories_rehearsed,
            'cr_protected': self.cr_rehearsal.get_protected_categories(top_n=5),
            'cr_last_rehearsed': list(self.cr_rehearsal.last_rehearsed),
            # ★ v35 E2: HyperGraph — n-ary 관계
            'hg_edge_count': len(self.hg.edges),
            'hg_node_count': len(self.hg.node_to_edges),
            'hg_role_count': len(self.hg.role_index),
            'hg_add_count': self.hg.add_count,
            'hg_query_count': self.hg.query_count,
            # ★ v35 E7: CreativeAdvanced — 창의 사이클
            'ca_creative_mode': self.ca.is_creative_mode(),
            'ca_compose_novel_count': self.ca.compose_novel_count,
            'ca_imagine_event_count': self.ca.imagine_event_count,
            'ca_bridge_count': self.ca.bridge_count,
            'ca_cycle_count': self.ca.cycle_count,
            'ca_cycle_completed': self.ca.cycle_completed,
            # ★ v35 E8: FrameSemantics — 한국어 SVO
            'fs_parse_count': self.fs.parse_count,
            'fs_frame_count': self.fs.frame_count,
            'fs_hg_added_count': self.fs.hg_added_count,
            'fs_failed_parse': self.fs.failed_parse,
            # ★ v35 E9: SemanticDistance — 의미 거리
            'sd_dist_count': self.sd_dist.distance_count,
            'sd_cache_hits': self.sd_dist.cache_hits,
            'sd_find_similar_count': self.sd_dist.find_similar_count,
            'sd_cluster_count': self.sd_dist.cluster_count,
            # ★ v35 F1: NormInternalization
            'ni_norm_count': len(self.ni.norms),
            'ni_observed_count': self.ni.observed_count,
            'ni_internalized_count': self.ni.internalized_count,
            'ni_applied_count': self.ni.applied_count,
            'ni_rejected_count': self.ni.rejected_count,
            # ★ v35 F2: AllostaticLearning
            'al_pattern_count': len(self.al.patterns),
            'al_observe_count': self.al.observe_count,
            'al_learned_count': self.al.learned_count,
            'al_apply_count': self.al.apply_count,
            'al_pending_obs': len(self.al.pending_observations),
        }

    def report(self):
        intro = self.introspect()
        print(f"\n{'=' * 60}")
        print(f"EVE 상태 보고 @ t={intro['time']:.1f}s")
        print(f"{'=' * 60}")
        print(f"feeling      : {intro['feeling']}")
        print(f"inner_voice  : {intro['inner_voice']}")
        print(f"mood         : v={intro['mood']['valence']:+.2f} "
              f"a={intro['mood']['arousal']:.2f} d={intro['mood']['dominance']:.2f}")
        print(f"primary horm : {intro['primary_hormone'][0]} "
              f"({intro['primary_hormone'][1]:.2f})")
        print(f"focus        : {intro['focus']}")
        print(f"active cats  : {len(intro['active_categories'])} "
              f"{intro['active_categories'][:8]}")
        print(f"WM           : {intro['wm_slots_used']}")
        print(f"DMN          : {intro['dmn_mode']} "
              f"(spont={intro['spontaneous_count']}, "
              f"self_intent={intro['self_intent_count']})")
        print(f"recent wander: {intro['recent_wandering']}")
        print(f"dominant_need: {intro['dominant_need']}")
        print(f"sim_hour     : {intro['sim_hour']:.1f}h")
        print(f"beliefs      : {intro['beliefs_loaded']} "
              f"(discovered={intro['discovered_count']}, "
              f"refusals={intro['refusal_count']})")
        print(f"graph        : {intro['graph_size']} cats, "
              f"{intro['graph_connections']} pairs "
              f"(forget runs={intro['forget_runs']})")
        # ★ B3
        print(f"doubt        : baseline={intro['doubt_baseline']:.2f}, "
              f"with_horm={intro['doubt_baseline_with_hormone']:.2f}, "
              f"evals={intro['doubt_eval_count']}, "
              f"rejects={intro['doubt_reject_count']}, "
              f"last={intro['last_doubt_action']}")
        # ★ B2
        print(f"episodes     : {intro['episodes_count']} "
              f"(cats={intro['episodes_categories']}, "
              f"encoded={intro['em_encode_count']}, "
              f"recalled={intro['em_recall_count']}, "
              f"reminded={intro['em_remind_count']}, "
              f"forgot={intro['em_forget_count']})")
        # ★ B4
        print(f"active inf   : predicts={intro['ai_predict_count']}, "
              f"observes={intro['ai_observe_count']}, "
              f"updates={intro['ai_update_count']}, "
              f"pending={intro['ai_pending_predictions']}")
        # ★ B5
        print(f"agency       : evals={intro['ag_eval_count']}, "
              f"do/hes/skip={intro['ag_do_count']}/"
              f"{intro['ag_hesitate_count']}/{intro['ag_skip_count']}, "
              f"int/spo/rea={intro['ag_intentional_count']}/"
              f"{intro['ag_spontaneous_count']}/{intro['ag_reactive_count']}")
        # ★ B6
        print(f"metacog      : monitors={intro['mc_monitor_count']}, "
              f"endorse/override={intro['mc_endorse_count']}/"
              f"{intro['mc_override_count']}, "
              f"calibrations={intro['mc_calibration_count']} "
              f"(inconsistent={intro['mc_inconsistency_count']})")
        # ★ C1
        print(f"causal       : nodes={intro['cg_nodes']}, "
              f"edges={intro['cg_edges']}, "
              f"learns={intro['cg_learn_count']}, "
              f"queries={intro['cg_query_count']}, "
              f"cycles_rej/repl={intro['cg_cycles_rejected']}/"
              f"{intro['cg_cycles_replaced']}")
        # ★ C3
        print(f"counterfact  : sims={intro['cf_simulate_count']}, "
              f"regret/relief={intro['cf_regret_count']}/"
              f"{intro['cf_relief_count']}")
        # ★ C5
        print(f"temporal     : recalls={intro['tm_recall_count']}, "
              f"futures={intro['tm_future_count']}, "
              f"precedes={intro['tm_precedes_count']}, "
              f"subjective={intro['tm_subjective_factor']:.2f}x")
        # ★ C7
        eff = intro['er_avg_effectiveness']
        eff_str = ', '.join(
            f"{s[:4]}={v:+.2f}" if v is not None else f"{s[:4]}=N/A"
            for s, v in eff.items()
        )
        print(f"emotion_reg  : regulates={intro['er_regulate_count']} "
              f"(auto={intro['er_auto_count']}), "
              f"distr/reap/supp={intro['er_distraction_count']}/"
              f"{intro['er_reappraisal_count']}/"
              f"{intro['er_suppression_count']}, "
              f"effects=[{eff_str}]")
        # ★ C4
        print(f"analogy      : find={intro['an_find_count']}, "
              f"explain={intro['an_explain_count']}, "
              f"sim={intro['an_similarity_count']}")
        # ★ C2 — 티어 C 마지막, 전체 지식 view
        print(f"world_model  : describes={intro['world_describe_count']}, "
              f"domains={intro['world_list_domains_count']}, "
              f"schemas={intro['world_schema_count']}, "
              f"relates={intro['world_relate_count']}")
        # ★ C6
        top = intro['gm_top_goal']
        top_str = (f"{top['category']}({top['priority']:.2f},{top['progress']:.2f})"
                  if top else "none")
        print(f"goals        : active={intro['gm_active_count']}, "
              f"completed={intro['gm_completed_count']}, "
              f"abandoned={intro['gm_abandoned_count']}, "
              f"proposed={intro['gm_proposed_count']}, "
              f"top={top_str}")
        # ★ D1 (티어 D 시작)
        print(f"creative     : mode={'ON' if intro['cr_creative_mode'] else 'off'}, "
              f"remote={intro['cr_remote_count']}, "
              f"blend={intro['cr_blend_count']}, "
              f"incubate={intro['cr_incubate_count']}, "
              f"illuminate={intro['cr_illuminate_count']}, "
              f"incubating={intro['cr_incubating']}")
        # ★ D2
        bo_marker = ' BURNOUT' if intro['suf_in_burnout'] else ''
        print(f"empathy      : resonate={intro['suf_resonate_count']}, "
              f"distinguish={intro['suf_distinguish_count']}, "
              f"comfort={intro['suf_compassion_count']}, "
              f"load={intro['suf_current_load']:.2f}{bo_marker}")
        # ★ D3
        recent_jokes = intro['hu_recent_jokes']
        joke_str = ', '.join(f"{s}→{p}" for s, p in recent_jokes[-2:]) if recent_jokes else 'none'
        print(f"humor        : mode={'ON' if intro['hu_humor_mode'] else 'off'}, "
              f"jokes={intro['hu_generate_count']}, "
              f"laughs={intro['hu_laugh_count']}, "
              f"recent=[{joke_str}]")
        # ★ D4
        sizes = intro['ms_stream_sizes']
        sizes_str = '/'.join(str(sizes[k]) for k in
                            ['external', 'inner', 'emotion', 'planning', 'recall'])
        recent_focus = intro['ms_recent_focus']
        focus_str = recent_focus[-1][1] + ':' + recent_focus[-1][2] if recent_focus else 'none'
        print(f"streams      : queues={sizes_str} (ext/inn/emo/plan/rec), "
              f"integrations={intro['ms_integration_count']}, "
              f"last_focus={focus_str}")
        # ★ D5
        recent_tools = intro['tu_recent_tools']
        tools_str = ', '.join(recent_tools[-3:]) if recent_tools else 'none'
        print(f"tool_use     : registered={intro['tu_tools_registered']}, "
              f"invoked={intro['tu_invoke_count']}, "
              f"selected={intro['tu_select_count']}, "
              f"recent=[{tools_str}]")
        # ★ D6 — 자기 서사 (티어 D 마지막)
        top_id = intro['ns_top_identity']
        id_str = ', '.join(f"{c}({s:.2f})" for c, s in top_id) if top_id else 'forming'
        print(f"narrative    : identity_size={intro['ns_identity_size']}, "
              f"top=[{id_str}], "
              f"narrates={intro['ns_narrate_count']}")
        # ★ v35 E1 — 코퍼스 학습
        top_cats = intro['cl_top_categories']
        top_str = ', '.join(f"{c}({n})" for c, n in top_cats[:3]) if top_cats else 'empty'
        print(f"corpus       : docs={intro['cl_documents']}, "
              f"sentences={intro['cl_sentences']}, "
              f"new_cats={intro['cl_categories_added']}, "
              f"top=[{top_str}]")
        # ★ v35 E3 — 평생 학습 (APC)
        print(f"apc_lifelong : observe={intro['apc_observe_count']}, "
              f"hits={intro['apc_hits']}, "
              f"surprises={intro['apc_surprises']}, "
              f"misses={intro['apc_misses']}, "
              f"acc={intro['apc_accuracy']:.2f}")
        # ★ v35 E4 — 멀티홉 추론 (DR)
        print(f"deep_reason  : queries={intro['dr_query_count']}, "
              f"paths_found={intro['dr_path_count']}, "
              f"no_path={intro['dr_no_path_count']}, "
              f"deepest={intro['dr_deepest_path']}")
        # ★ v35 E5 — 의미 합성 (VSA)
        print(f"vsa_compose  : dim={intro['vsa_dim']}, "
              f"vectors={intro['vsa_cached_vectors']}, "
              f"binds={intro['vsa_bind_count']}, "
              f"composes={intro['vsa_compose_count']}, "
              f"bound_cats={intro['vsa_bound_categories']}")
        # ★ v35 E6 — 정체성 보존 (Rehearsal)
        protected = intro['cr_protected']
        prot_str = ', '.join(protected[:3]) if protected else 'none'
        print(f"rehearsal    : runs={intro['cr_rehearse_count']}, "
              f"total_cats={intro['cr_categories_rehearsed']}, "
              f"protected=[{prot_str}]")
        # ★ v35 E2 — n-ary 관계 (HyperGraph)
        print(f"hypergraph   : edges={intro['hg_edge_count']}, "
              f"nodes={intro['hg_node_count']}, "
              f"roles={intro['hg_role_count']}, "
              f"queries={intro['hg_query_count']}")
        # ★ v35 E7 — 창의 강화 (CA)
        print(f"creative_adv : mode={intro['ca_creative_mode']}, "
              f"compose={intro['ca_compose_novel_count']}, "
              f"imagine={intro['ca_imagine_event_count']}, "
              f"bridges={intro['ca_bridge_count']}, "
              f"cycles={intro['ca_cycle_count']}/{intro['ca_cycle_completed']}")
        # ★ v35 E8 — Frame Semantics
        print(f"frame_sem    : parses={intro['fs_parse_count']}, "
              f"frames={intro['fs_frame_count']}, "
              f"hg_added={intro['fs_hg_added_count']}, "
              f"failed={intro['fs_failed_parse']}")
        # ★ v35 E9 — 의미 거리
        print(f"semantic_dist: distance={intro['sd_dist_count']}, "
              f"cache_hits={intro['sd_cache_hits']}, "
              f"similars={intro['sd_find_similar_count']}, "
              f"clusters={intro['sd_cluster_count']}")
        # ★ v35 F1 — Norm Internalization
        print(f"norm_internal: norms={intro['ni_norm_count']}, "
              f"observed={intro['ni_observed_count']}, "
              f"internalized={intro['ni_internalized_count']}, "
              f"applied={intro['ni_applied_count']}, "
              f"rejected={intro['ni_rejected_count']}")
        # ★ v35 F2 — Allostatic Learning
        print(f"allostatic   : patterns={intro['al_pattern_count']}, "
              f"observed={intro['al_observe_count']}, "
              f"learned={intro['al_learned_count']}, "
              f"applied={intro['al_apply_count']}")
        print(f"{'=' * 60}\n")

    def __repr__(self):
        return (f"EVE_TierAB(t={self.time:.1f}s, "
                f"horm={self.hs.primary_hormone()[0]}, "
                f"focus={self.wm.get_focus()}, "
                f"doubt={self.sd.baseline:.2f})")


# 하위 호환 alias
EVE_TierA = EVE_TierAB
