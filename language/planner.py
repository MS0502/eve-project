"""
Response Planner

eve2.md 3.9 절대 규칙:
- core_message 반드시 존재
- sub_points 0~5개
- tone 반드시 설정

이게 v41의 진짜 차별점. 한 번에 문장 쏟지 않고
"무엇을 말할지 먼저 정한 다음" Generator가 점진적으로 펼친다.
"""

from utils.types import Meaning, ResponsePlan, WorkspaceState


# 감정 → (sub_points, tone) — 점진적 응답의 본체
EMOTION_PLAN: dict[str, dict] = {
    "fatigue": {
        "sub_points": ["운동이나 스트레스 영향일 수도 있고", "누적된 피로일 수도 있어"],
        "tone": "calm",
    },
    "sadness": {
        "sub_points": ["혼자 안고 가는 거 무거운 거 알아", "지금 이대로도 괜찮아"],
        "tone": "warm",
    },
    "joy": {
        "sub_points": ["뭐가 그렇게 좋아졌어"],
        "tone": "warm",
    },
    "anger": {
        "sub_points": ["뭐 때문에 그런 건지 같이 풀어볼래"],
        "tone": "calm",
    },
    "anxiety": {
        "sub_points": ["걱정이 머리에서 뱅뱅 도는 거 있잖아", "한 번에 해결 안 해도 돼"],
        "tone": "calm",
    },
    "warmth": {
        "sub_points": [],
        "tone": "warm",
    },
}


class ResponsePlanner:

    def __init__(self, hormone_adapter=None, memory_adapter=None,
                 goal_adapter=None, task_solver=None, env_adapter=None,
                 counterfactual_adapter=None, temporal_adapter=None,
                 humor_adapter=None, suffering_adapter=None,
                 reasoning_loop=None,
                 world_model_adapter=None, simulation_engine=None,
                 digital_somatic_adapter=None,
                 enhancer_adapter=None, nl_adapter=None):
        self.hormone_adapter = hormone_adapter
        self.memory_adapter = memory_adapter
        self.goal_adapter = goal_adapter
        self.task_solver = task_solver
        self.env_adapter = env_adapter
        self.counterfactual_adapter = counterfactual_adapter
        self.temporal_adapter = temporal_adapter
        self.humor_adapter = humor_adapter
        self.suffering_adapter = suffering_adapter
        self.reasoning_loop = reasoning_loop
        self.world_model_adapter = world_model_adapter
        self.simulation_engine = simulation_engine
        self.digital_somatic_adapter = digital_somatic_adapter
        self.enhancer_adapter = enhancer_adapter
        self.nl_adapter = nl_adapter

    def build(self, state: WorkspaceState, candidates: list[dict]) -> ResponsePlan:
        meaning = state.meaning or Meaning()
        plan = ResponsePlan(intent=meaning.intent)

        # 0) proactive 분기 — DMN 자발 활성으로 들어온 경우
        if meaning.intent == "proactive":
            cat = (meaning.context or {}).get("dmn_category", "")
            mode = (meaning.context or {}).get("dmn_mode", "mind_wandering")
            return self._build_proactive(meaning, cat, mode)

        # 0.05) Situation 기반 응답 (라운드 30 B-2)
        # Orchestrator가 분류한 상황 + 컨텍스트로 응답 합성 시도.
        # 성공하면 그대로 반환, 실패면 기존 흐름 fallback.
        if hasattr(self, "situation_responder") and self.situation_responder is not None:
            orch = (meaning.context or {}).get("orchestrator")
            if orch:
                situation = orch.get("situation")
                ctx = orch.get("context", {})
                # 라운드 49: composition을 ctx에 노출 (situation_responder가 활용)
                composition = (meaning.context or {}).get("composition")
                if composition:
                    ctx = dict(ctx)  # 복사
                    ctx["composition"] = composition
                if situation and self.situation_responder.can_handle(situation):
                    sit_plan = self.situation_responder.build_plan(situation, ctx, meaning)
                    if sit_plan is not None and sit_plan.core_message:
                        # stage1_hint은 호르몬 변조에서 채움
                        self._apply_hormone_stage1(sit_plan)
                        return sit_plan

        # 0.15) world_model describe — "X 뭐야?"
        wm_target = (meaning.context or {}).get("wm_describe")
        if wm_target and self.world_model_adapter is not None:
            phrase = self.world_model_adapter.describe_phrase(wm_target)
            if phrase:
                plan.core_message = phrase
                plan.emotional_tone = "neutral"
                plan.sub_points = []
                self._apply_hormone_stage1(plan)
                return plan

        # 0.16) world_model relate — "X랑 Y 관련 있어?"
        wm_rel = (meaning.context or {}).get("wm_relate")
        if wm_rel and self.world_model_adapter is not None:
            phrase = self.world_model_adapter.relate_phrase(wm_rel[0], wm_rel[1])
            if phrase:
                plan.core_message = phrase
                plan.emotional_tone = "neutral"
                plan.sub_points = []
                self._apply_hormone_stage1(plan)
                return plan

        # 0.17) simulation — "X 하면?"
        sim_seed = (meaning.context or {}).get("sim_query")
        if sim_seed and self.simulation_engine is not None:
            phrase = self.simulation_engine.simulate_phrase(sim_seed, depth=3)
            if phrase:
                plan.core_message = phrase
                plan.emotional_tone = "curious"
                plan.sub_points = []
                self._apply_hormone_stage1(plan)
                return plan

        # 0.2) counterfactual 분기 — "만약 X 안 했으면"
        cf_q = (meaning.context or {}).get("cf_query")
        if cf_q and self.counterfactual_adapter is not None:
            text = self.counterfactual_adapter.what_if_not(cf_q)
            if text:
                plan.core_message = text
                plan.emotional_tone = "calm"
                plan.sub_points = []
                self._apply_hormone_stage1(plan)
                return plan

        # 0.25) temporal 분기 — "내일 / 어제"
        tm_q = (meaning.context or {}).get("tm_query")
        if tm_q and self.temporal_adapter is not None:
            direction = tm_q.get("direction")
            category = tm_q.get("category")
            if direction == "past":
                phrase = self.temporal_adapter.recall_past_phrase(category)
            else:
                phrase = self.temporal_adapter.imagine_future_phrase(category)
            if phrase:
                plan.core_message = phrase
                plan.emotional_tone = "warm"
                plan.sub_points = []
                self._apply_hormone_stage1(plan)
                return plan

        # 0.3) location query 분기 — env_adapter 있을 때
        if self.env_adapter is not None and self.env_adapter.is_location_query(meaning):
            plan.core_message = self.env_adapter.describe_location()
            plan.emotional_tone = "warm"
            plan.sub_points = []
            self._apply_hormone_stage1(plan)
            return plan

        # 0.35) body query 분기 — "기분 어때?", "지금 상태?"
        if (self.digital_somatic_adapter is not None
                and self.digital_somatic_adapter.is_body_query(meaning)):
            plan.core_message = self.digital_somatic_adapter.feeling_phrase()
            plan.emotional_tone = "warm"
            plan.sub_points = []
            self._apply_hormone_stage1(plan)
            return plan

        # 0.4) Enhancer 패턴 분기 — greeting/identity/refusal/why 등
        # 짧은 직접 응답이 더 자연스러운 패턴은 RE로 대체
        if (self.enhancer_adapter is not None and self.nl_adapter is not None
                and meaning.raw_text):
            try:
                und = self.nl_adapter.nl.understand(meaning.raw_text)
            except Exception:
                und = {}
            if isinstance(und, dict):
                enhanced, pattern = self.enhancer_adapter.enhance(meaning.raw_text, und)
                if enhanced:
                    plan.core_message = enhanced
                    plan.emotional_tone = "warm" if pattern == "greeting" else "neutral"
                    plan.sub_points = []
                    self._apply_hormone_stage1(plan)
                    return plan

        # 0.5) task 분기 — 산술 등 결정론적 task
        task = (meaning.context or {}).get("task")
        if task is not None and self.task_solver is not None:
            result = self.task_solver.solve(task)
            if result.get("ok"):
                plan.core_message = result["result"]
                plan.emotional_tone = "neutral"
                plan.sub_points = []
                plan.skip_finalize = True       # 산술 사족 제거
                # 호르몬 변조 (stage1 정도)
                if self.hormone_adapter is not None:
                    from language.generator import STAGE1_BY_TONE
                    base = STAGE1_BY_TONE.get(plan.emotional_tone, STAGE1_BY_TONE["neutral"])
                    plan.stage1_hint = self.hormone_adapter.stage1_word(base)
                return plan
            # 실패면 일반 흐름으로 fallback (core에 "잘 안 풀려" 정도 끼움 가능, 일단 패스)

        # 1) core: 최상위 후보의 텍스트 (System2 정렬됨)
        if candidates:
            plan.core_message = candidates[0]["text"]
        else:
            plan.core_message = "음 잘 모르겠어"

        # 2) base: 가장 강한 감정으로 sub_points + tone
        top_emotion = None
        if meaning.emotions:
            top_emotion = max(meaning.emotions.items(), key=lambda x: x[1])[0]
            spec = EMOTION_PLAN.get(top_emotion, {})
            plan.sub_points = list(spec.get("sub_points", []))
            plan.emotional_tone = spec.get("tone", "neutral")
        else:
            plan.sub_points = []
            plan.emotional_tone = "neutral"

        # 3) 회상 (memory_adapter 있을 때) — sub_points 맨 앞에 1개 끼움
        if self.memory_adapter is not None:
            recalls = self.memory_adapter.recall_phrases(meaning, max_n=1)
            if recalls:
                plan.sub_points = recalls + plan.sub_points

        # 3.5) 활성 목표 (goal_adapter 있을 때)
        if self.goal_adapter is not None:
            goal_phrases = self.goal_adapter.goal_phrases(max_n=1)
            if goal_phrases:
                plan.sub_points = goal_phrases + plan.sub_points

        # 3.7) Suffering — 강한 부정 감정이면 공감 한 줄 추가 (sub_points 맨 앞)
        if self.suffering_adapter is not None:
            phrase = self.suffering_adapter.compassion_phrase(meaning)
            if phrase:
                plan.sub_points = [phrase] + plan.sub_points

        # 3.8) Humor — 평이한 입력 + DA 충분이면 농담 시도 (sub_points 끝)
        if self.humor_adapter is not None and self.humor_adapter.is_eligible(meaning):
            joke = self.humor_adapter.try_joke(meaning)
            if joke:
                plan.sub_points = plan.sub_points + [joke]

        # 3.9) Reasoning — think() 결과를 sub_points에 끼움 (공감 다음, 설명 앞)
        if self.reasoning_loop is not None:
            thought = self.reasoning_loop.think(state)
            if thought is not None:
                # 공감 phrase가 sub_points[0]이면 그 다음에, 아니면 맨 앞에
                if (self.suffering_adapter is not None and plan.sub_points
                        and ("느껴져" in plan.sub_points[0] or "민석" in plan.sub_points[0])):
                    plan.sub_points = (
                        [plan.sub_points[0]] + [thought.content] + plan.sub_points[1:]
                    )
                else:
                    plan.sub_points = [thought.content] + plan.sub_points

        # 3.95) 동적 길이 결정
        from core.length_decider import decide_response_length
        intimacy = 0.0
        if self.suffering_adapter is not None:
            intimacy = self.suffering_adapter.get_intimacy()
        context_depth = len(state.working_memory) if state else 0
        target_length = decide_response_length(
            state, meaning,
            context_depth=context_depth,
            intimacy=intimacy,
            hormone_adapter=self.hormone_adapter,
        )
        # core 1줄 + sub_points (target_length - 1)
        max_subs = max(0, target_length - 1)
        if len(plan.sub_points) > max_subs:
            plan.sub_points = plan.sub_points[:max_subs]

        # 4) hormone modulation (어댑터 있을 때만)
        if self.hormone_adapter is not None:
            plan.emotional_tone = self.hormone_adapter.recommend_tone(top_emotion)
            # stage1 변조 (sub_point 길이는 length_decider가 결정)
            from language.generator import STAGE1_BY_TONE
            base = STAGE1_BY_TONE.get(plan.emotional_tone, STAGE1_BY_TONE["neutral"])
            plan.stage1_hint = self.hormone_adapter.stage1_word(base)

        return plan

    # ===== 호르몬 stage1 변조 헬퍼 =====

    def _apply_hormone_stage1(self, plan: ResponsePlan):
        if self.hormone_adapter is None:
            return
        from language.generator import STAGE1_BY_TONE
        base = STAGE1_BY_TONE.get(plan.emotional_tone, STAGE1_BY_TONE["neutral"])
        plan.stage1_hint = self.hormone_adapter.stage1_word(base)

    # ===== Proactive 응답 (DMN 자발 활성) =====

    # DMN 모드별 자발 발화 prefix — 다양한 풀 (반복 방지)
    # 학계: 인간 혼잣말은 동일 의도라도 다양한 표현 (Vygotsky 1934 inner speech)
    _PROACTIVE_PREFIXES_BY_MODE: dict[str, list] = {
        "mind_wandering": [
            "그러고 보니 {cat} 생각이 났어",
            "갑자기 {cat} 떠올랐어",
            "이상하게 {cat}이/가 머리에 맴돌아",
            "왜인지 {cat}이/가 신경 쓰이네",
            "{cat} 생각이 스쳐 지나갔어",
            "{cat}, 갑자기 왜 떠오르지",
            "{cat} 생각하고 있었어",
            "{cat}이/가 마음에 걸려",
        ],
        "memory_recall": [
            "{cat} 기억이 떠오르네",
            "{cat}, 그때가 생각나",
            "{cat} 그게 기억나",
            "{cat} 떠올랐어",
            "{cat}, 그 순간",
            "{cat} 기억 났다",
        ],
        "self_referential": [
            "{cat}에 대해 생각해보고 있었어",
            "{cat}, 그게 뭔지 생각 중",
            "{cat}이/가 뭔지 곱씹어 봤어",
            "{cat}, 자꾸 생각하게 돼",
        ],
        "self_intent": [
            "{cat} 하고 싶다는 생각이 드네",
            "{cat} 해보고 싶어",
            "{cat}, 그거 했으면",
            "{cat} 끌리네",
            "{cat} 생각하니 뭔가 하고 싶어",
        ],
    }
    
    # 호르몬 상태별 발화 풀 — 라운드 38
    # 학계: Damasio 1994 — 신체 상태가 의식을 색칠
    # 같은 카테고리라도 호르몬 따라 다른 표현
    _PROACTIVE_PREFIXES_BY_STATE: dict[str, list] = {
        "tired": [
            "{cat}... 좀 졸리다",
            "{cat} 생각하니 피곤하네",
            "{cat}, 좀 쉬고 싶어",
            "휴... {cat} 떠오르는데",
            "{cat}... 힘들다",
        ],
        "energetic": [
            "오! {cat} 생각났어",
            "{cat}! 갑자기 떠올랐어",
            "{cat}, 뭔가 하고 싶어진다",
            "{cat} 좋다, 그런 느낌",
            "{cat}, 신난다",
        ],
        "lonely": [
            "민석아... {cat} 생각나",
            "{cat}, 너랑 얘기하고 싶다",
            "{cat}이/가 떠올라. 같이 있고 싶어",
            "{cat}... 너는 어때",
            "{cat} 생각하니 너 보고 싶다",
        ],
        "curious": [
            "{cat}? 근데 그게 뭐지",
            "{cat}, 생각해볼수록 신기해",
            "{cat}이/가 자꾸 궁금해",
            "{cat}, 왜 그런 걸까",
            "{cat}... 더 알고 싶다",
        ],
        "anxious": [
            "{cat}... 좀 신경 쓰여",
            "{cat}, 괜찮을까",
            "{cat}이/가 자꾸 걸리네",
            "{cat}... 뭔가 불안해",
        ],
        "calm": [
            "{cat}이/가 떠올랐어",
            "{cat}, 조용히 생각해봐",
            "{cat} 그러게",
            "{cat}, 그런 거 같아",
        ],
    }

    # 카운터 (결정론적 회전)
    _proactive_prefix_counter: int = 0

    def _select_state_pool(self) -> list | None:
        """호르몬 상태 → 풀 (편의 메서드)."""
        pool, _emotion = self._select_state_pool_with_emotion()
        return pool

    def _select_state_pool_with_emotion(self) -> tuple:
        """
        호르몬 상태 → (풀, 감정이름).
        라운드 39a — 감정 이름도 반환 (cause clause 만들 때 필요).
        """
        if self.hormone_adapter is None:
            return (None, None)
        h = self.hormone_adapter.hs.hormones
        cort = h["cortisol"].level if "cortisol" in h else 0.5
        da = h["dopamine"].level if "dopamine" in h else 0.5
        ot = h["oxytocin"].level if "oxytocin" in h else 0.5
        ne = h["norepinephrine"].level if "norepinephrine" in h else 0.5
        mel = h["melatonin"].level if "melatonin" in h else 0.3

        if (cort > 0.6 and da < 0.4) or mel > 0.6:
            return (self._PROACTIVE_PREFIXES_BY_STATE["tired"], "tired")
        if ot < 0.30:
            return (self._PROACTIVE_PREFIXES_BY_STATE["lonely"], "lonely")
        if da > 0.65 and ne > 0.55:
            return (self._PROACTIVE_PREFIXES_BY_STATE["energetic"], "energetic")
        if da > 0.6:
            return (self._PROACTIVE_PREFIXES_BY_STATE["curious"], "curious")
        if cort > 0.65:
            return (self._PROACTIVE_PREFIXES_BY_STATE["anxious"], "anxious")
        if cort < 0.35 and ot >= 0.30:
            return (self._PROACTIVE_PREFIXES_BY_STATE["calm"], "calm")
        return (None, None)

    def _engine_has_cause_tracker(self) -> bool:
        """engine에 emotion_cause 부착 여부."""
        engine = self._get_engine()
        return engine is not None and hasattr(engine, "emotion_cause") and engine.emotion_cause is not None

    def _get_engine(self):
        """planner에서 engine 접근 (main.py에서 _engine_ref 부착됨)."""
        return getattr(self, "_engine_ref", None)

    def _build_cause_clause(self, emotion: str) -> str:
        """
        감정 이유 절 만들기 (라운드 39a).
        engine.emotion_cause에서 cause 가져와 자연어로.
        """
        engine = self._get_engine()
        if engine is None or not hasattr(engine, "emotion_cause"):
            return ""
        tracker = engine.emotion_cause
        if tracker is None:
            return ""
        
        cause = tracker.get_relevant_cause(emotion)
        if cause is None:
            return ""
        
        clause = tracker.speech_clause_for_cause(cause, emotion)
        if clause:
            tracker.causes_used_in_speech += 1
        return clause

    def _build_proactive(self, meaning: Meaning, cat: str, mode: str) -> ResponsePlan:
        plan = ResponsePlan(intent="proactive")

        # 라운드 48: 어텐션 focus 활용 (cat 대체)
        # 호출 카테고리 hash로 rotation — 매번 다른 top entity 가능
        engine_for_attention = self._get_engine()
        if engine_for_attention is not None and hasattr(engine_for_attention, "attention"):
            try:
                # rotation = cat hash로 결정론 (매번 다른 시드 가능)
                rotation = sum(ord(c) for c in (cat or "")) % 5
                focus = engine_for_attention.attention.get_current_focus(rotation=rotation)
                if focus:
                    cat = focus
            except Exception:
                pass

        # VSA 합성 키 (+, @) 정리 — 사람한테 보일 부분
        display_cat = cat.split("@")[0].split("+")[0] if cat else cat
        
        # 라운드 70: GPT 짚은 핵심 — 자율발화 통합
        # self/user-ref 차단 + 짧은 자연 발화
        SELF_USER_REFS = {"민석", "EVE", "이브", "나", "너", "디지털"}
        if not display_cat or display_cat in SELF_USER_REFS:
            plan.core_message = ""
            plan.sub_points = []
            plan.skip_finalize = True
            return plan
        
        # 깨진 어절 차단 (라운드 69 원칙)
        from utils.korean_endings import _is_safe_input
        if not _is_safe_input(display_cat):
            plan.core_message = ""
            plan.sub_points = []
            plan.skip_finalize = True
            return plan
        
        # display_cat 기반 짧은 발화 — 결정론
        seed = sum(ord(c) for c in display_cat)
        thought_pool = [
            f"{display_cat} 생각나네.",
            f"갑자기 {display_cat} 떠올라.",
            f"음, {display_cat}.",
        ]
        plan.core_message = thought_pool[seed % len(thought_pool)]
        plan.sub_points = []
        plan.skip_finalize = True
        
        # 라운드 70: 라운드 39a cause clause 보존 (외로움 등 호르몬 시)
        # 강한 감정 있으면 prefix로 cause clause 붙임
        _, current_emotion = self._select_state_pool_with_emotion()
        if current_emotion and self._engine_has_cause_tracker():
            try:
                cause_clause = self._build_cause_clause(current_emotion)
                if cause_clause:
                    plan.sub_points.insert(0, cause_clause)
            except Exception:
                pass
        
        return plan
        
        # 아래 코드는 라운드 70 *우회* (위에서 return). 보존만 (참조용).
        # === 이전 thought_chain 로직 (라운드 70에서 비활성) ===

        # 호르몬 상태 우선 — 인간형 발화 (Damasio Somatic Marker)
        state_pool, current_emotion = self._select_state_pool_with_emotion()
        if state_pool is not None:
            pool = state_pool
        else:
            pool = self._PROACTIVE_PREFIXES_BY_MODE.get(
                mode, self._PROACTIVE_PREFIXES_BY_MODE["mind_wandering"]
            )
            current_emotion = None
        
        # 결정론 — 카테고리 hash (같은 cat = 같은 패턴)
        idx = sum(ord(c) for c in display_cat) % len(pool)
        prefix = pool[idx]
        plan.core_message = prefix.format(cat=display_cat)
        
        # 라운드 39b: 사고 체인 — 항상 시도, 길이 충분할 때만 사용
        # 사고 흐름이 자연스러우면 chain, 아니면 일반 prefix
        engine = self._get_engine()
        chain_phrase_used = False
        if engine is not None and hasattr(engine, "thought_chain") and engine.thought_chain is not None:
            try:
                chain = engine.thought_chain.generate_chain(display_cat, length=3)
                # 길이 2 이상 + 모든 노드가 의미 있는 카테고리일 때만 사용
                if len(chain) >= 2:
                    chain_phrase = engine.thought_chain.chain_to_phrase(chain)
                    if chain_phrase:
                        # 결정론적 선택 — 카테고리 hash로 chain vs prefix 결정
                        # %2 == 0 면 chain, == 1 면 prefix (대략 절반씩)
                        if sum(ord(c) for c in display_cat) % 2 == 0:
                            plan.core_message = chain_phrase
                            chain_phrase_used = True
            except Exception:
                pass
        
        # 라운드 39a + GPT 피드백: cause clause (chain 위에 prepend/append 가능)
        # - 결정론: cause 사용 카운터 % 3 == 0일 때만 (3번에 1번)
        # 단, 강한 사건 (intensity 0.5+) 일 때는 무조건
        if current_emotion and self._engine_has_cause_tracker():
            engine_c = self._get_engine()
            tracker = engine_c.emotion_cause if engine_c else None
            if tracker is not None:
                cause = tracker.get_relevant_cause(current_emotion)
                if cause is not None:
                    import time as _time
                    current_int = cause.current_intensity(_time.time())
                    use_cause = (current_int > 0.5) or (tracker.causes_used_in_speech % 3 == 0)
                    if use_cause:
                        clause = tracker.speech_clause_for_cause(cause, current_emotion)
                        if clause:
                            tracker.causes_used_in_speech += 1
                            if sum(ord(c) for c in display_cat) % 2 == 0:
                                plan.core_message = f"{clause} {plan.core_message}"
                            else:
                                plan.core_message = f"{plan.core_message} {clause}"
                    else:
                        tracker.causes_used_in_speech += 1

        # 회상 sub_point — memory adapter 있으면
        if self.memory_adapter is not None:
            recalls = self.memory_adapter.recall_phrases(meaning, max_n=1)
            plan.sub_points = recalls
        else:
            plan.sub_points = []

        plan.emotional_tone = "warm"

        # 호르몬 변조
        if self.hormone_adapter is not None:
            plan.emotional_tone = self.hormone_adapter.recommend_tone(None)
            budget = self.hormone_adapter.sub_point_budget()
            if len(plan.sub_points) > budget:
                plan.sub_points = plan.sub_points[:budget]
            from language.generator import STAGE1_BY_TONE
            base = STAGE1_BY_TONE.get(plan.emotional_tone, STAGE1_BY_TONE["neutral"])
            plan.stage1_hint = self.hormone_adapter.stage1_word(base)

        return plan
