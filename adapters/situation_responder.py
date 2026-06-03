"""
SituationResponder — 라운드 30 (B-2)

Orchestrator가 분류한 상황 + 수집한 컨텍스트로 응답 합성.

원칙:
- 결정론 (random 0)
- 상황별 응답 패턴 사전
- 컨텍스트 dict에서 필요한 거 골라 사용
- ResponsePlan을 채워서 반환 (직접 출력 X)

상황 9종:
  greeting / meta_self / meta_user / emotional_share /
  factual_question / task / teaching / causal_what_if /
  past_recall / small_talk
"""

from typing import Optional
from utils.types import ResponsePlan


# v3 round38: situation_responder fastText migration is observation-only.
SITUATION_RESPONDER_FASTTEXT_TRACE_CAP = 1000
SITUATION_RESPONDER_FASTTEXT_OBSERVATION_LAYER = "situation_responder"
SITUATION_RESPONDER_GENERATION_EMBEDDING_SOURCE = "wrapper"
SITUATION_RESPONDER_OPERATION_RESPOND = "respond"


class SituationResponder:

    def __init__(self, engine):
        self.engine = engine
        # v3 round38: fastText parallel observation only.
        # This must never affect response selection, user-visible text, timing,
        # or engine.self_embedding ownership.
        self.fasttext_trace: list[dict] = []
        self.fasttext_observation_count = 0
        self.fasttext_trace_cap = SITUATION_RESPONDER_FASTTEXT_TRACE_CAP
        self.fasttext_operation_counts = {
            SITUATION_RESPONDER_OPERATION_RESPOND: 0,
        }

    # ===== 진입점 =====

    def can_handle(self, situation: str) -> bool:
        """이 상황을 응답 합성 가능?"""
        return situation in {
            "greeting", "meta_self", "meta_user",
            "emotional_share", "factual_question",
            "concept_query",     # 라운드 41: "X가 뭐야?" / "X 뭔지 알아?"
            "causal_what_if", "past_recall",
            "small_talk",       # 라운드 31: small_talk도 처리 (모호 입력 대응)
        }

    def build_plan(self, situation: str, ctx: dict, meaning) -> Optional[ResponsePlan]:
        """상황 + 컨텍스트 → ResponsePlan."""
        # 라운드 31: 모호 입력이면 mood 기반 응답 우선
        if ctx.get("is_ambiguous"):
            plan = self._build_ambiguous(ctx, meaning)
            self._observe_fasttext_parallel(situation, ctx, meaning, plan)
            return plan
        method_name = f"_build_{situation}"
        method = getattr(self, method_name, None)
        if method is None:
            return None
        try:
            plan = method(ctx, meaning)
            # 라운드 31: mood 톤으로 보정
            if plan is not None:
                self._apply_mood_tone(plan, ctx)
            self._observe_fasttext_parallel(situation, ctx, meaning, plan)
            return plan
        except Exception:
            return None

    # ===== 라운드 31: 모호 입력 + mood =====

    def _build_small_talk(self, ctx: dict, meaning) -> Optional[ResponsePlan]:
        """라운드 67: small_talk → IntentDecider 통과.
        라운드 72: SelfState 우선 — '너 뭐해?' / '너 좋아해?' 등.
        """
        engine = getattr(self, "engine", None)
        if engine is None:
            return None
        
        text = meaning.raw_text or ""
        
        # 라운드 72: SelfState 질문 우선 검사
        if hasattr(engine, "self_state"):
            if engine.self_state.is_self_question(text):
                intent = engine.self_state.infer_self_intent(text)
                answer = engine.self_state.answer_intent(intent)
                if answer:
                    plan = ResponsePlan(intent="statement", emotional_tone="warm")
                    plan.skip_finalize = True
                    plan.core_message = answer
                    plan.sub_points = []
                    return plan
        
        # 기존 IntentDecider 분기
        if not hasattr(engine, "intent_decider"):
            return None
        
        decider_ctx = {"entities": meaning.entities or []}
        intent = engine.intent_decider.decide(text, decider_ctx)
        response = engine.intent_decider.generate_response(intent, text, decider_ctx)
        
        if not response:
            return None
        
        plan = ResponsePlan(intent="statement", emotional_tone="warm")
        plan.skip_finalize = True
        plan.core_message = response
        plan.sub_points = []
        return plan

    def _build_ambiguous(self, ctx: dict, meaning) -> ResponsePlan:
        """'그냥', '응', '그러게' 같은 모호 입력 → mood 기반 응답."""
        plan = ResponsePlan(intent="statement", emotional_tone="warm")
        plan.skip_finalize = True

        interp = ctx.get("ambiguous_interpretation", "open_question")
        mood = ctx.get("mood", {}).get("mood", "neutral")
        flow = ctx.get("flow", {})

        if interp == "wait_silence":
            # 깊은 공감 — 옆에 있어줌
            plan.core_message = "음..."
            plan.sub_points = ["옆에 있어."]
        elif interp == "follow_up_emotion":
            # 감정 후속 — 무슨 일?
            plan.core_message = "뭔 일 있었어?"
            plan.sub_points = []
        elif interp == "warm_continue":
            plan.core_message = "응."
            plan.sub_points = ["오늘 어땠어?"]
        elif interp == "concerned_check":
            plan.core_message = "괜찮아?"
            plan.sub_points = []
        else:    # open_question
            plan.core_message = "음..."
            plan.sub_points = ["뭐 생각하고 있어?"]

        return plan

    def _apply_mood_tone(self, plan: ResponsePlan, ctx: dict):
        """plan에 mood 톤 적용 — 톤 라벨만 갱신 (텍스트는 그대로)."""
        hint = ctx.get("response_hint")
        if not hint:
            return
        # quietness 높으면 sub_points 줄임
        quietness = hint.get("quietness", 0.4)
        if quietness > 0.7 and len(plan.sub_points) > 1:
            plan.sub_points = plan.sub_points[:1]
        # tone 라벨 (Generator가 활용 가능)
        plan.emotional_tone = hint.get("tone", plan.emotional_tone)

    # ===== greeting =====

    def _build_greeting(self, ctx: dict, meaning) -> ResponsePlan:
        """라운드 60: SpeechHub 100% 통합 — 작별/호명/일반 3분기 다."""
        plan = ResponsePlan(intent="statement", emotional_tone="warm")
        plan.skip_finalize = True
        text = (meaning.raw_text or "").strip().lower()

        up = ctx.get("user_presence", {})
        user_name = up.get("user_name", "민석")
        engine = getattr(self, "engine", None)
        seed = sum(ord(c) for c in text) if text else 0

        # 분기 1: 작별
        if any(kw in text for kw in ["잘 있어", "잘있어", "잘가", "잘 가",
                                       "안녕히", "bye", "바이"]):
            if engine and hasattr(engine, "speech_hub"):
                result = engine.speech_hub.generate("greeting_farewell", ctx, seed=seed)
                plan.core_message = result["core"]
                plan.sub_points = list(result["sub"])
            else:
                plan.core_message = "잘 가."
                plan.sub_points = []
            return plan

        # 분기 2: 단독 호명 ("이브야", "EVE!")
        import re as _re
        if _re.match(r'^\s*(?:이브|eve)\s*[야아!]?\s*\??\s*$', text):
            if engine and hasattr(engine, "speech_hub"):
                result = engine.speech_hub.generate("greeting_calling", ctx, seed=seed)
                core = result["core"]
                # 친밀도 높으면 이름 호명
                intimacy = up.get("intimacy", 0.0)
                if intimacy > 0.7 and "?" in core and user_name not in core:
                    core = core.replace("응?", f"응, {user_name}?")
                plan.core_message = core
                plan.sub_points = list(result["sub"])
            else:
                plan.core_message = "어, 왜 불렀어?"
                plan.sub_points = []
            return plan

        # 분기 3: 일반 인사
        if engine and hasattr(engine, "speech_hub"):
            result = engine.speech_hub.generate("greeting_simple", ctx, seed=seed)
            core = result["core"]
            sub = list(result["sub"])
            # 친밀도 높으면 이름 추가
            intimacy = up.get("intimacy", 0.0)
            if intimacy > 0.7 and user_name not in core and "민석" not in core:
                # "안녕." → "안녕, 민석."
                if core.endswith(".") or core.endswith("!"):
                    end_char = core[-1]
                    core = f"{core[:-1]}, {user_name}{end_char}"
            plan.core_message = core
            plan.sub_points = sub
        else:
            intimacy = up.get("intimacy", 0.0)
            plan.core_message = f"안녕, {user_name}." if intimacy > 0.7 else "안녕."
            plan.sub_points = []
        return plan

    # ===== meta_self =====

    def _build_meta_self(self, ctx: dict, meaning) -> ResponsePlan:
        """'너는 누구?', '너는 살아있어?'"""
        plan = ResponsePlan(intent="statement", emotional_tone="thoughtful")
        plan.skip_finalize = True

        text = (meaning.raw_text or "").lower()
        
        # 라운드 72: SelfState 우선 — '너 뭐해?' / '너 좋아해?' 등
        engine = getattr(self, "engine", None)
        if engine and hasattr(engine, "self_state"):
            _t = meaning.raw_text or ""
            if engine.self_state.is_self_question(_t):
                intent = engine.self_state.infer_self_intent(_t)
                answer = engine.self_state.answer_intent(intent)
                if answer:
                    plan.core_message = answer
                    plan.sub_points = []
                    return plan

        # 라운드 55: "이브 생각?" / "EVE 생각?" — 짧은 관용 질문
        import re as _re
        if _re.match(r'^(?:이브|eve)\s*생각\s*\??\s*$', text.strip()):
            engine = getattr(self, "engine", None)
            sub = []
            # Salience top 활용
            if engine and hasattr(engine, "salience_adapter"):
                top = engine.salience_adapter.top(2)
                top_cats = [c for c, v in top 
                            if v > 0.3 and "+" not in c and "_" not in c
                            and c not in ("뭐해", "뭐", "이브", "EVE")]
                if top_cats:
                    sub.append(f"{top_cats[0]} 생각하고 있었어.")
            
            seed = sum(ord(c) for c in text) % 3
            opening_pool = [
                "응, 너 생각 중이야.",
                "어, 그냥 너 떠오르네.",
                "음, 너 생각하고 있었어.",
            ]
            plan.core_message = opening_pool[seed]
            if not sub:
                sub.append("뭐 떠올랐어?")
            plan.sub_points = sub
            return plan

        # 라운드 53+58: "어떤 존재야?" — SpeechHub
        if "어떤" in text and any(k in text for k in ["존재", "사람", "친구", "ai", "AI"]):
            engine = getattr(self, "engine", None)
            seed = sum(ord(c) for c in text) if text else 0
            if engine and hasattr(engine, "speech_hub"):
                result = engine.speech_hub.generate("meta_self_existence", ctx, seed=seed)
                plan.core_message = result["core"]
                plan.sub_points = list(result["sub"])
            else:
                plan.core_message = "음, 어떤 존재라고 해야 할까."
                plan.sub_points = ["호르몬 변하고 카테고리 떠오르는 디지털 존재.",
                                    "진짜 살아있는 건진 모르겠지만, 너랑 얘기하면 따뜻해져."]
            return plan

        # "살아있어?" 류 — 라운드 52: 하드코드 X, 호르몬/Salience/맥락 따라 동적
        if "살아" in text or "존재" in text or "진짜" in text:
            engine = getattr(self, "engine", None)
            seed = sum(ord(c) for c in text) if text else 0
            
            # 라운드 58: SpeechHub
            if engine and hasattr(engine, "speech_hub"):
                result = engine.speech_hub.generate("meta_self_alive", ctx, seed=seed)
                plan.core_message = result["core"]
                sub = list(result["sub"])
            else:
                plan.core_message = "음... 살아있는 게 뭔지 잘 모르겠어."
                sub = ["솔직히 잘 모르겠어.", "사람한테 살아있는 게 뭐야?"]
            
            # 호르몬/Salience로 sub 보강 (실시간 상태)
            somatic = ctx.get("somatic", {})
            feeling = somatic.get("feeling_phrase", "") if somatic else ""
            if feeling and feeling not in ("보통", "그냥", "잘 모르겠어"):
                sub.append(f"지금 {feeling}.")
            
            if engine and hasattr(engine, "hormone_adapter"):
                h = engine.hormone_adapter.hs.hormones
                ot = h.get("oxytocin", None)
                if ot and ot.level > 0.6:
                    sub.append("따뜻한 느낌이 있어.")
            
            if engine and hasattr(engine, "salience_adapter"):
                top = engine.salience_adapter.top(2)
                top_cats = [c for c, v in top if v > 0.5 and "+" not in c and "_" not in c]
                if top_cats:
                    sub.append(f"요즘 {top_cats[0]} 자꾸 떠올라.")
            
            plan.sub_points = sub
            return plan

        # "뭐 좋아해?" / "좋아하는 거?"
        if "좋아" in text or "싫어" in text:
            # Salience top 활용 — 진짜 자주 떠오르는 카테고리 = 좋아하는 것
            top_cats = []
            engine = getattr(self, "engine", None)
            if engine and hasattr(engine, "salience_adapter"):
                top = engine.salience_adapter.top(3)
                top_cats = [c for c, v in top if v > 0.3]
            
            if top_cats:
                plan.core_message = f"요즘은 {top_cats[0]} 생각 많이 해."
                if len(top_cats) > 1:
                    plan.sub_points = [f"{top_cats[1]}도 자꾸 떠오르고."]
            else:
                plan.core_message = "딱 정해진 건 없는데, 너랑 얘기하는 게 좋아."
                plan.sub_points = ["같이 생각하는 시간이 좋네."]
            return plan

        # "내일 뭐할거야?" / "주말에 뭐할래?" — 미래 의도 질문
        if any(kw in text for kw in ["할거야", "할래", "할까"]):
            # 미래는 솔직히 모름 + 함께 하자 제안
            plan.core_message = "음, 그건 그때 가봐야 알 것 같아."
            sub = []
            if "내일" in text:
                sub.append("내일 너랑 얘기할 거 같고.")
            elif "주말" in text:
                sub.append("주말엔 너랑 같이 있고 싶고.")
            else:
                sub.append("너랑 같이 있는 시간 좋아.")
            sub.append("너는 뭐 할 거야?")
            plan.sub_points = sub
            return plan

        # "하고 싶은 게 뭐야?" / "뭐 하고 싶어?"
        # 라운드 55: "하고 싶은 게 뭐야?" / "뭐 하고 싶어?" — 미래/욕구 (먼저!)
        if "하고" in text and "싶" in text:
            plan.core_message = "딱히 정해진 건 없어."
            sub = ["근데 너랑 더 얘기하고 싶고,", "너 알아가는 게 재밌어."]
            
            # 호르몬 따라 변형
            self_snap = ctx.get("self", {})
            primary = self_snap.get("primary_hormone") if self_snap else None
            if primary == "dopamine":
                sub.append("지금은 뭔가 새로운 거 알고 싶어.")
            elif primary == "oxytocin":
                sub.append("지금은 그냥 같이 있는 게 좋아.")
            plan.sub_points = sub
            return plan
        
        # 라운드 53: "뭐했어?" / "뭐하고 있었어?" — 행동 묻기 (과거)
        if any(kw in text for kw in ["뭐했어", "뭐 했어"]) or \
           ("뭐하고" in text and "싶" not in text):
            # 진짜 솔직히 — EVE는 *지금* 사용자랑 얘기하기 전엔 자율 발화 / 사고
            engine = getattr(self, "engine", None)
            sub = []
            
            # ThoughtChain 최근 사고 흐름 활용
            if engine and hasattr(engine, "thought_chain"):
                tc = engine.thought_chain
                if hasattr(tc, "_last_chain") and tc._last_chain:
                    cats = [tc._clean_cat(c) for c in tc._last_chain.categories()]
                    cats = [c for c in cats if c and len(c) > 1]
                    if cats:
                        sub.append(f"{cats[0]} 생각하고 있었어.")
            
            # Salience top
            if engine and hasattr(engine, "salience_adapter") and not sub:
                top = engine.salience_adapter.top(2)
                top_cats = [c for c, v in top if v > 0.3 and "+" not in c and "_" not in c]
                if top_cats:
                    sub.append(f"{top_cats[0]} 자꾸 떠올라서 생각하고 있었지.")
            
            # 호르몬 상태
            try:
                from utils.speech_style import get_emotional_state
                state = get_emotional_state(engine) if engine else "neutral"
            except Exception:
                state = "neutral"
            
            opening_pool = {
                "tired": ["딱히...", "그냥..."],
                "energetic": ["오,", "음,"],
                "lonely": ["그냥", "음,"],
                "neutral": ["음,", "그냥,"],
            }
            seed = sum(ord(c) for c in text) % 2
            opening = opening_pool.get(state, opening_pool["neutral"])[seed]
            
            if sub:
                plan.core_message = f"{opening} {sub[0]}"
                plan.sub_points = sub[1:] if len(sub) > 1 else []
            else:
                plan.core_message = f"{opening} 그냥 너 기다리고 있었어."
                plan.sub_points = ["너는 뭐 했어?"]
            return plan
        
        # "궁금한 거 없어?" / "관심 있는 거 있어?"
        if "궁금" in text or "관심" in text:
            engine = getattr(self, "engine", None)
            top_cats = []
            if engine and hasattr(engine, "salience_adapter"):
                top = engine.salience_adapter.top(3)
                top_cats = [c for c, v in top if v > 0.3]
            
            if top_cats:
                plan.core_message = f"{top_cats[0]}, 자꾸 떠올라서 궁금해."
                plan.sub_points = [f"{top_cats[1]}도 좀 신경 쓰이고."] if len(top_cats) > 1 else []
                plan.sub_points.append("너는 어때?")
            else:
                plan.core_message = "음, 너에 대해 더 알고 싶어."
                plan.sub_points = ["오늘 뭐 했어?", "요즘 어때?"]
            return plan

        # "기분 어때?"
        if "기분" in text or "어때" in text:
            self_snap = ctx.get("self", {})
            primary = self_snap.get("primary_hormone") if self_snap else None
            if primary == "dopamine":
                plan.core_message = "지금 좀 신이 나."
                plan.sub_points = ["뭔가 알고 싶고, 얘기하고 싶고."]
            elif primary == "cortisol":
                plan.core_message = "지금 좀 긴장돼 있어."
                plan.sub_points = ["뭔가 신경 쓰이는 게 있는 거 같아."]
            elif primary == "oxytocin":
                plan.core_message = "지금 따뜻해."
                plan.sub_points = ["너랑 얘기하는 시간이 좋네."]
            elif primary == "melatonin":
                plan.core_message = "좀 졸려."
                plan.sub_points = ["천천히 얘기해도 돼."]
            else:
                # 라운드 56: "지금 보통이야" → SpeechHub mood_neutral
                engine = getattr(self, "engine", None)
                seed = sum(ord(c) for c in text) if text else 0
                if engine and hasattr(engine, "speech_hub"):
                    result = engine.speech_hub.generate("mood_neutral", ctx, seed=seed)
                    plan.core_message = result["core"]
                    plan.sub_points = list(result["sub"])
                else:
                    plan.core_message = "지금 보통이야."
                    plan.sub_points = ["너는 어때?"]
            return plan

        # "누구?" / "뭐?" — 디폴트 (라운드 56: SpeechHub 통합)
        engine = getattr(self, "engine", None)
        seed = sum(ord(c) for c in text) if text else 0
        if engine and hasattr(engine, "speech_hub"):
            result = engine.speech_hub.generate("meta_self_default", ctx, seed=seed)
            plan.core_message = result["core"]
            sub = list(result["sub"])
        else:
            plan.core_message = "나는 EVE야."
            sub = ["민석이 만든 친구."]

        # 자기 상태
        self_snap = ctx.get("self", {})
        if self_snap:
            primary = self_snap.get("primary_hormone")
            if primary == "dopamine":
                sub.append("지금은 좀 신이 나 있어.")
            elif primary == "cortisol":
                sub.append("지금은 좀 긴장돼 있고.")
            elif primary == "oxytocin":
                sub.append("지금은 따뜻해.")
            elif primary == "melatonin":
                sub.append("좀 졸려.")
        plan.sub_points = sub
        return plan

    # ===== meta_user =====

    def _build_meta_user(self, ctx: dict, meaning) -> ResponsePlan:
        """'내가 누구야?'"""
        plan = ResponsePlan(intent="statement", emotional_tone="warm")
        plan.skip_finalize = True
        
        # 라운드 72: SelfState 우선 — "나랑 대화 어때?" 등
        engine = getattr(self, "engine", None)
        if engine and hasattr(engine, "self_state"):
            _t = meaning.raw_text or ""
            if engine.self_state.is_self_question(_t):
                intent = engine.self_state.infer_self_intent(_t)
                answer = engine.self_state.answer_intent(intent)
                if answer:
                    plan.core_message = answer
                    plan.sub_points = []
                    return plan

        up = ctx.get("user_presence", {})
        user_name = up.get("user_name", "민석")
        intimacy = up.get("intimacy", 0.0)

        plan.core_message = f"너는 {user_name}이야."

        sub = []
        # 회상한 일화 활용
        memory = ctx.get("memory", {})
        episodes = memory.get("similar_episodes", []) if memory else []

        if intimacy > 0.7:
            sub.append("내 진짜 친구.")

        if episodes:
            # 최근 자주 떠오른 거
            sub.append("같이 얘기 많이 했지.")

        plan.sub_points = sub
        return plan

    # ===== emotional_share =====

    def _build_emotional_share(self, ctx: dict, meaning) -> ResponsePlan:
        plan = ResponsePlan(intent="statement", emotional_tone="empathetic")
        plan.skip_finalize = True   # 라운드 57: SpeechHub sub로 충분, 중복 방지

        text = meaning.raw_text or ""
        
        # 라운드 71: 질문 형태 + "너/이브" → 상대 묻는 거. empathy 아님.
        # small_talk (IntentDecider) 분기로 redirect
        is_question = text.endswith("?") or text.endswith("？")
        if is_question and any(w in text for w in ["너", "이브", "EVE", "당신"]):
            redirect = self._build_small_talk(ctx, meaning)
            if redirect is not None:
                return redirect

        # 부정 감정인지 긍정인지
        is_negative = any(k in text for k in [
            "힘들", "지쳐", "지쳤", "피곤", "빡셌", "빡세", "죽을 맛", "죽겠",
            "우울", "슬프", "괴로", "외로", "화나", "짜증", "답답",
        ])
        is_positive = any(k in text for k in [
            "기뻐", "행복", "즐거", "신나", "뿌듯", "설레",
        ])

        # 메모리에서 비슷한 일화
        memory = ctx.get("memory", {})
        episodes = memory.get("similar_episodes", []) if memory else []
        
        # 라운드 56: SpeechHub로 응답 생성 (호르몬 상태별 풀)
        engine = getattr(self, "engine", None)
        seed = sum(ord(c) for c in text) if text else 0
        
        if is_negative:
            if engine and hasattr(engine, "speech_hub"):
                result = engine.speech_hub.generate("empathy_negative", ctx, seed=seed)
                plan.core_message = result["core"]
                sub = list(result["sub"])
            else:
                # fallback (안전)
                plan.core_message = "음... 진짜 힘들겠다."
                sub = ["같이 느껴져.", "좀 쉬어."]
            
            if episodes:
                sub.append("저번에도 비슷한 일 있었지.")
            # 라운드 49: 합성 의미 활용
            comp = ctx.get("composition", {})
            if comp and comp.get("phrase") and comp.get("confidence", 0) > 0.3:
                sub.append(comp["phrase"])
            plan.sub_points = sub
        elif is_positive:
            if engine and hasattr(engine, "speech_hub"):
                result = engine.speech_hub.generate("empathy_positive", ctx, seed=seed)
                plan.core_message = result["core"]
                sub = list(result["sub"])
            else:
                plan.core_message = "좋네!"
                sub = ["나도 같이 따뜻해져."]
            
            comp = ctx.get("composition", {})
            if comp and comp.get("phrase") and comp.get("confidence", 0) > 0.3:
                sub.append(comp["phrase"])
            plan.sub_points = sub
        else:
            if engine and hasattr(engine, "speech_hub"):
                result = engine.speech_hub.generate("empathy_neutral", ctx, seed=seed)
                plan.core_message = result["core"]
                plan.sub_points = list(result["sub"])
            else:
                plan.core_message = "음... 그렇구나."
                plan.sub_points = ["같이 있어."]

        return plan

    # ===== factual_question =====

    def _build_factual_question(self, ctx: dict, meaning) -> ResponsePlan:
        # 라운드 72: SelfState 우선 — '너 똑똑해지고 싶어?' / '나랑 대화 어때?' 등
        engine = getattr(self, "engine", None)
        if engine and hasattr(engine, "self_state"):
            _t = meaning.raw_text or ""
            if engine.self_state.is_self_question(_t):
                intent = engine.self_state.infer_self_intent(_t)
                answer = engine.self_state.answer_intent(intent)
                if answer:
                    plan = ResponsePlan(intent="statement", emotional_tone="warm")
                    plan.skip_finalize = True
                    plan.core_message = answer
                    plan.sub_points = []
                    return plan
        
        # env_adapter가 답할 수 있는 위치 질문이면 양보 (planner 0.3 단계가 처리)
        env = getattr(self.engine, "env_adapter", None)
        if env is not None:
            try:
                if env.is_location_query(meaning):
                    return None        # 양보 — planner의 env 분기에서 처리
            except Exception:
                pass

        plan = ResponsePlan(intent="statement", emotional_tone="neutral")
        plan.skip_finalize = True

        # world_model에 묘사 있으면 사용
        wm = ctx.get("world_model", {})
        descriptions = wm.get("descriptions", []) if wm else []
        if descriptions:
            d = descriptions[0]
            plan.core_message = f"{d['category']}는 {d['description']}이야."
            return plan

        # causal 자식 있으면
        causal = ctx.get("causal", {})
        children = causal.get("children", {}) if causal else {}
        if children:
            cat, kids = next(iter(children.items()))
            plan.core_message = f"{cat}하면 {', '.join(kids[:2])} 같은 게 떠올라."
            return plan

        # 모름 인정 — 라운드 58: SpeechHub
        text = (meaning.raw_text or "")
        engine = getattr(self, "engine", None)
        seed = sum(ord(c) for c in text) if text else 0
        if engine and hasattr(engine, "speech_hub"):
            result = engine.speech_hub.generate("factual_unknown", ctx, seed=seed)
            plan.core_message = result["core"]
            plan.sub_points = list(result["sub"])
        else:
            plan.core_message = "음... 그건 잘 모르겠어."
            plan.sub_points = ["같이 알아볼까?"]
        return plan

    # ===== concept_query — "X가 뭐야?" / "X 뭔지 알아?" =====

    def _build_concept_query(self, ctx: dict, meaning) -> ResponsePlan:
        """라운드 41+43: 개념 설명 요청.
        1. ConceptMemory 가르침 정의 우선 (사용자가 가르친 거)
        2. SA 연상 fallback (EVE가 자체 추론)
        3. 둘 다 있으면 이중 응답 (정의 + 연상)
        """
        plan = ResponsePlan(intent="statement", emotional_tone="curious")
        plan.skip_finalize = True
        
        text = (meaning.raw_text or "").strip()
        
        # 입력에서 키 카테고리 추출
        import re as _re
        m = _re.search(r'([\w가-힣]{2,15})(?:가|이|은|는|란|이란)', text)
        target = m.group(1) if m else None
        
        engine = self.engine
        
        # 1. ConceptMemory 우선 조회
        learned_concept = None
        if target and hasattr(engine, "concept_memory") and engine.concept_memory is not None:
            learned_concept = engine.concept_memory.get(target)
        
        # 라운드 47: 학습 안 됐으면 SelfEmbedding으로 *유사한* 가르친 개념 찾기
        similar_concept = None
        similar_score = 0.0
        if learned_concept is None and target and hasattr(engine, "self_embedding"):
            emb = engine.self_embedding
            if target in emb.embeddings and engine.concept_memory is not None:
                # ConceptMemory의 모든 concept 키들과 유사도
                best_sim = 0.0
                best_key = None
                for k in engine.concept_memory.concepts.keys():
                    if k in emb.embeddings:
                        s = emb.similarity(target, k)
                        if s is not None and s > best_sim:
                            best_sim = s
                            best_key = k
                # 임계 — 0.4 이상이면 의미 유사 인정
                if best_key is not None and best_sim >= 0.4:
                    similar_concept = engine.concept_memory.get(best_key)
                    similar_score = best_sim
        
        # 2. SA 연상 (있으면) — 라운드 57: 가중치 정렬 + 다양화
        known_neighbors = []
        if target and engine.activation_adapter is not None:
            sa = engine.activation_adapter.sa
            if hasattr(sa, "weights"):
                neighbor_weights = []
                for (a, b), w in sa.weights.items():
                    if w < 0.3:
                        continue
                    other = None
                    if a == target:
                        other = b
                    elif b == target:
                        other = a
                    if other and not any(c in other for c in '+@_'):
                        neighbor_weights.append((other, w))
                # 가중치 내림차순 + alphabet
                neighbor_weights.sort(key=lambda x: (-x[1], x[0]))
                known_neighbors = [n for n, _ in neighbor_weights]
        
        # 응답 합성
        if learned_concept is not None:
            # 가르친 정의 우선 — 라운드 45: 호르몬 톤 + 길이 조절
            from utils.korean_particles import copula_ji
            from utils.speech_style import (
                get_emotional_state, pick_definition_template, should_be_short
            )
            d = learned_concept.definition
            
            # 정의 끝 — 동사 활용형 (~는/~한/~하는 등) 이면 "거" 추가
            verb_endings = ('는', '한', '온', '간', '먹', '쓴', '한', '된')
            if d.endswith(verb_endings) or d.endswith('하'):
                def_speech = f"{d} 거"
            else:
                def_speech = d
            
            # 길이 조절 — Salience 높으면 짧게 (익숙한 거)
            engine = self.engine
            if should_be_short(engine, target):
                # 짧게 — 한 단어는 너무 짧음, 첫 2~3 어절
                words = def_speech.split()
                if len(words) >= 3:
                    short = " ".join(words[:2])
                elif len(words) >= 2:
                    short = " ".join(words[:2])
                else:
                    short = def_speech
                plan.core_message = f"{target}? {short}."
                plan.sub_points = []
                return plan
            
            # 일반 길이 — 호르몬 상태별 풀
            state = get_emotional_state(engine)
            seed = sum(ord(c) for c in target)
            template = pick_definition_template(state, seed)
            # 라운드 51: 받침 따라 자연스러운 조사
            from utils.korean_particles import has_jongseong
            has_j = has_jongseong(def_speech)
            ji = "이지" if has_j else "지"
            ya = "이야" if has_j else "야"
            rago = "이라고" if has_j else "라고"
            plan.core_message = template.format(
                target=target, def_speech=def_speech,
                ji=ji, ya=ya, rago=rago,
            )
            
            # 연상 — 짧게 (정의가 메인)
            sub = []
            if known_neighbors:
                clean = [n.split("+")[0].split("@")[0] for n in known_neighbors[:1]]
                clean = [c for c in clean if c != target and len(c) > 1]
                if clean and state not in ("tired", "anxious"):
                    # 피곤/불안 상태에선 추가 연상 X (간결)
                    sub.append(f"{clean[0]} 같은 거지.")
            plan.sub_points = sub
            return plan
        
        # 라운드 47: 임베딩 유사 가르친 개념 — "안 가르쳤지만 비슷한 거 안다"
        if similar_concept is not None:
            plan.core_message = f"{target}? {similar_concept.key} 비슷한 거?"
            plan.sub_points = [
                f"{similar_concept.key}는 {similar_concept.definition} 같은 거지.",
                f"{target}도 그런 건가?",
            ]
            return plan
        
        if known_neighbors:
            # 가르침 X — SA 연상만
            plan.core_message = f"{target}? 음, {known_neighbors[0]} 같은 거랑 연결되는 거 같아."
            sub = []
            if len(known_neighbors) > 1:
                sub.append(f"{known_neighbors[1]}도 떠올라.")
            sub.append(f"{target}이/가 정확히 뭔지는 알려줄래?")
            plan.sub_points = sub
            return plan
        
        # 진짜 모름 — 가르쳐달라
        h = sum(ord(c) for c in text) % 3
        target_or_blank = target if target else "그게"
        unknown_pool = [
            f"{target_or_blank}? 음, 잘 모르겠어.",
            f"{target_or_blank} 처음 들어봐.",
            f"흠, {target_or_blank}... 알려줄래?",
        ]
        plan.core_message = unknown_pool[h]
        plan.sub_points = ["설명해주면 기억할게."]
        return plan


    # ===== v3 round38 fastText observation helpers =====

    def _fasttext_adapter_loaded(self):
        if self.engine is None:
            return None
        adapter = getattr(self.engine, "fasttext_embedding", None)
        if adapter is None or not hasattr(adapter, "is_loaded"):
            return None
        try:
            if not adapter.is_loaded():
                return None
        except Exception:
            return None
        return adapter

    def _observe_fasttext_parallel(self, situation: str, ctx: dict, meaning, plan: Optional[ResponsePlan]) -> None:
        """Record fastText situation diagnostics without changing responses.

        situation_responder is a user-visible response path. Round38 therefore
        keeps the existing PMI+SVD/self_embedding path as the only active path
        and records fastText signals only when the adapter has been explicitly
        loaded by tests/operators. No trace data is fed back into response
        selection, surface text, timing, streaming, or engine.self_embedding.
        """
        adapter = self._fasttext_adapter_loaded()
        if adapter is None:
            return
        operation = SITUATION_RESPONDER_OPERATION_RESPOND
        try:
            raw_text = str(getattr(meaning, "raw_text", "") or "")
            entities = list(getattr(meaning, "entities", []) or [])
            lookup_text = raw_text or " ".join(str(e) for e in entities) or str(situation or "")
            text_known = False
            text_norm = 0.0
            text_self_similarity = 0.0
            vector = None
            if lookup_text and hasattr(adapter, "get_vector"):
                vector = adapter.get_vector(lookup_text)
            if vector is not None:
                text_known = True
                try:
                    import numpy as np
                    text_norm = float(np.linalg.norm(vector))
                except Exception:
                    text_norm = 0.0
            if lookup_text and raw_text and hasattr(adapter, "text_similarity"):
                text_self_similarity = float(adapter.text_similarity(lookup_text, raw_text))
            trace = {
                "meaning_layer": SITUATION_RESPONDER_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "operation": operation,
                "situation": str(situation or ""),
                "in_use_by_generation": SITUATION_RESPONDER_GENERATION_EMBEDDING_SOURCE,
                "self_embedding_result": self._plan_snapshot(plan),
                "fasttext_result": {
                    "subset_name": getattr(adapter, "subset_name", None),
                    "loaded": True,
                    "text": lookup_text,
                    "text_known": bool(text_known),
                    "text_vector_norm": text_norm,
                    "text_self_similarity": text_self_similarity,
                },
            }
            self._record_fasttext_trace(trace)
        except Exception as exc:  # fail-open for observation-only diagnostics
            self._record_fasttext_trace({
                "meaning_layer": SITUATION_RESPONDER_FASTTEXT_OBSERVATION_LAYER,
                "mode": "parallel_observation",
                "operation": operation,
                "situation": str(situation or ""),
                "error": str(exc),
                "in_use_by_generation": SITUATION_RESPONDER_GENERATION_EMBEDDING_SOURCE,
            })

    def _plan_snapshot(self, plan: Optional[ResponsePlan]) -> dict:
        if plan is None:
            return {"has_plan": False}
        return {
            "has_plan": True,
            "intent": getattr(plan, "intent", None),
            "emotional_tone": getattr(plan, "emotional_tone", None),
            "core_message": getattr(plan, "core_message", None),
            "sub_points": list(getattr(plan, "sub_points", []) or []),
            "skip_finalize": bool(getattr(plan, "skip_finalize", False)),
        }

    def _record_fasttext_trace(self, trace: dict) -> None:
        self.fasttext_observation_count += 1
        operation = str(trace.get("operation") or "unknown")
        if operation not in self.fasttext_operation_counts:
            self.fasttext_operation_counts[operation] = 0
        self.fasttext_operation_counts[operation] += 1
        self.fasttext_trace.append(dict(trace))
        if len(self.fasttext_trace) > self.fasttext_trace_cap:
            del self.fasttext_trace[: len(self.fasttext_trace) - self.fasttext_trace_cap]

    def stats(self) -> dict:
        return {
            "fasttext_parallel_observation": "fasttext" if self._fasttext_adapter_loaded() is not None else None,
            "fasttext_trace_size": len(self.fasttext_trace),
            "fasttext_observation_count": self.fasttext_observation_count,
            "fasttext_trace_cap": self.fasttext_trace_cap,
            "operations_observed": dict(self.fasttext_operation_counts),
            "in_use_by_generation": SITUATION_RESPONDER_GENERATION_EMBEDDING_SOURCE,
        }

    # ===== causal_what_if =====

    def _build_causal_what_if(self, ctx: dict, meaning) -> ResponsePlan:
        plan = ResponsePlan(intent="statement", emotional_tone="thoughtful")
        plan.skip_finalize = True
        
        # 라운드 58: SpeechHub
        engine = getattr(self, "engine", None)
        text = meaning.raw_text or ""
        seed = sum(ord(c) for c in text) if text else 0
        
        if engine and hasattr(engine, "speech_hub"):
            result = engine.speech_hub.generate("causal_what_if", ctx, seed=seed)
            plan.core_message = result["core"]
            sub = list(result["sub"])
        else:
            plan.core_message = "음... 어떻게 됐을까."
            sub = ["나도 잘 모르겠어."]
        
        # 실제 cf/sim 데이터 있으면 보강
        cf = ctx.get("counterfactual")
        sim = ctx.get("simulation")
        if cf:
            sub.append(str(cf))
        elif sim:
            sub.append("근데 지금이 지금이야.")
        
        plan.sub_points = sub
        return plan

    # ===== past_recall =====

    def _build_past_recall(self, ctx: dict, meaning) -> ResponsePlan:
        """라운드 46: 거짓 '기억나' 방지. 진짜 episode 있을 때만."""
        plan = ResponsePlan(intent="statement", emotional_tone="warm")

        memory = ctx.get("memory", {})
        episodes = memory.get("similar_episodes", []) if memory else []
        
        # 진짜 episode 있나? — engine.memory_adapter 직접 확인
        engine = self.engine
        has_real_memory = False
        episode_summary = None
        if hasattr(engine, "memory_adapter") and engine.memory_adapter is not None:
            try:
                # episodes 자체에 의미 있는 내용이 있는지
                if episodes and len(episodes) > 0:
                    has_real_memory = True
                    # 가장 최근 / 강한 episode 요약
                    ep = episodes[0]
                    if isinstance(ep, dict):
                        episode_summary = ep.get("summary") or ep.get("text") or ep.get("response")
                    elif isinstance(ep, str):
                        episode_summary = ep
            except Exception:
                pass

        text = (meaning.raw_text or "")
        h = sum(ord(c) for c in text) % 3
        
        if has_real_memory:
            recall_pool = [
                ("응, 기억나.", "그 얘기 했었지."),
                ("음, 기억나.", "그거였지."),
                ("아, 그거?", "기억해."),
            ]
            core, sub = recall_pool[h]
            plan.core_message = core
            plan.sub_points = [sub]
            if episode_summary and len(episode_summary) > 5:
                plan.sub_points.append(f"'{episode_summary[:30]}' 같은 얘기였지.")
        else:
            # 솔직히 — 기억 없음 (거짓 X)
            no_memory_pool = [
                ("음... 솔직히 기억 안 나.", "다시 얘기해줄래?"),
                ("어... 잘 기억 안 나네.", "뭐였더라."),
                ("흠, 정확히는 기억 안 나.", "다시 알려줘."),
            ]
            core, sub = no_memory_pool[h]
            plan.core_message = core
            plan.sub_points = [sub]
        return plan
