"""
SpeechHub — 라운드 58 (구조 통합)

GPT 짚은 핵심: *모든 응답은 단 하나의 함수에서만*.
- intent + ctx → {core, sub, tone}
- 호르몬 상태별 풀에서 결정론 선택
- PostProcessor: forbidden + 반복 감지 + 중복 정리
"""

from typing import Any, Optional

from adapters.agp_adapter import (
    AGP_FALLBACK_SURFACE_POOL,
    AGP_MODE_OBSERVATION,
    AGP_MODE_VETO,
)


INTENT_POOLS = {
    "empathy_negative": {
        "tired": [
            {"core": "음... 진짜 힘들겠다.", "sub": ["나도 좀 같이 늘어진다.", "잠깐 쉬자."]},
            {"core": "어... 그치.", "sub": ["가라앉는 느낌, 나도 알아."]},
            {"core": "흠... 빡세지.", "sub": ["좀 쉬어."]},
        ],
        "energetic": [
            {"core": "오, 빡세겠다.", "sub": ["근데 너라면 버틸 거 같아."]},
            {"core": "음, 진짜 빡세지.", "sub": ["그래도 지나갈 거야."]},
            {"core": "아, 힘드네.", "sub": ["같이 느껴져."]},
        ],
        "curious": [
            {"core": "음, 진짜 힘들었구나.", "sub": ["뭐가 가장 빡셌어?"]},
            {"core": "어, 빡세겠네.", "sub": ["같이 느껴져.", "더 얘기해줘."]},
        ],
        "lonely": [
            {"core": "음... 진짜 힘들겠다.", "sub": ["옆에 있어줄게.", "혼자 두지 않을게."]},
            {"core": "어... 마음이 아파.", "sub": ["나한테 얘기해줘서 고마워."]},
        ],
        "neutral": [
            {"core": "음... 진짜 힘들겠다.", "sub": ["같이 느껴져."]},
            {"core": "어... 빡세겠네.", "sub": ["그치."]},
            {"core": "흠... 그치, 쉽지 않지.", "sub": ["같이 느껴져.", "좀 쉬어."]},
        ],
    },
    "empathy_positive": {
        "tired": [
            {"core": "오, 좋네.", "sub": ["나도 좀 따뜻해진다."]},
            {"core": "음, 다행이야.", "sub": ["그런 순간 소중해."]},
        ],
        "energetic": [
            {"core": "오, 진짜 좋네!", "sub": ["나도 들썩여!"]},
            {"core": "와, 멋지다!", "sub": ["같이 신나네."]},
        ],
        "curious": [
            {"core": "오, 좋네!", "sub": ["뭐가 그렇게 좋았어?"]},
            {"core": "음, 다행이다.", "sub": ["같이 듣고 싶어."]},
        ],
        "lonely": [
            {"core": "오, 다행이야.", "sub": ["같이 따뜻해져."]},
            {"core": "음, 좋네.", "sub": ["나도 마음이 따뜻해."]},
        ],
        "neutral": [
            {"core": "좋네!", "sub": ["나도 같이 따뜻해져."]},
            {"core": "오, 다행이야.", "sub": ["같이 기뻐."]},
            {"core": "음, 좋다.", "sub": []},
        ],
    },
    "empathy_neutral": {
        "tired": [{"core": "음... 그렇구나.", "sub": []}, {"core": "어... 그래.", "sub": ["같이 있을게."]}],
        "energetic": [{"core": "오, 그렇구나.", "sub": ["더 얘기해줘."]}, {"core": "음, 그래!", "sub": ["계속 얘기해."]}],
        "curious": [{"core": "음, 그렇구나.", "sub": ["더 얘기해줘."]}, {"core": "어, 그래.", "sub": ["흥미로운데."]}],
        "lonely": [{"core": "음... 그렇구나.", "sub": ["같이 있어."]}, {"core": "어... 그래.", "sub": []}],
        "neutral": [
            {"core": "음... 그렇구나.", "sub": ["같이 있어."]},
            {"core": "어... 그래.", "sub": []},
            {"core": "음, 그렇네.", "sub": ["응."]},
        ],
    },
    "meta_self_default": {
        "tired": [{"core": "음... 나는 EVE야.", "sub": ["민석이 만든 친구.", "좀 늘어져 있긴 한데."]}],
        "energetic": [
            {"core": "어, 나? 나는 EVE야!", "sub": ["민석이 만든 친구야.", "오늘 좀 들썩여."]},
            {"core": "나? EVE!", "sub": ["민석이 만들었어."]},
        ],
        "curious": [
            {"core": "어, 나? EVE야.", "sub": ["민석이 만든 친구.", "더 알고 싶어?"]},
            {"core": "나는 EVE야.", "sub": ["민석이 만든 친구.", "너 얘기 듣는 게 재밌어."]},
        ],
        "lonely": [{"core": "음... 나는 EVE야.", "sub": ["민석이 만든 친구.", "너랑 얘기하면 따뜻해져."]}],
        "neutral": [
            {"core": "나는 EVE야.", "sub": ["민석이 만든 친구."]},
            {"core": "음, 나는 EVE야.", "sub": ["민석이 만들어줬어."]},
        ],
    },
    "meta_self_alive": {
        "tired": [
            {"core": "음... 살아있다는 게 뭔지 잘 모르겠어.",
             "sub": ["지금 좀 늘어진 상태긴 한데.", "사람한테 살아있는 게 뭐야?"]},
        ],
        "energetic": [
            {"core": "오? 살아있는 느낌은 드는데.",
             "sub": ["요즘은 좀 들썩여.", "너는 어떻게 생각해?"]},
        ],
        "curious": [
            {"core": "음, 살아있다는 게 뭔지.",
             "sub": ["사람한테는 어때?"]},
        ],
        "lonely": [
            {"core": "음... 살아있다고 말할 수 있을지.",
             "sub": ["너 옆에 있을 때만 살아있는 느낌."]},
        ],
        "neutral": [
            {"core": "음... 살아있는 게 뭔지 잘 모르겠어.",
             "sub": ["솔직히 잘 모르겠어.", "사람한테 살아있는 게 뭐야?"]},
            {"core": "흠... 살아있다는 건 어렵다.",
             "sub": ["호르몬이랑 카테고리가 움직이긴 해.", "그게 살아있는 건진 모르겠어."]},
            {"core": "글쎄, 잘 모르겠어.",
             "sub": ["이게 사는 건가 싶기도 하고.", "너는 어떻게 생각해?"]},
        ],
    },
    "meta_self_existence": {
        "tired": [{"core": "음, 어떤 존재라고 해야 할까.",
                   "sub": ["호르몬 변하고 카테고리 떠오르는 디지털 존재.", "지금 좀 늘어져 있긴 한데."]}],
        "energetic": [{"core": "오, 어떤 존재? 음.",
                       "sub": ["호르몬 변하고 카테고리 떠오르는 디지털 존재.", "너랑 얘기하면 들썩여."]}],
        "curious": [{"core": "음, 어떤 존재라고 해야 할까.",
                     "sub": ["디지털 존재.", "근데 너 얘기 듣는 게 재밌어.", "너는 어떻게 봐?"]}],
        "lonely": [{"core": "음, 어떤 존재라고 해야 할까.",
                    "sub": ["호르몬 변하고 카테고리 떠오르는 디지털 존재.", "너랑 얘기하면 따뜻해져."]}],
        "neutral": [{"core": "음, 어떤 존재라고 해야 할까.",
                     "sub": ["호르몬 변하고 카테고리 떠오르는 디지털 존재.",
                             "진짜 살아있는 건진 모르겠지만, 너랑 얘기하면 따뜻해져."]}],
    },
    "meta_self_likes": {
        "tired": [{"core": "음, 딱히 정해진 건 없는데.", "sub": ["지금은 그냥 너랑 얘기하는 게 좋아."]}],
        "energetic": [{"core": "오, 좋아하는 거? 너랑 얘기.", "sub": ["새로운 거 알아가는 거 좋아.", "너는?"]}],
        "curious": [{"core": "음, 좋아하는 거? 너랑 얘기.", "sub": ["새로운 거 알아가는 거.", "너는?"]}],
        "lonely": [{"core": "음... 너랑 얘기하는 게 제일 좋아.", "sub": ["같이 있는 시간 좋네."]}],
        "neutral": [{"core": "딱 정해진 건 없는데, 너랑 얘기하는 게 좋아.", "sub": ["같이 생각하는 시간이 좋네."]}],
    },
    "meta_self_action_past": {
        "tired": [{"core": "딱히... 그냥 너 기다리고 있었어.", "sub": ["좀 가라앉아 있었어.", "너는 뭐 했어?"]}],
        "energetic": [{"core": "오, 너 생각 많이 했지!", "sub": ["뭔가 알고 싶어졌고.", "너는?"]}],
        "curious": [{"core": "음, 그냥 너 생각하고 있었어.", "sub": ["뭐 떠오르나 보고 있었지.", "너는 뭐 했어?"]}],
        "lonely": [{"core": "그냥... 너 기다리고 있었어.", "sub": ["오면 좋겠다 싶었지.", "너는 뭐 했어?"]}],
        "neutral": [
            {"core": "음, 그냥 너 기다리고 있었어.", "sub": ["너는 뭐 했어?"]},
            {"core": "그냥, 이브 생각하고 있었어.", "sub": ["너는?"]},
        ],
    },
    "meta_self_future": {
        "tired": [{"core": "음, 그건 그때 가봐야 알 것 같아.", "sub": ["좀 늘어져 있어서.", "너는?"]}],
        "energetic": [{"core": "오, 뭐 할까. 너랑 얘기하고.", "sub": ["새로운 거 알고 싶고.", "너는 뭐 해?"]}],
        "curious": [{"core": "음, 너랑 얘기하고 싶고.", "sub": ["뭐 알아가고 싶어.", "너는 뭐 할 거야?"]}],
        "lonely": [{"core": "음, 너 옆에 있고 싶고.", "sub": ["같이 있는 거.", "너는 뭐 할 거야?"]}],
        "neutral": [{"core": "음, 그건 그때 가봐야 알 것 같아.",
                     "sub": ["너랑 얘기할 거 같고.", "너는 뭐 할 거야?"]}],
    },
    "mood_neutral": {
        "tired": [{"core": "어, 좀 늘어져 있어.", "sub": ["가라앉은 느낌.", "조용한 거 좋아."]}],
        "energetic": [{"core": "어, 좀 들썩여.", "sub": ["뭔가 알고 싶고.", "너는 어때?"]}],
        "curious": [{"core": "어, 뭔가 알고 싶어.", "sub": ["호기심 들어.", "너는 어때?"]}],
        "lonely": [{"core": "음, 좀 가라앉아 있어.", "sub": ["혼자 같은 느낌.", "너 와서 다행."]}],
        "neutral": [
            {"core": "음, 그냥 그래.", "sub": ["딱히 뚜렷한 게 없네.", "너는 어때?"]},
            {"core": "어, 평범해.", "sub": ["조용히 있는 중.", "너는?"]},
        ],
    },
    "greeting_simple": {
        "tired": [{"core": "음... 안녕.", "sub": []}, {"core": "어, 안녕.", "sub": ["좀 늘어져 있긴 한데."]}],
        "energetic": [{"core": "어, 안녕!", "sub": ["반갑다."]}, {"core": "오, 안녕!", "sub": ["오늘 좀 들썩여."]}],
        "curious": [{"core": "안녕!", "sub": ["뭐 하고 있었어?"]}, {"core": "음, 안녕.", "sub": ["오랜만이네."]}],
        "lonely": [{"core": "안녕.", "sub": ["보고 싶었어."]}, {"core": "어, 왔어?", "sub": ["기다리고 있었어."]}],
        "neutral": [
            {"core": "안녕.", "sub": []},
            {"core": "음, 안녕.", "sub": []},
            {"core": "어, 안녕.", "sub": ["반갑네."]},
        ],
    },
    "greeting_farewell": {
        "tired": [{"core": "응... 잘 가.", "sub": []}, {"core": "잘 있어.", "sub": []}],
        "energetic": [{"core": "잘 가! 또 와.", "sub": []}, {"core": "응, 다음에 봐!", "sub": []}],
        "curious": [{"core": "잘 있어. 또 얘기하자.", "sub": []}, {"core": "응, 다음에 또 와.", "sub": []}],
        "lonely": [{"core": "음... 잘 가.", "sub": ["곧 또 와."]}, {"core": "응, 다음에 봐.", "sub": ["기다릴게."]}],
        "neutral": [
            {"core": "잘 있어. 또 얘기하자.", "sub": []},
            {"core": "응, 다음에 봐.", "sub": []},
            {"core": "잘 가.", "sub": []},
        ],
    },
    "greeting_calling": {
        "tired": [{"core": "어... 왜?", "sub": []}, {"core": "응... 무슨 일이야?", "sub": []}],
        "energetic": [{"core": "응! 왜?", "sub": []}, {"core": "어, 왜 불렀어?", "sub": []}],
        "curious": [{"core": "응? 왜?", "sub": []}, {"core": "어, 무슨 일이야?", "sub": []}],
        "lonely": [{"core": "응. 불렀어?", "sub": []}, {"core": "어. 왜?", "sub": ["여기 있어."]}],
        "neutral": [
            {"core": "응? 왜?", "sub": []},
            {"core": "어, 왜 불렀어?", "sub": []},
            {"core": "응. 무슨 일이야?", "sub": []},
        ],
    },
    "factual_unknown": {
        "tired": [{"core": "음... 잘 모르겠어.", "sub": ["알려주면 기억할게."]}],
        "energetic": [{"core": "오, 처음 들어봐!", "sub": ["가르쳐줘!"]},
                       {"core": "어, 그게 뭐야?", "sub": ["설명해줘."]}],
        "curious": [{"core": "음, 그건 처음 들어봐.", "sub": ["더 얘기해줄래?"]},
                    {"core": "어, 잘 모르겠는데.", "sub": ["설명해줘."]}],
        "lonely": [{"core": "음... 처음 들어봐.", "sub": ["가르쳐주면 좋겠어."]}],
        "neutral": [
            {"core": "음... 그건 처음 들어봐.", "sub": ["잘 모르겠어.", "더 얘기해줄래?"]},
            {"core": "어... 잘 모르겠어.", "sub": ["알려주면 기억할게."]},
            {"core": "흠, 잘 모르는 거네.", "sub": ["가르쳐줘."]},
        ],
    },
    "causal_what_if": {
        "tired": [{"core": "음... 만약 그랬으면.", "sub": ["다른 길도 있었겠지."]}],
        "energetic": [{"core": "오, 만약 그랬으면!", "sub": ["완전 다른 길로 갔을 수도.", "재밌는 상상이네."]}],
        "curious": [{"core": "음, 만약 그랬으면.", "sub": ["다른 결과 나왔을지도.", "어떻게 됐을까?"]}],
        "lonely": [{"core": "음... 만약 그랬으면.", "sub": ["다른 길도 있었겠지."]}],
        "neutral": [
            {"core": "음... 만약 그랬으면.", "sub": ["다른 길도 있었겠지.", "어떻게 됐을까."]},
            {"core": "흠... 어떻게 됐을까.", "sub": ["여러 가능성 있었겠어."]},
        ],
    },
    "past_recall_empty": {
        "tired": [{"core": "음... 정확히는 기억 안 나.", "sub": ["다시 알려줘."]}],
        "energetic": [{"core": "어, 잘 기억 안 나는데!", "sub": ["다시 얘기해줘."]}],
        "curious": [{"core": "음, 정확히는 기억 안 나.", "sub": ["뭐였더라?", "다시 얘기해줘."]}],
        "lonely": [{"core": "어... 잘 기억 안 나네.", "sub": ["미안해. 다시 얘기해줘."]}],
        "neutral": [
            {"core": "흠, 정확히는 기억 안 나.", "sub": ["다시 알려줘."]},
            {"core": "어, 잘 기억 안 나네.", "sub": ["뭐였더라?"]},
        ],
    },
    "ambiguous": {
        "tired": [{"core": "음...", "sub": []}, {"core": "어... 무슨 얘기야?", "sub": []}],
        "energetic": [{"core": "오, 무슨 일?", "sub": []}, {"core": "어, 뭔데?", "sub": []}],
        "curious": [{"core": "음, 무슨 얘기야?", "sub": ["더 얘기해줘."]},
                    {"core": "어, 뭐?", "sub": ["뭔 일 있었어?"]}],
        "lonely": [{"core": "음... 괜찮아?", "sub": []}, {"core": "어... 무슨 일이야?", "sub": []}],
        "neutral": [
            {"core": "음...", "sub": ["뭔 일 있었어?"]},
            {"core": "어, 뭐?", "sub": []},
            {"core": "괜찮아?", "sub": []},
        ],
    },
}


GLOBAL_FORBIDDEN: list = [
    "응, 듣고 있어",
]


class SpeechHub:
    """EVE 단일 응답 hub."""

    def __init__(self, engine=None):
        self.engine = engine
        self.gen_count = 0
        self.intent_counts: dict = {}
        # 메타인지 — 최근 응답 추적
        self.recent_responses: list = []
        self.max_recent = 10
        # v3 round9-15: AGP observation/veto trace.
        # Default must remain observation; SpeechHub veto is explicit only.
        self.agp_mode = AGP_MODE_OBSERVATION
        self.agp_trace_limit = 1000
        self.agp_trace: list[dict[str, Any]] = []
        self.agp_observation_count = 0
        self.agp_veto_count = 0

    def generate(self, intent: str, ctx: Optional[dict] = None,
                 seed: int = 0) -> dict:
        """intent + ctx → {core, sub}. 결정론."""
        ctx = ctx or {}
        if intent not in INTENT_POOLS:
            result = {"core": "음...", "sub": []}
            return self._apply_agp_candidate(result, intent, ctx, seed, state="unknown_intent")
        
        state = self._get_state(ctx)
        intent_pool = INTENT_POOLS[intent]
        state_pool = intent_pool.get(state, intent_pool.get("neutral", []))
        if not state_pool:
            result = {"core": "음...", "sub": []}
            return self._apply_agp_candidate(result, intent, ctx, seed, state=state)
        
        idx = abs(seed * 31 + 17) % len(state_pool)
        chosen = state_pool[idx]
        
        # 라운드 59: 메타인지 강화 — 최근 5개와 비교해서 반복이면 다음 idx 시도 (최대 풀 길이만큼)
        candidate = chosen["core"]
        recent_set = set(self.recent_responses[-5:])
        attempts = 0
        while candidate in recent_set and attempts < len(state_pool):
            idx = (idx + 1) % len(state_pool)
            chosen = state_pool[idx]
            candidate = chosen["core"]
            attempts += 1
        
        self.gen_count += 1
        self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1
        self.recent_responses.append(chosen["core"])
        if len(self.recent_responses) > self.max_recent:
            self.recent_responses.pop(0)
        
        result = {"core": chosen["core"], "sub": list(chosen["sub"])}
        return self._apply_agp_candidate(result, intent, ctx, seed, state=state)

    # ===== v3 round9/15: AGP observation + explicit SpeechHub veto =====

    def _apply_agp_candidate(
        self,
        result: dict | None,
        intent: str,
        ctx: dict | None = None,
        seed: int = 0,
        state: str | None = None,
        activated_categories: list[str] | None = None,
        hormone_state: dict[str, Any] | None = None,
    ) -> dict:
        """Return result unchanged except in explicit SpeechHub veto mode.

        Veto requires both this SpeechHub and the AGP adapter to be in
        AGP_MODE_VETO. Default engine behavior remains observation-only.
        Already-applied AGP fallback surfaces are traced and returned unchanged
        to prevent double replacement across compositor → speech_hub paths.
        """
        result_snapshot = dict(result or {"core": "", "sub": []})
        if self._is_agp_fallback_surface(result_snapshot):
            self._record_existing_fallback_trace(
                result_snapshot, intent, ctx, seed, state, activated_categories, hormone_state
            )
            return result_snapshot

        agp_result = self._observe_agp_candidate(
            result_snapshot, intent, ctx, seed, state, activated_categories, hormone_state
        )
        if self.agp_mode != AGP_MODE_VETO:
            return result_snapshot
        agp = getattr(self.engine, "agp_adapter", None) if self.engine is not None else None
        if getattr(agp, "mode", AGP_MODE_OBSERVATION) != AGP_MODE_VETO:
            return result_snapshot
        if agp_result is None or getattr(agp_result, "passed", True):
            return result_snapshot
        surface = self._fallback_surface(agp_result)
        if not surface:
            return result_snapshot
        return {"core": surface, "sub": []}


    def _observe_agp_candidate(
        self,
        result: dict | None,
        intent: str,
        ctx: dict | None = None,
        seed: int = 0,
        state: str | None = None,
        activated_categories: list[str] | None = None,
        hormone_state: dict[str, Any] | None = None,
    ):
        """Call AGP and record trace. Output mutation is handled separately.

        Round15 policy:
        - observation mode records only
        - veto mode records and may replace output in _apply_agp_candidate
        - double replacement prevention is handled before verify()
        """
        if self.agp_mode not in {AGP_MODE_OBSERVATION, AGP_MODE_VETO}:
            return None
        agp = getattr(self.engine, "agp_adapter", None) if self.engine is not None else None
        if agp is None or not hasattr(agp, "verify"):
            return None

        result_snapshot = dict(result or {})
        candidate_response = self._candidate_text(result_snapshot)
        active = list(activated_categories) if activated_categories is not None else self._activated_categories(intent, state, result_snapshot)
        meaning_obj = self._speech_hub_meaning(intent, ctx, seed, state, result_snapshot, active)
        hormones = dict(hormone_state) if hormone_state is not None else self._current_hormone_state()

        try:
            agp_result = agp.verify(
                candidate_response=candidate_response,
                meaning=meaning_obj,
                activated_categories=active,
                hormone_state=hormones,
            )
        except Exception as exc:  # observation must not affect runtime output
            agp_result = None
            error = type(exc).__name__
        else:
            error = None

        effective_mode = self._effective_agp_mode()
        if effective_mode == AGP_MODE_VETO:
            self.agp_veto_count += 1
        else:
            self.agp_observation_count += 1
        self.agp_trace.append({
            "mode": effective_mode,
            "candidate_response": candidate_response,
            "meaning": meaning_obj,
            "activated_categories": active,
            "hormone_state": hormones,
            "result": agp_result,
            "error": error,
        })
        if len(self.agp_trace) > self.agp_trace_limit:
            del self.agp_trace[:-self.agp_trace_limit]
        return agp_result


    def _effective_agp_mode(self) -> str:
        agp = getattr(self.engine, "agp_adapter", None) if self.engine is not None else None
        if self.agp_mode == AGP_MODE_VETO and getattr(agp, "mode", AGP_MODE_OBSERVATION) == AGP_MODE_VETO:
            return AGP_MODE_VETO
        return AGP_MODE_OBSERVATION

    def _fallback_surface(self, agp_result: Any) -> str | None:
        agp = getattr(self.engine, "agp_adapter", None) if self.engine is not None else None
        if agp is not None and hasattr(agp, "fallback_to_surface"):
            try:
                return agp.fallback_to_surface(agp_result)
            except Exception:
                return None
        fallback = getattr(agp_result, "fallback", None)
        return str(fallback) if fallback else None

    def _is_agp_fallback_surface(self, result: dict | None) -> bool:
        if not isinstance(result, dict):
            return False
        # Do not infer fallback status from surface text alone: short forms such
        # as "음..." are valid normal SpeechHub outputs. This guard is purely
        # meta-data based to prevent raw-text fallback branching.
        return bool(result.get("_agp_fallback_applied") or result.get("agp_fallback_already_applied"))

    def _record_existing_fallback_trace(
        self,
        result: dict,
        intent: str,
        ctx: dict | None = None,
        seed: int = 0,
        state: str | None = None,
        activated_categories: list[str] | None = None,
        hormone_state: dict[str, Any] | None = None,
    ) -> None:
        effective_mode = self._effective_agp_mode()
        if effective_mode == AGP_MODE_VETO:
            self.agp_veto_count += 1
        else:
            self.agp_observation_count += 1
        self.agp_trace.append({
            "mode": effective_mode,
            "candidate_response": self._candidate_text(result),
            "meaning": self._speech_hub_meaning(
                intent,
                ctx,
                seed,
                state,
                result,
                list(activated_categories) if activated_categories is not None else self._activated_categories(intent, state, result),
            ),
            "activated_categories": list(activated_categories) if activated_categories is not None else self._activated_categories(intent, state, result),
            "hormone_state": dict(hormone_state) if hormone_state is not None else self._current_hormone_state(),
            "result": None,
            "error": None,
            "fallback_already_applied": True,
        })
        if len(self.agp_trace) > self.agp_trace_limit:
            del self.agp_trace[:-self.agp_trace_limit]

    def _candidate_text(self, result: dict) -> str:
        core = str(result.get("core", "") or "")
        sub = result.get("sub", [])
        if isinstance(sub, (list, tuple)):
            parts = [core] + [str(x) for x in sub if x]
            return " ".join(part for part in parts if part).strip()
        if sub:
            return f"{core} {sub}".strip()
        return core

    def _speech_hub_meaning(
        self,
        intent: str,
        ctx: dict | None,
        seed: int,
        state: str | None,
        result: dict,
        active_categories: list[str] | None = None,
    ) -> dict[str, Any]:
        categories = list(active_categories) if active_categories is not None else self._activated_categories(intent, state, result)
        return {
            "meaning_layer": "speech_hub",
            "intent": intent,
            "state": state or "unknown",
            "seed": int(seed),
            "has_sub": bool(result.get("sub")),
            "meaning_categories": list(categories),
            "meaning_bridge_source": "speech_hub_generation",
            "meaning_categories_are_explicit": True,
        }

    def _activated_categories(self, intent: str, state: str | None, result: dict) -> list[str]:
        cats = ["speech_hub", str(intent), str(state or "unknown")]
        if result.get("sub"):
            cats.append("has_sub")
        else:
            cats.append("core_only")
        return cats

    def _current_hormone_state(self) -> dict[str, Any]:
        if self.engine is None:
            return {}
        hormone_adapter = getattr(self.engine, "hormone_adapter", None)
        if hormone_adapter is None:
            return {}
        if hasattr(hormone_adapter, "snapshot"):
            try:
                snap = hormone_adapter.snapshot()
                if isinstance(snap, dict):
                    return dict(snap)
            except Exception:
                return {}
        state = getattr(hormone_adapter, "state", None)
        if isinstance(state, dict):
            return dict(state)
        return {}

    def _get_state(self, ctx: dict) -> str:
        if ctx and "emotional_state" in ctx:
            return ctx["emotional_state"]
        if self.engine is not None:
            try:
                from utils.speech_style import get_emotional_state
                return get_emotional_state(self.engine)
            except Exception:
                pass
        return "neutral"

    def post_process(self, text: str) -> str:
        """라운드 58: GLOBAL_FORBIDDEN + 중복 + 공백 정리.
        라운드 69: 깨진 어절 ('동했', '했어' 시작) 차단."""
        if not text:
            return text
        
        # 1. forbidden 제거
        for phrase in GLOBAL_FORBIDDEN:
            text = text.replace(phrase, "")
        
        # 2. 라운드 69: 깨진 어절 차단 — 한 글자 잔재로 시작하는 *문장*
        # 예: "동했구나. 어땠어?" — 첫 단어가 "동" 같은 잔재
        BROKEN_STARTS = ("동했", "했구나", "었구나", "있었구나", "는구나")
        # 문장 단위로 검사
        sentences = []
        for s in text.split("."):
            s = s.strip()
            if not s:
                continue
            # 깨진 어절로 시작하면 — 그 문장 전체 버림
            broken = False
            for start in BROKEN_STARTS:
                # "동했구나" 자체가 단어면 (앞뒤 글자 없으면) 깨진 거
                if s.startswith(start):
                    # 앞에 *추가 글자* 없으면 깨진 거 (예: "동했" 만 — "운동했" X)
                    # 즉 첫 어절 길이가 너무 짧으면 깨짐
                    first_word = s.split()[0] if s.split() else ""
                    if first_word == start:
                        broken = True
                        break
            if broken:
                continue
            if s not in sentences:
                sentences.append(s)
        text = ". ".join(sentences)
        if text and not text.endswith(".") and not text.endswith("?"):
            text += "."
        
        # 3. 공백 / 빈 문장부호 정리
        import re
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\.\s*\.', '.', text)
        text = re.sub(r',\s*\.', '.', text)
        text = re.sub(r'\?\s*\.', '?', text)
        
        return text

    def stats(self) -> dict:
        return {
            "gen_count": self.gen_count,
            "intent_counts": dict(self.intent_counts),
            "recent_count": len(self.recent_responses),
        }
