"""
Orchestrator — 라운드 29 (B-1)

역할:
1. classify(meaning) → 입력을 9가지 상황 중 하나로 분류
2. select_modules(situation) → 상황별 핵심 모듈 명시
3. gather_context(meaning, modules) → 선택 모듈에서 컨텍스트 수집

원칙:
- 결정론 (random 0). 동률 시 알파벳 tiebreaker.
- 추가만 — 기존 streaming 흐름 안 바꿈
- 응답 합성은 B-2에서 (이번 단계 X)
- B-1은 "어떤 상황인지 + 어떤 모듈이 필요한지" 만 결정

학계:
- Newell 1990 — Unified Theories of Cognition (controller architecture)
- Anderson 1996 — ACT-R production system (조건 → 행동 매칭)
- Sun 2006 — CLARION (top-level rule-based vs bottom-level)
"""

import re
from typing import Optional

from adapters.appraisal_classifier import AppraisalClassifier


# ===== 상황 9종 =====

# 상황별 분류 키워드 (한국어). 우선순위 = 위에서 아래.

# EVE v3 ROUND1 FROZEN TEMPORARY GUARDS
# ---------------------------------------
# The round73 patch8-12 semantic guards are now consolidated behind
# AppraisalClassifier (v3 round3). The legacy behavior remains frozen for
# compatibility. Do not extend them with new noun keyword lists.
# New noun keyword expansion is prohibited by v3 principle 3.
# Future semantic routing must move through lexical/concept mapping + AGP input
# stabilization, not new marker lists.

_APPRAISAL_CLASSIFIER = AppraisalClassifier()


def _analyze_appraisal_statement(text: str):
    return _APPRAISAL_CLASSIFIER.analyze(text)


def _is_positive_object_appraisal_statement(text: str) -> bool:
    return _analyze_appraisal_statement(text).label == "positive_object_appraisal_statement"


def _is_negative_object_appraisal_statement(text: str) -> bool:
    return _analyze_appraisal_statement(text).label == "negative_object_appraisal_statement"


def _is_external_affective_tone_statement(text: str) -> bool:
    return _analyze_appraisal_statement(text).label == "external_affective_tone_statement"


def _is_external_threat_or_discomfort_statement(text: str) -> bool:
    return _analyze_appraisal_statement(text).label == "external_threat_or_discomfort_statement"


def _is_ambient_weather_statement(text: str) -> bool:
    return _analyze_appraisal_statement(text).label == "ambient_weather_statement"

SITUATION_PATTERNS = [
    # ===== 라운드 41: 의문문 5분기 (GPT 분석) =====
    # 우선순위 무조건 지킬 것:
    # 1. greeting (인사)
    # 2. meta_self (EVE 관련)
    # 3. meta_user (사용자 관련)
    # 4. emotional_share (감정)
    # 5. factual_question (마지막, 진짜 정보 질문만)

    # statement 1번째 — 평서문 (의문문 분류 전에 필터)
    # 단, 끝에 ? 없을 때만 (의문문 보호)
    ("statement_self_intro", [
        re.compile(
            r'^(?!.*[?？])'   # 의문 마커 없을 때만
            # "나는 X야" / "내가 X야" — 다른 글자가 뒤에 안 붙어야 ("내일" X)
            r'^(?:나는|내가|나|내)\s+'
            r'.{2,40}?(?:야|이야|이다|다|이지|지)(?:\s|\.|$)(?!\s*\?)'
        ),
    ]),

    # teaching — "X라고/다고/이라고/냐고/자고 말해/해" + 정의문 ("X는 Y야") + 설명형
    # 라운드 41: 의문문 분류보다 우선 (가르침이 의문문 형태일 수 있음)
    ("teaching", [
        # 인용 가르침 (다양한 종결어미 + 발화 동사)
        re.compile(
            r'(?:라고|이라고|다고|는다고|ㄴ다고|냐고|으냐고|느냐고|자고|아니라고|라며|이라며|다며|는다며|라는|이라는|다는|는다는|하고)'
            r'\s*(?:말|얘기|이야기|답|대답|부르|불러|물어|물으|질문|표현)'
        ),
        # 정의문 "X는/은 Y야" — 의문 어휘 없을 때만
        re.compile(
            r'^(?!.*(?:누구|뭐|뭔|어디|언제|왜|어떻|어찌|얼마))'
            r'.{2,15}?(?:은|는|란|이란|라는|이라는|다는|는다는)(?:\s+|\s*(?:거|것|건)\s*)'
            r'.{2,40}?(?:야|이야|이다|다|이지|지)(?:\s|\.|$)(?!\s*\?)'
        ),
        # 설명형 가르침 — "~ 하는 거야" / "~ 하는 말이야" / "~ 뜻이야"
        # ("뭐해가 뭔지" 답변 포함 — "너가 무엇을 하는지 물어보는거야")
        re.compile(
            r'^(?!.*\?)'    # 의문 마커 없을 때만
            r'.+(?:는\s*(?:거|것|건)|는\s*말|뜻|의미)(?:야|이야|이다)(?:\s|\.|$)'
        ),
    ]),

    # 1️⃣ greeting — 인사 (의문문 분류보다 *먼저*)
    ("greeting", [
        re.compile(r'(?:^|\s)안녕'),
        re.compile(r'(?:^|\s)하이'),
        re.compile(r'(?:^|\s)헤이'),
        re.compile(r'(?:^|\s)안뇽'),
        # 라운드 55: 오타 / 변형 인사
        re.compile(r'(?:^|\s)안년'),
        re.compile(r'(?:^|\s)안냐'),
        re.compile(r'(?:^|\s)할로'),
        re.compile(r'(?:^|\s)하잉'),
        re.compile(r'(?:^|\s)왔어'),
        re.compile(r'오랜만'),
        re.compile(r'잘\s*있었'),
        re.compile(r'잘\s*지냈'),
        re.compile(r'^\s*뭐\s*해\s*\??\s*$'),
        re.compile(r'^\s*뭐하\s*\??\s*$'),
        re.compile(r'^\s*잘\s*있어\s*\??\s*$'),
        # 라운드 55: "뭐해 이브?" / "이브 뭐해?" — 이름 붙은 짧은 안부
        re.compile(r'^\s*뭐\s*해\s*(?:이브|EVE|니|넌|당신)\s*\??\s*$'),
        re.compile(r'^\s*(?:이브|EVE)\s*뭐\s*해\s*\??\s*$'),
        # 라운드 53: 인사 명령
        re.compile(r'인사\s*(?:해|해봐|해줘)'),
        # 라운드 57: 단독 호명 ("이브야", "이브!", "EVE야")
        re.compile(r'^\s*(?:이브|EVE)\s*[야아!]?\s*\??\s*$'),
        # 라운드 57: 작별 ("잘있어", "잘 가", "안녕히")
        re.compile(r'^\s*잘\s*있어\s*\.?\s*$'),
        re.compile(r'^\s*잘\s*가\s*\.?\s*$'),
        re.compile(r'안녕히\s*(?:가|계)'),
        re.compile(r'(?:bye|바이)'),
    ]),

    # 2️⃣ meta_self — EVE 자체에 대한 질문 (확장)
    ("meta_self", [
        # 너는 누구/뭐
        re.compile(r'(?:^|[\s]+)(?:너|니|넌|당신|이브|EVE)\s*(?:는|가)?\s*(?:누구|뭐|뭔|살아|존재|진짜)'),
        re.compile(r'EVE\s*(?:는|가)?\s*(?:누구|뭐|살아|존재)'),
        # 너 뭐 좋아해? / 너는 뭘 좋아해?
        re.compile(r'(?:너|니|넌|당신|이브|EVE)\s*(?:는|가)?\s*(?:뭘|뭐를|뭔)\s*좋아'),
        re.compile(r'(?:너|니|넌|당신|이브|EVE).*(?:좋아해|싫어해|원해|원하|하고\s*싶|뭐\s*하고)'),
        # 좋아하는게 뭐야? / 좋아하는거 뭐
        re.compile(r'좋아하(?:는|시는)?(?:게|거)\s*뭐'),
        # 하고싶은 게 뭐야? / 뭘 하고싶어?
        re.compile(r'하고\s*싶은\s*(?:게|것|거)\s*(?:뭐|뭔)'),
        re.compile(r'뭐\s*하고\s*싶'),
        re.compile(r'뭘\s*하고\s*싶'),
        # 뭐할거야 / 뭐할래 (미래 질문)
        re.compile(r'뭐\s*(?:할거|할래|할까)'),
        # 궁금한 거
        re.compile(r'궁금'),
        re.compile(r'관심\s*(?:있|없)'),
        # 기분 / 상태
        re.compile(r'(?:기분|상태)\s*(?:어때|어떄)'),
        # 살아있어? / 진짜야? / 어떤 존재
        re.compile(r'(?:살아|진짜야|실제야)'),
        re.compile(r'어떤\s*(?:존재|사람|친구|AI|ai)'),
        # 라운드 55: 짧은 관용 질문 "이브 생각?" / "EVE 생각?"
        re.compile(r'^(?:이브|EVE)\s*생각\s*\??\s*$'),
        # 라운드 57: 단독 "뭐했어?" / "뭐 했어?" — EVE한테 묻는 행동 질문
        re.compile(r'^\s*뭐\s*했어\s*\??\s*$'),
        re.compile(r'^\s*뭐하고\s*있었\s*\??\s*$'),
    ]),

    # 3️⃣ meta_user — 사용자에 대한 질문 (확장)
    ("meta_user", [
        # 내가 누구?
        re.compile(r'(?:내|나)\s*(?:가|는)?\s*(?:누구|누군지)'),
        # 나 알아? / 나 기억해?
        re.compile(r'(?:^|[\s]+)(?:나|내)\s*(?:를|에 대해|에대해)?\s*(?:알아|기억|알고)'),
        re.compile(r'(?:^|[\s]+)(?:나|내)\s*(?:를|에 대해|에대해)'),
        # 나 어때?
        re.compile(r'(?:^|[\s]+)(?:나|내)\s*(?:는|가)?\s*어때'),
    ]),

    # 4️⃣ causal_what_if
    ("causal_what_if", [
        re.compile(r'만약'),
        re.compile(r'(?:안|않았)\s*었으면|었더라면'),
        re.compile(r'(?:이라면|라면|다면)\s*(?:어땠|어떨)'),
    ]),

    # 5️⃣ past_recall
    ("past_recall", [
        re.compile(r'저번에|전에|어제|예전에|지난'),
        re.compile(r'기억\s*(?:나|해|하)'),
    ]),

    # 6️⃣ task — 계산/도구
    ("task", [
        re.compile(r'\d+\s*(?:더하기|빼기|곱하기|나누기|\+|\-|\*|/)\s*\d+'),
        re.compile(r'(?:계산|풀어|구해)'),
    ]),

    # 6.5️⃣ concept_query — "X 뭐야?" / "X가 뭔지 알아?" / "X가 무슨 뜻이야?"
    # meta_self/meta_user 다음, factual 전 (구체적 개념 질문)
    ("concept_query", [
        # "뭐해가 뭔지 알아?" / "PT가 뭐야?"
        re.compile(r'.{2,15}?(?:가|이|은|는)?\s*뭔지\s*(?:알|모르)'),
        re.compile(r'.{2,15}?(?:가|이)\s*뭐(?:야|지)?\??\s*$'),
        re.compile(r'.{2,15}?(?:이|가|은|는|란|이란)\s*뭐'),
        # "X가 무슨 뜻이야?" / "X 뜻이 뭐야?"
        re.compile(r'.{2,15}?(?:가|이|은|는)?\s*무슨\s*뜻'),
        re.compile(r'.{2,15}?\s*뜻(?:이|은)?\s*뭐'),
    ]),

    # 7️⃣ emotional_share — 감정 표현
    ("emotional_share", [
        re.compile(r'힘들|지쳐|지쳤|피곤|졸|우울|슬프|슬퍼|슬펐|외로|괴로|아파|아프|빡셌|빡세|죽겠|죽을\s*맛'),
        re.compile(r'기뻐|기쁨|행복|좋아|즐거|신나|뿌듯|설레|좋다'),
        re.compile(r'화나|짜증|싫어|미워|답답|찝찝|열받|빡쳐'),
        re.compile(r'(?:기분|마음|감정).{0,8}(?:별로|안\s*좋|안좋|싫)'),
        re.compile(r'무서워|무섭|불안|두려|걱정|초조|불편'),
    ]),

    # 8️⃣ factual_question — 진짜 정보 질문만 (마지막)
    # 의도적으로 좁게 — meta/greeting/emotional 다 위에서 이미 처리됨
    ("factual_question", [
        re.compile(r'(?:어디|언제|어떻게|왜|어느)\s*(?:야|지|.+\?)'),
        re.compile(r'(?:몇|얼마)'),
        re.compile(r'(?:^|\s)(?:뭐야|뭐지|뭔가|뭔지)$'),
    ]),
]


# ===== 상황별 핵심 모듈 (B-2에서 응답 합성에 활용) =====

# 모듈 이름 = engine 속성 이름
SITUATION_MODULES = {
    "greeting": [
        "user_presence_adapter",   # 친밀도 + gap 인사
        "hormone_adapter",          # 호르몬에 따른 톤
        "enhancer_adapter",         # 인사 패턴
    ],
    "meta_self": [
        "integrated_self_adapter",  # 현재 자기 상태
        "digital_somatic_adapter",  # 신체 감각
        "narrative_adapter",        # 자기 이야기
        "hormone_adapter",          # 현재 기분
    ],
    "meta_user": [
        "user_presence_adapter",    # 사용자 정체
        "memory_adapter",           # 사용자 일화
        "narrative_adapter",
    ],
    "emotional_share": [
        "suffering_adapter",        # 공감
        "user_presence_adapter",
        "memory_adapter",           # 비슷한 일화
        "hormone_adapter",          # 호르몬 동조
        "narrative_adapter",
    ],
    "factual_question": [
        "world_model_adapter",      # 카테고리 묘사
        "cg",                       # 인과 (simulation_engine.cg)
        "hypergraph_adapter",
        "memory_adapter",
    ],
    "task": [
        "task_solver",
        "deep_reasoning_adapter",
    ],
    "teaching": [
        "teaching_adapter",         # 평가 + 학습
        "user_presence_adapter",    # 친밀도 (수용 결정)
        "continual_adapter",        # 보호 카테고리
    ],
    "causal_what_if": [
        "counterfactual_adapter",
        "simulation_engine",
        "temporal_adapter",
    ],
    "past_recall": [
        "memory_adapter",
        "temporal_adapter",
        "narrative_adapter",
    ],
    "small_talk": [
        "activation_adapter",
        "dmn_adapter",
        "enhancer_adapter",
    ],
}


class OrchestratorAdapter:

    def __init__(self, engine):
        self.engine = engine
        # 통계
        self.classification_history: list[dict] = []

    # ===== 분류 =====

    def classify(self, meaning) -> str:
        """
        입력을 상황 9종 중 하나로 분류.
        결정론 — 같은 입력 → 같은 분류.
        
        라운드 36 (C-4): 한국어 구조 분석을 *보조*로 (high-confidence 특정 intent만).
        """
        text = (meaning.raw_text or "").strip() if hasattr(meaning, "raw_text") else ""
        if not text:
            return "small_talk"

        # === 0. 의미 가드 ===
        # "오늘 날씨 좋다"는 positive affect 단어(좋다)를 포함하지만
        # 감정 공유가 아니라 외부 환경 관찰이다. emotional_share 오탐을 먼저 차단.
        appraisal = _analyze_appraisal_statement(text)
        if appraisal.should_route_small_talk:
            return "small_talk"

        # === 1. 정규식 패턴 우선 (특화된 패턴 먼저) ===
        # causal_what_if, past_recall, task 등은 패턴이 더 정확
        for situation, patterns in SITUATION_PATTERNS:
            for pat in patterns:
                if pat.search(text):
                    return situation

        # === 2. 한국어 구조 분석 (보조 — 패턴 못 잡은 경우) ===
        if (hasattr(self.engine, "korean_adapter")
                and self.engine.korean_adapter is not None):
            try:
                struct = self.engine.korean_adapter.analyze(text)
                # high-confidence (≥ 0.8) + 특정 intent만 받아들임
                if struct.intent_confidence >= 0.8:
                    intent_to_situation = {
                        "teaching":   "teaching",
                        "definition": "teaching",
                        "meta_self":  "meta_self",
                        "meta_user":  "meta_user",
                        "emotional":  "emotional_share",
                    }
                    sit = intent_to_situation.get(struct.intent)
                    if sit:
                        return sit
            except Exception:
                pass

        # === 3. intent 기반 fallback ===
        intent = getattr(meaning, "intent", "statement")
        if intent == "question":
            return "factual_question"
        if intent == "command":
            return "task"

        return "small_talk"

    def classify_with_trace(self, meaning) -> dict:
        """분류 + 어떻게 결정됐는지 trace."""
        text = (meaning.raw_text or "").strip() if hasattr(meaning, "raw_text") else ""
        appraisal = _analyze_appraisal_statement(text)
        if appraisal.should_route_small_talk:
            return {
                "situation": appraisal.route_override,
                "matched_pattern": appraisal.label,
                "match_text": text,
                "via": appraisal.via,
                "appraisal": appraisal.as_meaning_dict(),
            }
        for situation, patterns in SITUATION_PATTERNS:
            for pat in patterns:
                m = pat.search(text)
                if m:
                    return {
                        "situation": situation,
                        "matched_pattern": pat.pattern,
                        "match_text": m.group(0),
                        "via": "pattern",
                    }
        intent = getattr(meaning, "intent", "statement")
        if intent == "question":
            return {"situation": "factual_question", "via": "intent_question"}
        if intent == "command":
            return {"situation": "task", "via": "intent_command"}
        return {"situation": "small_talk", "via": "fallback"}

    # ===== 모듈 선택 =====

    def select_modules(self, situation: str) -> list[str]:
        """상황별 핵심 모듈 이름 반환."""
        return list(SITUATION_MODULES.get(situation, SITUATION_MODULES["small_talk"]))

    def get_adapter(self, name: str):
        """모듈 이름 → 실제 어댑터 객체. 없으면 None."""
        # 특수 케이스
        if name == "cg":
            sim = getattr(self.engine, "simulation_engine", None)
            return getattr(sim, "cg", None) if sim else None
        if name == "task_solver":
            return getattr(self.engine, "task_solver", None)
        # 일반 — engine.X
        return getattr(self.engine, name, None)

    # ===== 컨텍스트 수집 =====

    def gather_context(self, meaning, modules: Optional[list[str]] = None,
                       situation: Optional[str] = None) -> dict:
        """
        선택된 모듈들에서 응답 합성에 쓸 컨텍스트 수집.
        B-2에서 응답 생성 시 이 dict를 사용.
        """
        if modules is None:
            if situation is None:
                situation = self.classify(meaning)
            modules = self.select_modules(situation)

        ctx: dict = {"situation": situation}

        for mod_name in modules:
            adapter = self.get_adapter(mod_name)
            if adapter is None:
                continue
            try:
                key, value = self._extract_module_context(mod_name, adapter, meaning)
                if value is not None:
                    ctx[key] = value
            except Exception:
                continue

        return ctx

    def _extract_module_context(self, name: str, adapter, meaning):
        """모듈별 컨텍스트 추출 (어댑터 인터페이스가 다 다르니 wrapper)."""
        if name == "user_presence_adapter":
            return ("user_presence", {
                "intimacy": adapter.get_intimacy(),
                "gap_phrase": adapter.gap_phrase(),
                "user_name": adapter.user_name,
            })

        if name == "hormone_adapter":
            try:
                hormones = {n: h.level for n, h in adapter.hs.hormones.items()}
                return ("hormones", {
                    "levels": hormones,
                    "primary": adapter.hs.primary_hormone() if hasattr(adapter.hs, "primary_hormone") else None,
                })
            except Exception:
                return ("hormones", None)

        if name == "digital_somatic_adapter":
            try:
                return ("somatic", {"feeling_phrase": adapter.feeling_phrase()})
            except Exception:
                return ("somatic", None)

        if name == "integrated_self_adapter":
            try:
                snap = adapter.snapshot()
                # SelfSnapshot dataclass → dict
                if hasattr(snap, "__dataclass_fields__"):
                    from dataclasses import asdict
                    snap_dict = asdict(snap)
                    # primary_hormone은 tuple → 첫 요소만
                    if isinstance(snap_dict.get("primary_hormone"), tuple):
                        snap_dict["primary_hormone"] = snap_dict["primary_hormone"][0]
                    return ("self", snap_dict)
                return ("self", snap)
            except Exception:
                return ("self", None)

        if name == "memory_adapter":
            # 현재 카테고리 기반 일화 회상
            try:
                cats = set(meaning.entities or [])
                if cats:
                    episodes = adapter.em.recall(cats, top_n=3, min_overlap=1)
                    return ("memory", {"similar_episodes": episodes})
            except Exception:
                pass
            return ("memory", {"similar_episodes": []})

        if name == "world_model_adapter":
            # 카테고리 묘사
            try:
                cats = meaning.entities or []
                descriptions = []
                for c in cats[:3]:
                    desc = adapter.describe(c) if hasattr(adapter, "describe") else None
                    if desc:
                        descriptions.append({"category": c, "description": desc})
                return ("world_model", {"descriptions": descriptions})
            except Exception:
                return ("world_model", None)

        if name == "cg":
            try:
                cats = meaning.entities or []
                children = {}
                for c in cats[:3]:
                    kids = list(adapter.children(c))[:3] if hasattr(adapter, "children") else []
                    if kids:
                        children[c] = kids
                return ("causal", {"children": children})
            except Exception:
                return ("causal", None)

        if name == "counterfactual_adapter":
            try:
                cats = meaning.entities or []
                if cats and hasattr(adapter, "what_if"):
                    return ("counterfactual", {"what_if": adapter.what_if(cats[0])})
            except Exception:
                pass
            return ("counterfactual", None)

        if name == "simulation_engine":
            try:
                cats = meaning.entities or []
                if cats:
                    sims = adapter.simulate_from(cats[0], depth=2) if hasattr(adapter, "simulate_from") else None
                    return ("simulation", sims)
            except Exception:
                pass
            return ("simulation", None)

        if name == "narrative_adapter":
            try:
                if hasattr(adapter, "tell_about_self"):
                    return ("narrative", adapter.tell_about_self())
            except Exception:
                pass
            return ("narrative", None)

        if name == "suffering_adapter":
            try:
                # 공감 신호 — 호르몬 + 감정 키워드
                emotions = getattr(meaning, "emotions", {}) or {}
                return ("suffering_signal", {
                    "user_emotions": emotions,
                    "should_empathize": bool(emotions),
                })
            except Exception:
                return ("suffering_signal", None)

        if name == "temporal_adapter":
            try:
                if hasattr(adapter, "infer_time_frame"):
                    return ("time_frame", adapter.infer_time_frame(meaning))
            except Exception:
                pass
            return ("time_frame", None)

        if name == "teaching_adapter":
            text = getattr(meaning, "raw_text", "") or ""
            try:
                preview = adapter.detect_teaching(text)
                return ("teaching_detected", preview)     # tuple or None
            except Exception:
                return ("teaching_detected", None)

        if name == "continual_adapter":
            try:
                return ("protected", list(adapter.protected_categories()))
            except Exception:
                return ("protected", [])

        if name == "task_solver":
            # task_solver.solve는 dict task object를 받음.
            # streaming에서 task 추출은 understanding 단계에서 일어남.
            # B-1에서는 task_solver 호출 안 하고 "사용 가능" 만 표시.
            return ("task_result", {"available": True})

        if name == "deep_reasoning_adapter":
            try:
                if hasattr(adapter, "reason_about"):
                    cats = meaning.entities or []
                    if cats:
                        return ("reasoning", adapter.reason_about(cats[0]))
            except Exception:
                pass
            return ("reasoning", None)

        if name == "hypergraph_adapter":
            try:
                cats = meaning.entities or []
                if cats and hasattr(adapter, "neighbors"):
                    return ("hypergraph", {"neighbors": list(adapter.neighbors(cats[0]))[:5]})
            except Exception:
                pass
            return ("hypergraph", None)

        if name == "activation_adapter":
            try:
                top = sorted(adapter.sa.activations.items(), key=lambda x: x[1], reverse=True)[:5]
                return ("active_categories", [(c, lvl) for c, lvl in top if lvl > 0.1])
            except Exception:
                return ("active_categories", [])

        if name == "dmn_adapter":
            try:
                if hasattr(adapter, "current_mode"):
                    return ("dmn_mode", adapter.current_mode())
            except Exception:
                pass
            return ("dmn_mode", None)

        if name == "enhancer_adapter":
            return ("enhancer", "available")

        return (name, None)

    # ===== 종합 호출 (B-1 핵심 진입점) =====

    def process(self, meaning) -> dict:
        """
        chat_stream에서 호출 — 분류 + 컨텍스트 한 번에.
        라운드 31: mood + 대화 흐름도 ctx에 포함.
        라운드 32: 모듈 선택 학습 통합.
        """
        trace = self.classify_with_trace(meaning)
        situation = trace["situation"]
        modules = self.select_modules(situation)

        # 라운드 32: 학습된 utility로 모듈 재정렬
        learner = getattr(self.engine, "module_learning_adapter", None)
        if learner is not None:
            try:
                # 마지막 turn 효과 측정 (재정렬 *전*에)
                last_input = getattr(meaning, "raw_text", "") or ""
                effect = learner.evaluate_last(last_input)
                # 모듈 재정렬
                modules = learner.reorder_modules(situation, modules)
                # 이번 선택 기록
                learner.record_selection(situation, modules)
            except Exception:
                pass

        ctx = self.gather_context(meaning, modules=modules, situation=situation)

        # 라운드 31: 분위기 + 맥락
        mood_adapter = getattr(self.engine, "mood_adapter", None)
        if mood_adapter is not None:
            try:
                ctx["mood"] = mood_adapter.infer_mood()
                ctx["flow"] = mood_adapter.conversation_flow()
                ctx["response_hint"] = mood_adapter.get_response_hint()
                text = getattr(meaning, "raw_text", "") or ""
                if mood_adapter.is_ambiguous_input(text):
                    ctx["is_ambiguous"] = True
                    ctx["ambiguous_interpretation"] = mood_adapter.interpret_ambiguous(text, ctx["mood"])
            except Exception:
                pass

        result = {
            "situation": situation,
            "trace": trace,
            "modules_selected": modules,
            "context": ctx,
        }
        self.classification_history.append({
            "text": getattr(meaning, "raw_text", "") or "",
            "situation": situation,
            "via": trace.get("via"),
        })
        return result

    # ===== 통계 =====

    def stats(self) -> dict:
        situations = {}
        for h in self.classification_history:
            s = h["situation"]
            situations[s] = situations.get(s, 0) + 1
        return {
            "total_classifications": len(self.classification_history),
            "situations": situations,
        }

    def history(self, n: int = 10) -> list[dict]:
        return list(self.classification_history[-n:])
