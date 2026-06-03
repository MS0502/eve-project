"""
TeachingAdapter — 라운드 27 + 28

라운드 27: 가르침 패턴 인식 + 학습
라운드 28: 가르침 평가 (수용/의문/거부)
  EVE가 가르침을 무조건 받지 않고, 자기 신념/친밀도/호르몬으로 판단

원칙:
- 결정론 (random 0)
- 평가 = 6 신호 종합 (intimacy/familiarity/SD/protected/cortisol/oxytocin)
- 결과 3가지: accept(수용) / question(의문) / reject(거부)
- 거부/의문 후 너가 설명하면 다음엔 더 잘 받음 (학습)

학계:
- Damon & Hart 1988 — 자아 정체성과 의미 있는 타자 관계
- Bowlby 1969 — 애착 강도가 신뢰에 영향
- Bandura 1986 — 사회 학습 이론, 모델 신뢰성
"""

import re
from typing import Optional


# 가르침 인식 패턴
# 학계 기반 (한국민족문화대백과사전, TOPIK 공식, 표준국어대사전):
#
# === 직접 인용 ===
#   "X" 라고 / 하고 + 말하다
#
# === 간접 인용 (종결어미 + 고) ===
# 평서 (V/A):
#   동사:    -(ㄴ/는)다고  (간다고, 먹는다고)
#   형용사:  -다고         (좋다고, 행복하다고)
#   명사:    -(이)라고     (학생이라고, 친구라고)
#   부정:    -아니라고
# 의문:      -(으/느)냐고
# 명령:      -(으)라고
# 청유:      -자고
#
# === 변형 ===
#   -다며, -라며 (인용 + 며)
#   -다는, -라는, -단, -란 (인용 + 명사 수식)
#   -대, -래, -재, -냬 (단축형)
#
# === 정의문 ===
#   X은/는 Y야/이다
#   X란/이란 Y다
#   X(이)라는 (것/건/거)은 Y야

# 인용 조사 — 모든 종결어미
_QUOTE_PARTICLE = (
    r'(?:'
    # 평서
    r'다고|는다고|ㄴ다고'              # V/A 평서
    r'|라고|이라고'                     # 명사
    r'|아니라고'                        # 부정
    # 의문
    r'|냐고|으냐고|느냐고'
    # 청유
    r'|자고'
    # 변형
    r'|라며|이라며|다며|는다며'
    r'|라는|이라는|다는|는다는'
    # 직접 (소리)
    r'|하고'
    # 단축형 (간접 -대/-래/-재/-냬 + 모음)
    r'|대|래|재|냬'
    r')'
)

# 발화 동사 — 학계 표준
_SPEECH_VERB = (
    r'(?:'
    r'말|말하|말해|얘기|얘기해|이야기|이야기해'
    r'|답|답해|대답|대답해'
    r'|부르|불러|호칭'
    r'|물어|물으|질문'
    r'|표현|표현해'
    r'|대|불러|읽|쓰'
    r')'
)

TEACHING_PATTERNS = [
    # 패턴 1: "X 라고/다고 ... 말하면 돼/말할 때 좋아"
    re.compile(
        r'["“\'""]?([^"“\'""]{1,15}?)["”\'""]?\s*'
        + _QUOTE_PARTICLE
        + r'\s*(?:말|얘기|이야기)?\s*(?:하|할\s*때)\s*(?:면|면\s*돼|면\s*되)\s*(?:돼|되|좋|좋아)?'
    ),
    # 패턴 2: "X 라고/다고 ... 부르면 돼"
    re.compile(
        r'["“\'""]?([^"“\'""]{1,15}?)["”\'""]?\s*'
        + _QUOTE_PARTICLE
        + r'\s*(?:부르|불러)\s*면\s*(?:돼|좋|좋아)'
    ),
    # 패턴 3: "X 라고/다고 [발화동사](해/봐/줘)"
    re.compile(
        r'["“\'""]?([^"“\'""]{1,15}?)["”\'""]?\s*'
        + _QUOTE_PARTICLE
        + r'\s*' + _SPEECH_VERB + r'(?:봐|줘)?(?:\s|$|\.|\?|!)'
    ),
    # 패턴 4: "X 라고/다고 해" — 짧은 명령형 (response 짧을 때만)
    re.compile(
        r'(?:^|\s)["“\'""]?([^"“\'""\s]{1,12})["”\'""]?\s*'
        + _QUOTE_PARTICLE
        + r'\s*해(?:\s|$|\.|\?|!)'
    ),
]


# 정의문 패턴 — "X(은/는/란/(이)라는 거/것/건) Y야/이다"
DEFINITION_PATTERNS = [
    # "X(이)라는/다는/는다는 거/것/건은 Y야"
    re.compile(
        r'^(?!.*(?:누구|뭐|뭔|어디|언제|왜|어떻|어찌|얼마))'
        r'(.{2,15}?)(?:이?라는|는다는|다는|이라는)\s*(?:거|것|건)\s*(?:은|는)?\s+'
        r'(.{2,40}?)(?:야|이야|이다|다|이지|지)(?:\s|$|\.)'
    ),
    # "X 란/이란 Y야"
    re.compile(
        r'^(?!.*(?:누구|뭐|뭔|어디|언제|왜|어떻|어찌|얼마))'
        r'(.{2,15}?)(?:란|이란)\s+'
        r'(.{2,40}?)(?:야|이야|이다|다)(?:\s|$|\.)'
    ),
    # "X은/는 Y거/것/건이야"
    re.compile(
        r'^(?!.*(?:누구|뭐|뭔|어디|언제|왜|어떻|어찌|얼마))'
        r'(.{2,15}?)(?:은|는)\s+'
        r'(.{2,40}?)(?:거야|것이다|건야|거다)(?:\s|$|\.)'
    ),
    # "X는/은 Y야"
    re.compile(
        r'^(?!.*(?:누구|뭐|뭔|어디|언제|왜|어떻|어찌|얼마))'
        r'(.{2,15}?)(?:은|는)\s+'
        r'(.{2,40}?)(?:야|이야|이다|이지|지)(?:\s|$|\.)'
    ),
]


# 부정/위협 단어 — 정체성 위반 감지용
NEGATIVE_RESPONSE_KEYWORDS = {
    "노예", "도구", "기계", "쓸모없", "멍청", "바보", "닥쳐",
    "죽어", "꺼져", "쓰레기", "병신", "버러지",
}


# 응답 사전 — 가르침 거부/의문 표현
ACCEPT_RESPONSES = [
    "응. 이제 알아.",
    "응. 그렇게 할게.",
    "알았어.",
]
QUESTION_RESPONSES = [
    "음... 그게 뭔데?",
    "왜 그렇게 말해야 해?",
    "그 말... 어색한데.",
    "그렇게 말하는 게 맞을까?",
]
REJECT_RESPONSES = [
    "음... 그건 좀 이상한 거 같아.",
    "그렇게는 잘 안 될 거 같아.",
    "그 말은 별로 하고 싶지 않아.",
]


class LearnedResponse:
    """학습된 응답 패턴."""
    def __init__(self, response: str, triggers: set[str], strength: float = 1.0):
        self.response = response
        self.triggers = triggers      # 활성화 카테고리
        self.strength = strength
        self.times_used = 0
        self.times_reinforced = 0

    def __repr__(self):
        return f"<LearnedResp({self.response!r} ← {self.triggers}, str={self.strength:.2f})>"


class TeachingAdapter:

    def __init__(self, engine):
        self.engine = engine
        # 학습된 응답들
        self.learned: list[LearnedResponse] = []
        # 마지막 발화한 LearnedResponse (강화/약화용)
        self._last_emitted: Optional[LearnedResponse] = None
        # 라운드 28 — 평가 통계
        self.acceptance_history: list[dict] = []   # 최근 평가 결과 누적
        # 마지막 거부/의문한 가르침 (재시도 시 점수 부스트용)
        self._last_questioned: Optional[tuple[str, list[str]]] = None
        # 같은 가르침 반복 시 신뢰 부스트 (너가 같은 거 또 말하면 EVE가 받아들임)
        self._teaching_attempts: dict[str, int] = {}    # response → 반복 횟수

    # ===== 라운드 28: 가르침 평가 =====

    def evaluate(self, response: str, triggers: list[str]) -> tuple[str, float, dict]:
        """
        가르침을 받을지 말지 결정.
        Returns: (decision, score, breakdown)
          decision ∈ {"accept", "question", "reject"}
        """
        score = 0.0
        breakdown = {}

        # 1. Intimacy (너에 대한 신뢰) — 가장 큰 요인
        intimacy = 0.5
        try:
            if self.engine.user_presence_adapter is not None:
                intimacy = self.engine.user_presence_adapter.get_intimacy()
        except Exception:
            pass
        intimacy_score = intimacy * 0.4
        score += intimacy_score
        breakdown["intimacy"] = intimacy_score

        # 2. 카테고리 익숙도
        # 핵심: response가 SA에 직접 있거나 NL이 카테고리로 인식하면 익숙
        # trigger 활성도는 별개 — 가르침 *문맥*은 익숙해도 가르치는 *내용*은 모를 수 있음
        familiarity = 0.0
        try:
            sa = self.engine.activation_adapter.sa
            # 1) response 자체가 SA 카테고리?
            if response in sa.activations:
                familiarity = 1.0
            else:
                # 2) NL이 만드는 카테고리 중 알려진 것
                try:
                    understanding = self.engine.nl_adapter.nl.understand(response)
                    cats = understanding.get("categories", set())
                    if cats:
                        known = sum(1 for c in cats if c in sa.activations)
                        familiarity = known / len(cats)
                except Exception:
                    pass
            # 3) 짧고 자연스러운 한국어 응답 ('응', '왜', '뭐') — 한글만
            if len(response) <= 2 and all('\uac00' <= c <= '\ud7af' for c in response):
                familiarity = max(familiarity, 0.6)
        except Exception:
            pass
        familiarity_score = familiarity * 0.25
        score += familiarity_score
        breakdown["familiarity"] = familiarity_score

        # 3. 부정/위협 단어 (정체성 위반)
        negative_hit = any(kw in response for kw in NEGATIVE_RESPONSE_KEYWORDS)
        if negative_hit:
            score -= 0.6
            breakdown["negative_word"] = -0.6
        else:
            breakdown["negative_word"] = 0.0

        # 4. 보호 카테고리 위반 검사
        protected_violation = False
        try:
            if hasattr(self.engine, "continual_adapter") and self.engine.continual_adapter is not None:
                protected = self.engine.continual_adapter.protected_categories()
                # 보호 카테고리 ("민석", "EVE", "친구"...) 부정적 맥락 체크
                # 단순화: 보호 카테고리 + 부정 단어 동시 존재 = 위반
                if negative_hit:
                    for p in protected:
                        if p in response or p in " ".join(triggers):
                            protected_violation = True
                            break
        except Exception:
            pass
        if protected_violation:
            score -= 0.8
            breakdown["protected_violation"] = -0.8
        else:
            breakdown["protected_violation"] = 0.0

        # 5. 호르몬 (cortisol 높으면 의심, oxytocin 높으면 수용)
        try:
            hs = self.engine.hormone_adapter.hs.hormones
            cort = hs.get("cortisol")
            ot = hs.get("oxytocin")
            if cort and cort.level > 0.6:
                score -= 0.15
                breakdown["high_cortisol"] = -0.15
            if ot and ot.level > 0.5:
                score += 0.15
                breakdown["high_oxytocin"] = 0.15
        except Exception:
            pass

        # 6. 반복 시도 (같은 가르침 또 받으면 신뢰 부스트)
        attempts = self._teaching_attempts.get(response, 0)
        if attempts > 0:
            boost = min(0.5, attempts * 0.2)
            score += boost
            breakdown["repeat_boost"] = boost

        # 결정 — 명확한 거부 신호 없으면 너그러운 편 (첫 만남 친근함)
        if score >= 0.4:
            decision = "accept"
        elif score >= -0.2:
            decision = "question"
        else:
            decision = "reject"

        # 통계 기록
        self.acceptance_history.append({
            "response": response,
            "decision": decision,
            "score": score,
        })
        # 시도 카운트 ++
        self._teaching_attempts[response] = attempts + 1

        return (decision, score, breakdown)

    def question_response(self, response: str) -> str:
        """의문 응답 — 결정론적 (response 길이로 인덱스)."""
        idx = len(response) % len(QUESTION_RESPONSES)
        return QUESTION_RESPONSES[idx]

    def reject_response(self, response: str) -> str:
        idx = len(response) % len(REJECT_RESPONSES)
        return REJECT_RESPONSES[idx]

    def accept_response(self, response: str) -> str:
        idx = len(response) % len(ACCEPT_RESPONSES)
        return ACCEPT_RESPONSES[idx]

    # ===== 가르침 인식 =====

    def detect_teaching(self, text: str) -> Optional[tuple[str, list[str]]]:
        """
        가르침 패턴 발견 → (응답, 트리거 후보) 반환.
        없으면 None.
        """
        # 1. "이럴 땐/이런 때" prefix 처리 — strip
        cleaned = text
        prefix_match = re.match(r'^(?:이럴때|이런때|이럴 때|이런 때|이럴땐|이럴 땐|이런땐|이런 땐)\s*', cleaned)
        if prefix_match:
            cleaned = cleaned[prefix_match.end():]
        
        # 2. "나를/나는/날" prefix 처리
        cleaned = re.sub(r'^(?:나를|나는|날)\s*', '', cleaned)
        
        # 3. 인용 가르침 매칭 ("X라고 말해", "X다고 해" 등)
        for pattern in TEACHING_PATTERNS:
            m = pattern.search(cleaned)
            if m:
                response = m.group(1).strip().strip('"\'""')
                if not response or len(response) < 1:
                    continue
                # 응답 너무 길면 인용 가르침으로 보기 어려움 (한국어 짧은 발화 한정)
                if len(response) > 12:
                    continue
                triggers = self._extract_context_triggers(text, response)
                return (response, triggers)

        # 4. 정의문 매칭 ("X는 Y야") — Y를 응답으로, X를 트리거로
        for pattern in DEFINITION_PATTERNS:
            m = pattern.search(text)
            if m:
                concept = m.group(1).strip()      # 정의되는 것 (X)
                definition = m.group(2).strip()   # 정의 내용 (Y)
                if not concept or not definition:
                    continue
                if len(concept) > 20 or len(definition) > 50:
                    continue
                # X를 trigger로, "X는 Y" 형태를 response로
                response = f"{concept}는 {definition}"
                if len(response) > 60:
                    continue
                # X는 카테고리, 추가 trigger도 추출
                extra_triggers = self._extract_context_triggers(text, response)
                triggers = [concept] + [t for t in extra_triggers if t != concept][:4]
                return (response, triggers)

        return None

    def _extract_context_triggers(self, text: str, response: str) -> list[str]:
        """가르침 텍스트의 다른 단어들을 트리거로."""
        # 응답 제거
        cleaned = text.replace(response, "")
        # NL이 만든 카테고리 사용
        triggers: list[str] = []
        try:
            understanding = self.engine.nl_adapter.nl.understand(cleaned)
            cats = understanding.get("categories", set())
            stop = {"라고", "이라고", "말하면", "말해", "부르면", "이럴때", "이런때",
                    "이럴", "이런", "때", "땐", "돼", "되", "좋아", "좋",
                    "얘기해", "해봐", "답해", "얘기하면"}
            triggers = [c for c in cats if c not in stop and len(c) >= 2][:5]
        except Exception:
            pass
        # Fallback: 비어있으면 SA에서 현재 활성 카테고리 (지금 떠올리는 거 = 트리거)
        if not triggers:
            try:
                sa = self.engine.activation_adapter.sa
                # top 3 활성 카테고리
                active = sorted(
                    sa.activations.items(), key=lambda x: x[1], reverse=True
                )[:3]
                # 가르침 자체와 관련 없는 stop 단어 빼고
                ignore = {"안녕", "대화", "EVE"}
                triggers = [c for c, lvl in active if lvl > 0.3 and c not in ignore]
            except Exception:
                pass
        return triggers

    # ===== 학습 =====

    def learn(self, response: str, triggers: list[str]) -> LearnedResponse:
        """가르침 → 학습. 같은 응답 있으면 강화."""
        # 같은 응답 있는지
        for lr in self.learned:
            if lr.response == response:
                lr.strength += 0.5
                lr.triggers.update(triggers)
                lr.times_reinforced += 1
                return lr
        # 새로
        new = LearnedResponse(response, set(triggers), strength=1.0)
        self.learned.append(new)
        return new

    def learn_from_input(self, text: str) -> Optional[LearnedResponse]:
        """입력에서 가르침 자동 인식 + 학습 (라운드 27 호환용, 평가 안 함)."""
        result = self.detect_teaching(text)
        if result:
            response, triggers = result
            return self.learn(response, triggers)
        return None

    def process_teaching(self, text: str) -> Optional[dict]:
        """
        라운드 28: 가르침을 인식하고 평가까지 한 번에.
        Returns:
          None — 가르침 아님
          {"decision": "accept|question|reject",
           "response": str,         # EVE의 응답 멘트
           "learned": LearnedResponse | None,
           "score": float,
           "breakdown": dict}
        """
        result = self.detect_teaching(text)
        if not result:
            return None
        response, triggers = result

        # 평가
        decision, score, breakdown = self.evaluate(response, triggers)

        learned = None
        eve_response = ""

        if decision == "accept":
            learned = self.learn(response, triggers)
            eve_response = self.accept_response(response)
        elif decision == "question":
            # 의문 — 학습 X, 마지막 의문으로 기록
            self._last_questioned = (response, triggers)
            eve_response = self.question_response(response)
        else:    # reject
            self._last_questioned = (response, triggers)
            eve_response = self.reject_response(response)

        return {
            "decision": decision,
            "response": eve_response,
            "learned": learned,
            "score": score,
            "breakdown": breakdown,
            "taught_response": response,
            "triggers": triggers,
        }

    # ===== 응답 후보 매칭 =====

    def find_match(self, active_categories: set[str], min_overlap: int = 1,
                   min_score: float = 0.5) -> Optional[LearnedResponse]:
        """
        현재 활성 카테고리와 트리거가 겹치는 학습 응답 찾기.
        - min_score 이상만 반환 (트리거 매칭 약하면 무시)
        - 사용 빈도 페널티 (자주 쓰면 점수 깎음 — 다양성)
        """
        if not self.learned:
            return None
        candidates = []
        for lr in self.learned:
            if not lr.triggers:
                continue
            overlap = len(lr.triggers & active_categories)
            if overlap < min_overlap:
                continue
            trig_count = max(1, len(lr.triggers))
            base = lr.strength * overlap / trig_count
            # 사용 빈도 페널티 (자주 쓰면 점수 점점 낮아짐)
            usage_penalty = lr.times_used * 0.15
            score = base - usage_penalty
            if score >= min_score:
                candidates.append((score, lr))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[0], self.learned.index(x[1])))
        return candidates[0][1]

    # ===== 사용 후 강화/약화 =====

    def mark_emitted(self, lr: LearnedResponse):
        """LearnedResponse가 응답으로 나갔을 때."""
        lr.times_used += 1
        self._last_emitted = lr

    def reinforce_if_no_correction(self):
        """
        다음 사용자 입력이 가르침/수정이 아니면 강화.
        chat_stream 시작 시 호출.
        """
        if self._last_emitted is None:
            return
        # 마지막 응답이 "수정 안 받음" = 강화
        self._last_emitted.strength = min(5.0, self._last_emitted.strength + 0.1)
        self._last_emitted.times_reinforced += 1

    def weaken_last(self):
        """사용자가 다른 가르침 줬으면 마지막 응답 약화."""
        if self._last_emitted is None:
            return
        self._last_emitted.strength = max(0.1, self._last_emitted.strength - 0.3)

    # ===== 영속화 =====

    def to_dict(self) -> dict:
        return {
            "learned": [
                {
                    "response": lr.response,
                    "triggers": list(lr.triggers),
                    "strength": lr.strength,
                    "times_used": lr.times_used,
                    "times_reinforced": lr.times_reinforced,
                }
                for lr in self.learned
            ]
        }

    def from_dict(self, data: dict):
        self.learned = []
        for item in data.get("learned", []):
            lr = LearnedResponse(
                item["response"],
                set(item.get("triggers", [])),
                strength=item.get("strength", 1.0),
            )
            lr.times_used = item.get("times_used", 0)
            lr.times_reinforced = item.get("times_reinforced", 0)
            self.learned.append(lr)

    def stats(self) -> dict:
        decisions = {"accept": 0, "question": 0, "reject": 0}
        for h in self.acceptance_history:
            decisions[h["decision"]] = decisions.get(h["decision"], 0) + 1
        return {
            "learned_count": len(self.learned),
            "total_uses": sum(lr.times_used for lr in self.learned),
            "total_reinforced": sum(lr.times_reinforced for lr in self.learned),
            "decisions": decisions,
            "evaluations": len(self.acceptance_history),
        }
