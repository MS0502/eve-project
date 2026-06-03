"""
DialogueContextAdapter — 라운드 54

GPT 짚은 핵심: EVE가 *자기 직전 발화*를 기억 못함.
- EVE: "인사 처음 들어봐. 설명해주면 기억할게."  (질문!)
- 사용자: "처음 만났을때 하는거야 반가워서"  (답)
- EVE: "EVE 생각이 같이 떠오르네."  ← 답인 줄 모름

해결:
- EVE 직전 응답 추적
- 응답에 "X 알려줄래?", "설명해줘" 등 *질문 패턴* 감지
- 다음 사용자 입력을 *그 X의 정의*로 학습 시도

학계: Grice 1975 cooperative principle (대화 협력 원칙)
"""

import re
import time
from dataclasses import dataclass, field
from typing import Optional


# EVE가 *질문*했다는 신호 패턴
QUESTION_SIGNALS = [
    r'(?:알려|설명|가르쳐|얘기)\s*해?\s*(?:줄래|줘)',
    r'알려주면\s*기억',
    r'설명해주면',
    r'(?:뭔지|뭐야|뭔가)\s*\??$',
    r'정확히\s*뭔지',
]


@dataclass
class PendingQuestion:
    """EVE가 직전에 한 질문 — 다음 turn에 답 기다림."""
    target: str             # 묻고 있는 대상 ("인사" 등)
    eve_response: str       # EVE 직전 응답 전체
    timestamp: float
    answered: bool = False


class DialogueContextAdapter:
    """
    EVE 직전 응답 추적 + *EVE가 물어본 것*에 대한 다음 답 인식.
    """

    def __init__(self, engine=None, ttl_sec: float = 120.0):
        self.engine = engine
        self.ttl_sec = ttl_sec
        self.pending: Optional[PendingQuestion] = None
        # 통계
        self.questions_asked = 0
        self.answers_received = 0

    def record_eve_response(self, response: str, recent_user_input: str = ""):
        """EVE 응답 기록 — 질문 패턴 있으면 PendingQuestion."""
        if not response:
            return
        
        # 질문 패턴 검사
        is_question = False
        for pat in QUESTION_SIGNALS:
            if re.search(pat, response):
                is_question = True
                break
        
        if not is_question:
            return
        
        # 묻고 있는 대상 추출 — 사용자가 직전에 물은 단어
        # 예: 사용자 "인사가 뭔지 알아?" → EVE "인사 처음 들어봐, 설명해줘" → target = "인사"
        target = self._extract_target(recent_user_input, response)
        if target:
            self.pending = PendingQuestion(
                target=target,
                eve_response=response,
                timestamp=time.time(),
            )
            self.questions_asked += 1

    def get_pending_target(self) -> Optional[str]:
        """현재 EVE가 답 기다리는 대상."""
        if self.pending is None:
            return None
        # TTL 체크
        if time.time() - self.pending.timestamp > self.ttl_sec:
            self.pending = None
            return None
        if self.pending.answered:
            return None
        return self.pending.target

    def receive_answer(self, user_input: str) -> Optional[tuple]:
        """
        사용자 답 받기. EVE가 직전에 물었으면 (target, definition) 반환.
        """
        if self.pending is None:
            return None
        target = self.get_pending_target()
        if target is None:
            return None
        
        # 사용자 입력이 진짜 답 같은지 — 명시적 의문문 / 다른 주제 X
        ui = (user_input or "").strip()
        if not ui or "?" in ui or "?" in ui:
            return None
        # 너무 짧으면 답 X (예: "응", "그래")
        if len(ui) < 4:
            return None
        # 의문 어휘 있으면 답 X
        if any(q in ui for q in ["뭐야", "뭔지", "어떻게", "왜"]):
            return None
        
        # 답으로 받음
        self.pending.answered = True
        self.answers_received += 1
        return (target, ui)

    def _extract_target(self, user_input: str, eve_response: str) -> Optional[str]:
        """사용자 입력에서 EVE가 묻는 대상 추출."""
        if not user_input:
            return None
        # "X가 뭐야?" / "X 뭔지 알아?" 패턴
        patterns = [
            r'([\w가-힣]{2,15}?)(?:가|이)\s*뭔지',
            r'([\w가-힣]{2,15}?)(?:가|이|은|는)\s*뭐',
            r'([\w가-힣]{2,15}?)\s*뜻',
        ]
        for pat in patterns:
            m = re.search(pat, user_input)
            if m:
                target = m.group(1).strip()
                # stopword 거름
                if target in ("그게", "이게", "저게", "뭐가", "뭐"):
                    continue
                # 조사 한 번 더 제거 (안전)
                for p in ("가", "이", "은", "는"):
                    if target.endswith(p) and len(target) > len(p):
                        target = target[:-len(p)]
                        break
                return target
        return None

    def stats(self) -> dict:
        return {
            "questions_asked": self.questions_asked,
            "answers_received": self.answers_received,
            "pending": (self.pending.target if self.pending else None),
        }
