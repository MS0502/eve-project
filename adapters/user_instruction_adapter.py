"""
UserInstructionAdapter — 라운드 52

GPT 짚은 핵심: 사용자가 "X 빼고 말해" 같이 직접 지시해도 EVE가 무시.
- 지시 분류
- 응답 생성 후 constraint 적용
- forbidden phrases 자동 추가

학계: Kowalski 1979 algorithm = logic + control. 사용자가 control 직접 지시.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UserConstraint:
    """사용자 제약."""
    type: str           # "remove_phrase" / "short" / "no_emoji" / "tone"
    target: str = ""    # 제거할 문구 / 톤 이름 등
    duration: int = 5   # 몇 turn 동안 (디폴트 5)
    remaining: int = 5


# 항상 적용되는 forbidden phrases (사용자 지시 누적)
DEFAULT_FORBIDDEN: list = [
    # 라운드 52: 사용자가 "응 듣고있어 빼"라고 한 적 있음 → 디폴트 제외
]


class UserInstructionAdapter:
    """
    사용자 메타 지시 처리.
    - "응 듣고 있어 빼고 말해" → forbidden 추가
    - "짧게 답해" → 짧기 모드
    - "재밌게 답해" → 톤 변경
    """

    def __init__(self, engine=None):
        self.engine = engine
        self.constraints: list = []
        self.parse_count = 0

    def parse_instruction(self, text: str) -> Optional[UserConstraint]:
        """
        사용자 텍스트에서 메타 지시 추출.
        Returns: UserConstraint or None.
        """
        if not text:
            return None
        t = text.strip()
        
        # "X 빼고 말해" / "X 안 하고 말해" / "X 없이 답해"
        m = re.search(r'(.{1,30}?)\s*(?:빼고|안\s*하고|없이|말고)\s*(?:말해|답해|해|얘기해)', t)
        if m:
            target = m.group(1).strip().strip('"\' ').strip()
            # "뒤에 X" 같은 prefix 제거
            target = re.sub(r'^(?:뒤에|앞에|문장에|답에|응답에)\s*', '', target).strip()
            if target and len(target) >= 2:
                self.parse_count += 1
                return UserConstraint(
                    type="remove_phrase",
                    target=target,
                    duration=10,
                    remaining=10,
                )
        
        # "짧게 답해" / 내용 요구 + "짧게 말해"는 constraint 등록 후 계속
        if (m := re.search(r'^(?:좀\s*)?(?:(.+?)\s+)?(?:짧게|간단하게|간결하게)\s*(?:답|말|해|얘기)', t)):
            self.parse_count += 1
            c = UserConstraint(type="short", duration=5, remaining=5); return c if not m[1] else (self.add_constraint(c) or None)
        
        # "그냥 X라고만 해봐" — 단답 강제
        m = re.search(r'그냥\s*(.{1,15}?)(?:이?라고만|만)\s*(?:해|말해)', t)
        if m:
            target = m.group(1).strip().strip('"\' ')
            if target:
                self.parse_count += 1
                return UserConstraint(
                    type="exact_response",
                    target=target,
                    duration=1,
                    remaining=1,
                )
        
        # 라운드 54: "X만 해봐" / "X만 말해" — 그냥 없이도
        m = re.search(r'^(.{1,15}?)\s*만\s*(?:해|말해|얘기해)\s*(?:봐|줘)?\s*\??\s*$', t)
        if m:
            target = m.group(1).strip().strip('"\' ')
            # 의문문/명령문이 아닐 때
            if target and target not in ("어떻게", "왜", "뭐", "그냥"):
                self.parse_count += 1
                return UserConstraint(
                    type="exact_response",
                    target=target,
                    duration=1,
                    remaining=1,
                )
        
        return None

    def add_constraint(self, c: UserConstraint):
        """constraint 추가 (중복 제거)."""
        # 같은 type+target 있으면 갱신
        for i, ex in enumerate(self.constraints):
            if ex.type == c.type and ex.target == c.target:
                self.constraints[i] = c
                return
        self.constraints.append(c)

    def apply_constraints(self, text: str) -> str:
        """응답에 모든 활성 constraint 적용."""
        if not text:
            return text
        result = text
        
        for c in self.constraints:
            if c.remaining <= 0:
                continue
            if c.type == "remove_phrase":
                # 부분 매칭 — c.target 부분 문자열 다 제거
                result = result.replace(c.target, "")
                # 빈 문장부호 정리
                result = re.sub(r'\.\s*\.', '.', result)
                result = re.sub(r'\s+', ' ', result).strip()
            elif c.type == "exact_response":
                # 정확한 응답으로 강제 (한 번만)
                result = c.target
        
        # forbidden 항상 적용
        for phrase in DEFAULT_FORBIDDEN:
            result = result.replace(phrase, "")
        
        return result.strip()

    def tick(self):
        """매 turn 호출 — duration 감소."""
        for c in self.constraints:
            c.remaining -= 1
        # 만료된 거 제거
        self.constraints = [c for c in self.constraints if c.remaining > 0]

    def is_short_mode(self) -> bool:
        """짧게 모드 활성?"""
        return any(c.type == "short" and c.remaining > 0 
                   for c in self.constraints)

    def is_exact_mode(self) -> Optional[str]:
        """정확한 응답 강제 모드? 있으면 응답 텍스트 반환."""
        for c in self.constraints:
            if c.type == "exact_response" and c.remaining > 0:
                return c.target
        return None

    def stats(self) -> dict:
        return {
            "parse_count": self.parse_count,
            "active_constraints": [
                {"type": c.type, "target": c.target, "remaining": c.remaining}
                for c in self.constraints
            ],
        }

    def acknowledge_response(self, instruction: UserConstraint) -> str:
        """지시 받았다는 인정 응답."""
        if instruction.type == "remove_phrase":
            return f"알았어. '{instruction.target}' 빼고 말할게."
        elif instruction.type == "short":
            return "응. 짧게 답할게."
        elif instruction.type == "exact_response":
            return instruction.target
        return "응."
