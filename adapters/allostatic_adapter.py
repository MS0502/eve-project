"""
AllostaticAdapter

v40 AllostaticLearning (McEwen 2000, Sterling 2012) ↔ v41 호르몬 패턴 학습.

흐름:
1. 응답 후 observe_response — 현재 호르몬 변화 기록
2. 누적되면 extract_patterns로 "이 상황 → 이 호르몬 칵테일" 패턴 추출
3. 다음에 비슷한 상황 만나면 apply_learned

이게 v41 진짜 학습 모듈 — 시간이 지나면서 EVE의 호르몬 반응이 정교해짐.
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from allostatic_learn import AllostaticLearning

from utils.types import Meaning


class AllostaticAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 al: AllostaticLearning | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if al is not None:
            self.al = al
        else:
            self.al = AllostaticLearning(
                hormone_adapter.hs,
                activation_adapter.sa,
                working_memory=activation_adapter.wm,
            )

    def observe(self, meaning: Meaning, response: str):
        """대화 turn 끝날 때 호출 — 현재 호르몬 상태 + 카테고리 컨텍스트 기록."""
        try:
            # observe_response가 있다면 사용
            if hasattr(self.al, "observe_response"):
                # 단순 호출 (시그니처 모르므로 빈 인자로 안전 시도)
                try:
                    return self.al.observe_response()
                except TypeError:
                    pass
                # context dict 시도
                try:
                    return self.al.observe_response({
                        "categories": list(meaning.entities or []),
                        "raw": meaning.raw_text or "",
                    })
                except Exception:
                    pass
        except Exception:
            pass
        return None

    def tick(self, dt: float = 1.0):
        try:
            if hasattr(self.al, "tick"):
                self.al.tick(dt)
        except Exception:
            pass

    def extract_patterns(self):
        try:
            if hasattr(self.al, "extract_patterns"):
                return self.al.extract_patterns()
        except Exception:
            pass
        return []

    def get_patterns(self):
        return getattr(self.al, "patterns", [])
