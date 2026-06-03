"""
SufferingAdapter

v40 Suffering (Singer & Lamm 2009 dual circuit) ↔ v41 공감.

v41 EMOTION_PLAN에 이미 fatigue/sadness 위로 표현 있음.
이 어댑터는 거기에 더해:
- self-other distinction (민석 고통 vs EVE 고통)
- 누적 burnout 감지 (계속 부정 감정 → 자기 보호)
- intimacy 추적 (compassion 횟수)

흐름:
1. meaning.emotions에서 suffering 카테고리 감지
2. resonate / compassion_action 호출 → sub_point에 위로 추가
3. burnout 감지되면 compassion 출력 약화
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from suffering import Suffering

from utils.types import Meaning


# v41 emotion → suffering category 매핑
EMOTION_TO_SUFFERING = {
    "fatigue":  "피로",
    "sadness":  "슬픔",
    "anger":    "분노",
    "anxiety":  "불안",
}


class SufferingAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 suffering: Suffering | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if suffering is not None:
            self.suffering = suffering
        else:
            self.suffering = Suffering(activation_adapter.sa, hormone_adapter.hs)

    def detect(self, meaning: Meaning) -> str | None:
        """첫 번째 감지된 suffering 카테고리 반환."""
        for emo in (meaning.emotions or {}):
            if emo in EMOTION_TO_SUFFERING:
                return EMOTION_TO_SUFFERING[emo]
        return None

    def compassion_phrase(self, meaning: Meaning, target: str = "민석") -> str | None:
        """공감 한 줄. burnout이면 짧게/없음."""
        suffering_cat = self.detect(meaning)
        if not suffering_cat:
            return None
        try:
            result = self.suffering.compassion_action(target, suffering_cat)
        except Exception:
            return None
        # burnout이면 약하게 표현
        if isinstance(result, dict):
            if result.get("burnout"):
                return None  # 자기 보호 — compassion 안 함
            # 결과에서 자연어 추출
            phrase = result.get("phrase") or result.get("response") or result.get("action_text")
            if phrase:
                return phrase
        # 기본 공감 표현
        return f"{target}이 {suffering_cat}한 거 같이 느껴져"

    def get_intimacy(self, target: str = "민석") -> float:
        try:
            return float(self.suffering.compute_intimacy(target))
        except Exception:
            return 0.0
