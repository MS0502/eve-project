"""
AppraisalClassifier — EVE v3 round3

Consolidates the temporary semantic guards from round73 patch8-12 into a single
classifier surface. This is still a compatibility layer: the legacy marker sets
are FROZEN and must not be extended with new noun lists. Future rounds should
replace marker-based checks with lexical/concept mapping and AGP input
stabilization.

Purpose:
- distinguish user inner affect from target appraisal
- keep false-emotion routes out of emotional_share
- preserve existing behavior while creating one migration point
- expose minimal meaning fields for future AGP input stabilization
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional


LABEL_AMBIENT_WEATHER = "ambient_weather_statement"
LABEL_POSITIVE_OBJECT = "positive_object_appraisal_statement"
LABEL_NEGATIVE_OBJECT = "negative_object_appraisal_statement"
LABEL_EXTERNAL_AFFECTIVE_TONE = "external_affective_tone_statement"
LABEL_EXTERNAL_THREAT_DISCOMFORT = "external_threat_or_discomfort_statement"
LABEL_NONE = "none"

SUBJECT_TARGET = "target"
SUBJECT_USER = "user"
SUBJECT_UNKNOWN = "unknown"

AFFECT_POSITIVE = "positive"
AFFECT_NEGATIVE = "negative"
AFFECT_AFFECTIVE_TONE = "affective_tone"
AFFECT_THREAT_DISCOMFORT = "threat_or_discomfort"
AFFECT_WEATHER_APPRAISAL = "weather_appraisal"
AFFECT_NONE = "none"

TARGET_APPRAISAL_LABELS = frozenset(
    {
        LABEL_AMBIENT_WEATHER,
        LABEL_POSITIVE_OBJECT,
        LABEL_NEGATIVE_OBJECT,
        LABEL_EXTERNAL_AFFECTIVE_TONE,
        LABEL_EXTERNAL_THREAT_DISCOMFORT,
    }
)


@dataclass(frozen=True)
class AppraisalResult:
    """Result of target-appraisal vs user-affect stabilization."""

    label: str
    route_override: Optional[str]
    via: str
    text: str
    is_target_appraisal: bool
    is_user_affect: bool
    reason: str
    subject_type: str = SUBJECT_UNKNOWN
    affect_type: str = AFFECT_NONE
    meaning_layer: str = "appraisal"

    @property
    def should_route_small_talk(self) -> bool:
        return self.route_override == "small_talk"

    def as_meaning_dict(self) -> dict:
        """Return a stable, AGP-friendly meaning summary without changing routing."""
        return {
            "label": self.label,
            "subject_type": self.subject_type,
            "affect_type": self.affect_type,
            "route_override": self.route_override,
            "via": self.via,
            "is_target_appraisal": self.is_target_appraisal,
            "is_user_affect": self.is_user_affect,
            "reason": self.reason,
        }


class AppraisalClassifier:
    """Frozen compatibility classifier for patch8-12 semantic guards.

    Do not add new object noun keywords here. v3 requires the next expansion to
    come from lexical/concept mapping, not another marker list.
    """

    def _affect_type_for_label(self, label: str) -> str:
        if label == LABEL_AMBIENT_WEATHER:
            return AFFECT_WEATHER_APPRAISAL
        if label == LABEL_POSITIVE_OBJECT:
            return AFFECT_POSITIVE
        if label == LABEL_NEGATIVE_OBJECT:
            return AFFECT_NEGATIVE
        if label == LABEL_EXTERNAL_AFFECTIVE_TONE:
            return AFFECT_AFFECTIVE_TONE
        if label == LABEL_EXTERNAL_THREAT_DISCOMFORT:
            return AFFECT_THREAT_DISCOMFORT
        return AFFECT_NONE

    def analyze(self, text: str) -> AppraisalResult:
        t = (text or "").strip()
        if not t:
            return self._none(t, "empty")
        if "?" in t or "？" in t:
            return self._none(t, "question")

        checks = (
            (LABEL_AMBIENT_WEATHER, self._is_ambient_weather_statement),
            (LABEL_POSITIVE_OBJECT, self._is_positive_object_appraisal_statement),
            (LABEL_NEGATIVE_OBJECT, self._is_negative_object_appraisal_statement),
            (LABEL_EXTERNAL_AFFECTIVE_TONE, self._is_external_affective_tone_statement),
            (LABEL_EXTERNAL_THREAT_DISCOMFORT, self._is_external_threat_or_discomfort_statement),
        )
        for label, predicate in checks:
            if predicate(t):
                return AppraisalResult(
                    label=label,
                    route_override="small_talk",
                    via="semantic_guard",
                    text=t,
                    is_target_appraisal=True,
                    is_user_affect=False,
                    reason=label,
                    subject_type=SUBJECT_TARGET,
                    affect_type=self._affect_type_for_label(label),
                )
        return self._none(t, "no_target_appraisal")

    def route_override(self, text: str) -> Optional[str]:
        return self.analyze(text).route_override

    def _none(self, text: str, reason: str) -> AppraisalResult:
        return AppraisalResult(
            label=LABEL_NONE,
            route_override=None,
            via="appraisal_classifier",
            text=text,
            is_target_appraisal=False,
            is_user_affect=True,
            reason=reason,
            subject_type=SUBJECT_USER,
            affect_type=AFFECT_NONE,
        )

    # ===== FROZEN legacy semantic guards from round73 patch8-12 =====

    def _is_positive_object_appraisal_statement(self, text: str) -> bool:
        t = text
        positive = any(k in t for k in ("좋", "멋지", "예쁘", "괜찮", "훌륭"))
        if not positive:
            return False

        inner_affect_markers = (
            "기분", "마음", "감정", "행복", "기뻐", "기쁨", "즐거", "신나",
            "설레", "뿌듯", "외로", "우울", "슬프", "힘들", "피곤", "지쳐",
        )
        if any(k in t for k in inner_affect_markers):
            return False

        interpersonal_markers = ("너", "니", "널", "넌", "당신", "이브", "EVE")
        if any(k in t for k in interpersonal_markers):
            return False

        object_markers = (
            "노래", "음악", "영화", "드라마", "책", "글", "사진", "그림", "영상",
            "게임", "커피", "음식", "밥", "맛", "코드", "결과", "답변", "아이디어",
            "분위기", "하늘", "색", "디자인", "옷", "방", "장면",
        )
        if any(k in t for k in object_markers):
            return True

        if re.search(r'^(?:나(?:는|도)?\s+)?(?:이|그|저)\s*.{1,12}\s*(?:좋아|좋다|괜찮아|괜찮다|멋지다|예쁘다)\s*$', t):
            return True
        return False

    def _is_negative_object_appraisal_statement(self, text: str) -> bool:
        t = text
        negative = any(k in t for k in ("싫", "별로", "안 좋", "안좋", "마음에 안 들", "맘에 안 들", "별루"))
        if not negative:
            return False

        inner_affect_markers = (
            "기분", "마음이", "마음은", "마음도", "감정", "우울", "슬프", "힘들", "피곤", "지쳐",
            "괴로", "아파", "불안", "걱정", "짜증", "화나", "답답", "찝찝",
        )
        if any(k in t for k in inner_affect_markers):
            return False

        interpersonal_markers = ("너", "니", "널", "넌", "당신", "이브", "EVE")
        if any(k in t for k in interpersonal_markers):
            return False

        object_markers = (
            "노래", "음악", "영화", "드라마", "책", "글", "사진", "그림", "영상",
            "게임", "커피", "음식", "밥", "맛", "코드", "결과", "답변", "아이디어",
            "분위기", "하늘", "색", "디자인", "옷", "방", "장면",
        )
        if any(k in t for k in object_markers):
            return True

        if re.search(r'^(?:나(?:는|도)?\s+)?(?:이|그|저)\s*.{1,12}\s*(?:싫어|싫다|별로야|별로다|별루야|안\s*좋아|안\s*좋다|마음에\s*안\s*들어|맘에\s*안\s*들어)\s*$', t):
            return True
        return False

    def _is_external_affective_tone_statement(self, text: str) -> bool:
        t = text
        affective_tone = any(k in t for k in ("슬프", "슬퍼", "우울", "외롭", "외로", "괴롭"))
        if not affective_tone:
            return False

        inner_markers = (
            "나 슬", "난 슬", "나는 슬", "내가 슬",
            "나 우울", "난 우울", "나는 우울", "내가 우울",
            "기분", "마음", "감정", "눈물", "울었", "울고", "울 것",
        )
        if any(k in t for k in inner_markers):
            return False

        reaction_markers = ("보고", "보니까", "보니", "듣고", "들으니까", "들으니", "때문", "해서", "하니까")
        if any(k in t for k in reaction_markers):
            return False

        target_markers = (
            "영화", "드라마", "노래", "음악", "영상", "장면", "사진", "그림",
            "책", "글", "이야기", "스토리", "분위기", "엔딩", "결말", "게임",
        )
        if any(k in t for k in target_markers):
            return True

        if re.search(r'^(?:이|그|저)\s*.{1,16}\s*(?:슬프다|슬퍼|우울하다|우울해|외롭다|외로워|괴롭다|괴로워)\s*$', t):
            return True
        return False

    def _is_external_threat_or_discomfort_statement(self, text: str) -> bool:
        t = text
        markers = ("무섭", "무서워", "불편", "위험", "찝찝")
        if not any(k in t for k in markers):
            return False

        inner_markers = (
            "나 무서", "난 무서", "나는 무서", "내가 무서",
            "기분", "마음", "감정", "불안", "두려", "걱정", "초조",
        )
        if any(k in t for k in inner_markers):
            return False

        interpersonal_markers = ("너", "니", "널", "넌", "당신", "이브", "EVE")
        if any(k in t for k in interpersonal_markers):
            return False

        target_markers = (
            "영화", "공포영화", "노래", "음악", "영상", "게임", "장면", "사진",
            "의자", "방", "장소", "여기", "거기", "저기", "길", "밤길",
            "상황", "분위기", "코드", "결과", "답변", "아이디어", "계획",
        )
        if any(k in t for k in target_markers):
            return True

        if re.search(r'^(?:이|그|저)\s*.{1,16}\s*(?:무섭다|무서워|불편해|불편하다|위험해|위험하다|찝찝해|찝찝하다)\s*$', t):
            return True
        return False

    def _is_ambient_weather_statement(self, text: str) -> bool:
        t = text
        if "날씨" not in t:
            return False

        positive_weather = any(k in t for k in ("좋", "맑", "화창", "쾌청", "따뜻", "시원"))
        if not positive_weather:
            return False

        user_affect_markers = (
            "기분", "마음", "나는", "내가", "나 ", "난 ", "나도",
            "행복", "기뻐", "신나", "설레", "뿌듯", "힘들", "피곤",
        )
        if any(k in t for k in user_affect_markers):
            return False
        return True
