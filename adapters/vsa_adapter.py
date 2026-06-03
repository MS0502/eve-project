"""
VSAAdapter

v40 VSABinding (Smolensky 1990, Plate 1995 HRR) ↔ v41 System1 후보 확장.

흐름:
1. Meaning에서 (entity, emotion) 페어 추출
2. compose({"entity": e, "emotion": emo}) → 합성 벡터/키
3. nearest_category로 비슷한 합성과 매칭된 카테고리 찾음
4. System1 후보에 추가 (activation_adapter 후보 위에 한 층 더)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from vsa_binding import VSABinding

from utils.types import Meaning


class VSAAdapter:

    def __init__(self,
                 hormone_adapter=None,
                 activation_adapter=None,
                 vsa: VSABinding | None = None):
        if vsa is not None:
            self.vsa = vsa
        else:
            from spreading_activation import SpreadingActivation
            from hormone_system import HormoneSystem
            sa = activation_adapter.sa if activation_adapter else SpreadingActivation()
            hs = hormone_adapter.hs if hormone_adapter else HormoneSystem()
            self.vsa = VSABinding(sa, hs)

    def compose_meaning(self, meaning: Meaning) -> str | None:
        """Meaning을 VSA composite key로. 없으면 None."""
        if not meaning.entities:
            return None
        role_filler: dict[str, str] = {}
        # 첫 entity를 actor 역할로
        role_filler["actor"] = meaning.entities[0]
        # 가장 강한 감정을 emotion 역할로
        if meaning.emotions:
            top_emo = max(meaning.emotions.items(), key=lambda x: x[1])[0]
            role_filler["emotion"] = top_emo
        if meaning.actions:
            role_filler["action"] = meaning.actions[0]
        if len(role_filler) < 2:  # 합성할 게 없음
            return None
        try:
            result = self.vsa.compose(role_filler)
            return result.get("composite_name") if isinstance(result, dict) else None
        except Exception:
            return None

    def nearest_categories(self, composite_name: str, k: int = 3,
                           exclude: set[str] | None = None) -> list[tuple[str, float]]:
        try:
            return self.vsa.nearest_category(composite_name, k=k, exclude=exclude or set())
        except Exception:
            return []

    def expand_candidates(self, meaning: Meaning) -> list[dict]:
        """v41 System1 후보 풀에 추가할 항목들."""
        composite = self.compose_meaning(meaning)
        if composite is None:
            return []
        # 입력 entity + emotion은 제외 (자기 자신 매칭 제거)
        exclude = set(meaning.entities or [])
        exclude |= set((meaning.emotions or {}).keys())

        out: list[dict] = []
        # 라운드 65: 입력 entity 자체에 *반응 어미* 적용 시도
        # ("운동했어" → "운동했구나" — VSA 의존 X)
        from utils.korean_endings import reaction_for
        for ent in (meaning.entities or []):
            r = reaction_for(ent)
            if r is not None:
                out.append({
                    "text": r,
                    "score": 0.55,    # VSA 후보보다 높게
                    "source": f"reaction:{ent}",
                })
                # 한 번만 — 첫 entity 기준
                break

        for cat, sim in self.nearest_categories(composite, k=3, exclude=exclude):
            if sim < 0.2:
                continue
            # "X@role" 같은 합성 마커 정리 — 사람한테 보일 부분
            display = cat.split("@")[0].split("+")[0]
            if not display or display in exclude:
                continue
            # 라운드 65: "X 같은 느낌도 같이 떠올라" 제거
            # 이건 *시스템 내부 출력* — 사람 말투 X.
            # VSA 유사도는 score만 영향, 별도 텍스트 후보 X.
            # (만약 reaction_for로 후보 못 만들면 system1의 minimal_ack가 fallback)
            pass
        return out
