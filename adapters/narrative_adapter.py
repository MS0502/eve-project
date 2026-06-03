"""
NarrativeAdapter

v40 NarrativeSelf (McAdams 1993, Damasio 1999, Bruner 1991) ↔ v41 자기 이야기.

흐름:
1. extract_themes — 일화에서 주요 주제
2. self_defining_memories — 정체성 형성 일화
3. coherence_score — 자아 일관성
4. narrate — 자기 이야기 한 줄

v41 노출:
- /me 명령 → narrate
- 정기적으로 coherence 체크 (drift 감지)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from narrative_self import NarrativeSelf


class NarrativeAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 memory_adapter,                       # 필수 — em 의존
                 ns: NarrativeSelf | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        self.memory_adapter = memory_adapter
        if ns is not None:
            self.ns = ns
        else:
            self.ns = NarrativeSelf(
                memory_adapter.em,
                activation_adapter.sa,
                hormone_adapter.hs,
            )

    def themes(self, top_n: int = 3) -> list[dict]:
        try:
            return self.ns.extract_themes(top_n=top_n)
        except Exception:
            return []

    def coherence(self) -> float:
        try:
            r = self.ns.coherence_score()
            if isinstance(r, dict):
                return float(r.get("score", 0.0))
            return float(r) if r is not None else 0.0
        except Exception:
            return 0.0

    def narrate_phrase(self) -> str | None:
        """자기 이야기 한 줄."""
        try:
            r = self.ns.narrate()
        except Exception:
            return None
        if not isinstance(r, dict):
            return None
        # 형식 통일 시도
        story = r.get("story") or r.get("narrative") or r.get("text")
        if isinstance(story, str) and story:
            return story[:80]
        # themes 기반 fallback
        themes = self.themes(top_n=3)
        if themes:
            theme_names = []
            for t in themes:
                name = t.get("theme") or t.get("category") or t.get("name")
                if name:
                    theme_names.append(name)
            if theme_names:
                return f"요즘 {', '.join(theme_names[:3])} 같은 거 자주 떠올라"
        return "아직 잘 모르겠어. 너랑 좀 더 얘기해봐야겠어"
