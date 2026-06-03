"""
WorldModelAdapter

v40 WorldModel ↔ v41 카테고리 묘사 + 관계.

- describe_concept(X) → X에 대해 아는 것
- relate(X, Y) → X와 Y의 관계 강도/타입
- knowledge_overview → 자기가 아는 영역 요약

v41 통합:
- "X 뭐야?" / "X에 대해 알아?" 패턴 → describe_concept
- "X랑 Y 관련 있어?" → relate
- planner 분기로 응답
"""

import re

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from world_model import WorldModel

from utils.types import Meaning


# 묘사 질문 패턴
_DESCRIBE_PATTERNS = [
    re.compile(r"(\S+?)\s*(?:이|가)?\s*뭐야"),
    re.compile(r"(\S+?)에\s*대해\s*(?:알아|말해)"),
]

# 관계 질문 패턴
_RELATE_PATTERN = re.compile(r"(\S+?)\s*(?:이|가|와|과|랑)?\s*(\S+?)\s*(?:이|가)?\s*관련")


def extract_describe_query(text: str) -> str | None:
    for p in _DESCRIBE_PATTERNS:
        m = p.search(text)
        if m:
            target = m.group(1).strip()
            for suf in ("을", "를", "이", "가", "은", "는"):
                if target.endswith(suf) and len(target) > 1:
                    target = target[:-1]
                    break
            if target and len(target) >= 1:
                return target
    return None


def extract_relate_query(text: str) -> tuple[str, str] | None:
    m = _RELATE_PATTERN.search(text)
    if not m:
        return None
    a, b = m.group(1).strip(), m.group(2).strip()
    if a and b and a != b:
        return (a, b)
    return None


class WorldModelAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 memory_adapter=None,
                 tm_adapter=None,
                 an_adapter=None,
                 wm: WorldModel | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        if wm is not None:
            self.wm = wm
        else:
            from causal_graph import CausalGraph
            cg = CausalGraph(activation_adapter.sa)
            em = memory_adapter.em if memory_adapter else None
            tm_inst = tm_adapter.tm if tm_adapter else None
            an_inst = an_adapter.an if an_adapter else None
            self.wm = WorldModel(
                activation_adapter.sa,
                causal_graph=cg,
                episodic_memory=em,
                analogy=an_inst,
                temporal=tm_inst,
                hormone_system=hormone_adapter.hs,
            )

    def enrich_meaning(self, meaning: Meaning):
        # task가 이미 있으면 skip — 산술/strlen이 우선
        if (meaning.context or {}).get("task") is not None:
            return
        text = meaning.raw_text or ""
        target = extract_describe_query(text)
        if target:
            meaning.context = meaning.context or {}
            meaning.context["wm_describe"] = target
            return
        rel = extract_relate_query(text)
        if rel:
            meaning.context = meaning.context or {}
            meaning.context["wm_relate"] = rel

    def describe_phrase(self, category: str) -> str | None:
        try:
            d = self.wm.describe_concept(category)
        except Exception:
            return None
        if not isinstance(d, dict):
            return None
        if not d.get("in_sa_graph"):
            return f"{category} 잘 모르겠는데"

        # 부모/자식 카테고리 (도메인 정보)
        parents = list((d.get("parents") or {}).keys())
        children = list((d.get("children") or {}).keys())
        episodes = d.get("episode_count", 0)

        parts: list[str] = []
        if parents:
            parts.append(f"{parents[0]} 같은 거에 속하는 거 같고")
        if children:
            child_str = ", ".join(children[:2])
            parts.append(f"{child_str}이랑 연결돼 있어")
        if episodes > 0:
            parts.append(f"전에 {episodes}번 정도 얘기했었지")

        if not parts:
            return f"{category}? 알긴 아는데 잘 정리는 안 됐어"
        return f"{category}는 " + ", ".join(parts)

    def relate_phrase(self, a: str, b: str) -> str | None:
        try:
            r = self.wm.relate(a, b)
        except Exception:
            return None
        if not isinstance(r, dict):
            return None
        strength = r.get("strength", 0.0)
        sa_w = r.get("sa_weight", 0.0)
        causal = r.get("causal")

        if strength < 0.05 and not causal:
            return f"{a}랑 {b}는 별 관련 없어 보여"
        if causal:
            return f"{a}는 {b}로 이어지는 관계인 거 같아"
        if sa_w > 0.5:
            return f"{a}랑 {b}는 자주 같이 떠올라"
        return f"{a}랑 {b}가 약하게 연결돼 있긴 해"

    def overview_phrase(self) -> str:
        try:
            o = self.wm.knowledge_overview()
        except Exception:
            return "아직 잘 모르겠어"
        n = o.get("total_categories", 0)
        domains = o.get("domains", {})
        top_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)[:3]
        if not top_domains:
            return f"카테고리 {n}개 정도 알아"
        domain_str = ", ".join(d for d, _ in top_domains)
        return f"카테고리 {n}개 정도 — 주로 {domain_str} 쪽 알아"
