"""
FrameAdapter

v40 FrameSemantics (Fillmore 1982 Frame Semantics) ↔ v41 SVO 추출.

흐름:
1. parse_sentence → {agent, patient, action, tense}
2. meaning.context["frame"] 박음
3. text_to_hg → 자동으로 HyperGraph에 저장 (장기 기억)
"""

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from frame_semantics import FrameSemantics

from utils.types import Meaning


class FrameAdapter:

    def __init__(self,
                 hormone_adapter,
                 activation_adapter,
                 nl_adapter,
                 hypergraph_adapter=None,
                 fs: FrameSemantics | None = None):
        self.hormone_adapter = hormone_adapter
        self.activation_adapter = activation_adapter
        self.nl_adapter = nl_adapter
        self.hypergraph_adapter = hypergraph_adapter

        if fs is not None:
            self.fs = fs
        else:
            hg = hypergraph_adapter.hg if hypergraph_adapter else None
            self.fs = FrameSemantics(
                natural_lang=nl_adapter.nl if nl_adapter else None,
                hypergraph=hg,
                spreading_activation=activation_adapter.sa,
                hormone_system=hormone_adapter.hs,
            )

    def enrich_meaning(self, meaning: Meaning):
        text = meaning.raw_text or ""
        try:
            frame = self.fs.parse_sentence(text)
        except Exception:
            return
        if not frame:
            return
        meaning.context = meaning.context or {}
        meaning.context["frame"] = {
            "agent":   self._role_filler(frame, "agent"),
            "patient": self._role_filler(frame, "patient"),
            "action":  frame.get("verb"),
            "tense":   frame.get("tense"),
        }

    def _role_filler(self, frame: dict, role: str) -> str | None:
        roles = frame.get("roles", {})
        for node, r in roles.items():
            if r == role:
                return node
        return None

    def store_to_hg(self, meaning: Meaning):
        """meaning의 frame을 HyperGraph에 저장 (장기 기억)."""
        if self.hypergraph_adapter is None:
            return
        text = meaning.raw_text or ""
        try:
            self.fs.text_to_hg(text)
        except Exception:
            pass

    def frame_phrase(self, meaning: Meaning) -> str | None:
        """프레임을 '~가 ~를 ~했다' 형태로."""
        f = (meaning.context or {}).get("frame")
        if not isinstance(f, dict):
            return None
        agent = f.get("agent")
        patient = f.get("patient")
        action = f.get("action")
        if not action:
            return None
        parts = []
        if agent:
            parts.append(f"{agent}가")
        if patient:
            parts.append(f"{patient}를")
        parts.append(f"{action}.")
        return " ".join(parts) if parts else None
