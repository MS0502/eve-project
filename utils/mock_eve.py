"""
_MockEve

v40 모듈들 (IntegratedSelf, UserPresence, Persistence)이 기대하는
eve 객체 인터페이스를 v41 어댑터들에서 빌려와 노출.

v40 EVE 클래스는 sa, wm, hs, em, cg, ag, dmn, sd, ai, ds, ng 등 슬롯 직접.
v41은 어댑터로 감쌌으니 mock 객체 하나로 슬롯 재노출.
"""

from typing import Set


class MockEve:
    """v40 eve 인터페이스 모방."""

    def __init__(self, engine):
        e = engine
        # 핵심 v40 모듈
        self.sa = e.activation_adapter.sa if e.activation_adapter else None
        self.wm = e.activation_adapter.wm if e.activation_adapter else None
        self.hs = e.hormone_adapter.hs if e.hormone_adapter else None
        self.em = e.memory_adapter.em if e.memory_adapter else None

        # 옵션 모듈 (있을 때만)
        if e.nl_adapter: self.nl = e.nl_adapter.nl
        if e.sd_adapter: self.sd = e.sd_adapter.sd
        if e.dmn_adapter: self.dmn = e.dmn_adapter.dmn
        if e.vsa_adapter: self.vsa = e.vsa_adapter.vsa
        if e.ai_adapter: self.ai = e.ai_adapter.ai
        if e.goal_adapter: self.gm = e.goal_adapter.gm
        if e.norm_adapter: self.norm = e.norm_adapter.norm
        if e.continual_adapter: self.cr = e.continual_adapter.cr
        if e.allostatic_adapter: self.al = e.allostatic_adapter.al
        if e.counterfactual_adapter: self.cf = e.counterfactual_adapter.cf
        if e.analogy_adapter: self.an = e.analogy_adapter.an
        if e.temporal_adapter: self.tm = e.temporal_adapter.tm
        if e.humor_adapter: self.humor = e.humor_adapter.humor
        if e.suffering_adapter: self.suffering = e.suffering_adapter.suffering
        if e.narrative_adapter: self.ns = e.narrative_adapter.ns
        if e.creative_adapter: self.creative = e.creative_adapter.cr
        if e.metacognition_adapter: self.mc = e.metacognition_adapter.mc
        if e.deep_reasoning_adapter: self.dr = e.deep_reasoning_adapter.dr
        if e.world_model_adapter: self.world_model = e.world_model_adapter.wm
        if e.agency_adapter: self.ag = e.agency_adapter.ag
        if hasattr(e, "digital_somatic_adapter") and e.digital_somatic_adapter:
            self.ds = e.digital_somatic_adapter.ds

        # CG (causal graph) — sim/cf/dr 어떤 거든 빌림
        cg = None
        if e.simulation_engine and e.simulation_engine.cg:
            cg = e.simulation_engine.cg
        elif e.counterfactual_adapter:
            cg = e.counterfactual_adapter.cf.cg
        elif e.deep_reasoning_adapter:
            cg = e.deep_reasoning_adapter.dr.cg
        if cg is not None:
            self.cg = cg

        # 보호 카테고리 (Continual에서)
        if e.continual_adapter:
            try:
                self.protected_categories = e.continual_adapter.protected_categories()
            except Exception:
                self.protected_categories = set()
        else:
            self.protected_categories: Set[str] = set()

        # sim_hour (env에 있으면)
        self.sim_hour = 0.0

        # 환경 정보
        if e.env_adapter:
            self._env_adapter = e.env_adapter
