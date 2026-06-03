"""
SimulationEngine — GPT 라운드 12: world_model + simulation.

원칙:
- 결정론적 (random 0)
- 인과 그래프 walk → step-by-step 시나리오
- 각 step의 호르몬 변화 예측

흐름:
1. simulate(seed_category, depth=N)
2. CG에서 cause→effect 체인 따라감
3. 각 노드에서 호르몬 영향 누적 (Active Inference 활용 가능)
4. 결과: list of {step, category, hormone_delta}

v41 통합:
- "X 하면 어떻게 돼?" 같은 질문 → simulate
- planner가 결과를 step별 sub_points로 표현
"""

from typing import Any


class SimulationEngine:

    def __init__(self,
                 activation_adapter,
                 hormone_adapter,
                 cf_adapter=None,
                 ai_adapter=None,
                 dr_adapter=None):
        self.sa = activation_adapter.sa
        self.hs = hormone_adapter.hs
        self.cf = cf_adapter
        self.ai = ai_adapter
        self.dr = dr_adapter
        # CG는 cf_adapter 또는 dr_adapter가 들고 있는 거 빌림
        self.cg = None
        if cf_adapter is not None:
            self.cg = cf_adapter.cf.cg
        elif dr_adapter is not None:
            self.cg = dr_adapter.dr.cg

    def simulate(self, seed: str, depth: int = 3) -> list[dict[str, Any]]:
        """
        seed에서 시작해 인과 그래프 따라 N step.
        결정론: 각 step마다 가장 강한 effect edge 선택.

        Returns:
            [{step:0, category:seed, source:"seed"}, {step:1, category:..., weight:...}, ...]
        """
        steps: list[dict] = [{"step": 0, "category": seed, "source": "seed"}]
        if self.cg is None:
            return steps
        if depth <= 0:
            return steps

        current = seed
        visited: set[str] = {seed}

        for i in range(1, depth + 1):
            next_node, weight = self._strongest_effect(current, visited)
            if next_node is None:
                break
            steps.append({
                "step": i,
                "category": next_node,
                "weight": weight,
                "source": current,
            })
            visited.add(next_node)
            current = next_node

        return steps

    def _strongest_effect(self, node: str, visited: set[str]) -> tuple[str | None, float]:
        """node의 effect 중 가장 강한 거 (이미 방문한 건 제외)."""
        cg = self.cg
        if cg is None:
            return (None, 0.0)
        # v40 CausalGraph.children(node) → {child: weight}
        try:
            children = cg.children(node) if hasattr(cg, "children") else {}
        except Exception:
            children = {}

        best_node = None
        best_weight = 0.0
        for child, weight in children.items():
            if child in visited:
                continue
            if weight > best_weight:
                best_node = child
                best_weight = weight
        return (best_node, best_weight)

    def simulate_phrase(self, seed: str, depth: int = 3) -> str | None:
        """시뮬레이션 결과를 한 줄 한국어로."""
        steps = self.simulate(seed, depth=depth)
        if len(steps) < 2:
            return None
        chain = " → ".join(s["category"] for s in steps)
        return f"{chain}로 흘러갈 거 같아"

    def simulate_sub_points(self, seed: str, depth: int = 3) -> list[str]:
        """시뮬레이션을 step별 sub_points로."""
        steps = self.simulate(seed, depth=depth)
        if len(steps) < 2:
            return []
        out: list[str] = []
        for s in steps[1:]:
            out.append(f"{s['source']} → {s['category']}")
        return out
