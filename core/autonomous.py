"""
AutonomousLoop — GPT 라운드 14 핵심.

EVE가 idle 시간 동안 스스로 한 사이클:
1. internal tick — SA decay + 호르몬 변동 + DMN 진행
2. think — ReasoningLoop으로 무엇 생각할지
3. evaluate need — 호르몬 결핍 / 활성 목표 → 후보 행동
4. decide — Agency로 do/hesitate
5. emit — proactive_stream으로 표현 (or silent)
6. attribute — self 귀속 (자아감)

원칙:
- background thread 0 (외부에서 step() 호출)
- 결정론 (random 0)
- step() 한 번 = 한 사이클 = "한 호흡"

REPL 통합:
    while True:
        line = read_with_timeout(N초)
        if line:
            engine.chat_stream(line)
        else:
            engine.autonomous.step()  # 자율 한 호흡
"""

from utils.types import Meaning


# 호르몬 결핍 → 후보 행동 (NEED → ACTION 매핑)
NEED_TO_ACTIONS: dict[str, list[str]] = {
    "low_oxytocin":   ["대화", "공감"],         # ot 낮음 → 사회 연결
    "low_dopamine":   ["탐색", "학습"],         # da 낮음 → 보상 추구
    "low_serotonin":  ["휴식", "회복"],         # 5HT 낮음 → 안정
    "high_cortisol":  ["진정", "쉬다"],         # cort 높음 → 진정
    "high_glutamate": ["휴식", "정리"],         # 인지 부하 높음
}


def _detect_needs(hormone_adapter) -> list[str]:
    """호르몬 상태에서 need 라벨들."""
    if hormone_adapter is None:
        return []
    hs = hormone_adapter.hs
    needs: list[str] = []
    if hs.hormones["oxytocin"].level < 0.25:
        needs.append("low_oxytocin")
    if hs.hormones["dopamine"].level < 0.25:
        needs.append("low_dopamine")
    if hs.hormones["serotonin"].level < 0.35:
        needs.append("low_serotonin")
    if hs.hormones["cortisol"].level > 0.7:
        needs.append("high_cortisol")
    if hs.hormones["glutamate"].level > 0.7:
        needs.append("high_glutamate")
    return needs


class AutonomousLoop:
    """idle 시간 동안 EVE의 자율 사이클."""

    def __init__(self, engine):
        """engine: StreamingEngine (어댑터들 다 들고 있음)"""
        self.engine = engine
        self.step_count = 0
        self.last_emit_step = -1
        self.history: list[dict] = []      # step별 trace

    def step(self, dt: float = 1.0, emit: bool = True) -> dict:
        """
        한 자율 사이클 실행.
        
        emit=True면 proactive_stream으로 발화 가능, False면 silent (사고만).

        Returns:
            {step, needs, action, decision, emitted, response}
        """
        self.step_count += 1
        result: dict = {
            "step": self.step_count,
            "needs": [],
            "action": None,
            "decision": None,
            "emitted": False,
            "response": "",
            "thought": None,
        }

        # 1) need 감지 (현재 상태 기준)
        needs = _detect_needs(self.engine.hormone_adapter)
        result["needs"] = needs

        # 2) need에서 후보 행동 추출
        candidates: list[str] = []
        for need in needs:
            for action in NEED_TO_ACTIONS.get(need, []):
                if action not in candidates:
                    candidates.append(action)

        # 활성 목표도 후보에 추가
        if self.engine.goal_adapter is not None:
            for cat in self.engine.goal_adapter.active_goal_categories():
                if cat not in candidates:
                    candidates.append(cat)

        # 3) 시간 흐름 (행동 결정 후 — Agency가 호르몬 보고 결정)
        self._internal_tick(dt)

        if not candidates:
            self.history.append(result)
            return result    # 할 거 없음 — silent

        # 3.5) 외출/복귀 자율 결정 (env_adapter 있을 때)
        if self.engine.env_adapter is not None:
            destination = self.engine.env_adapter.decide_outing_destination()
            if destination:
                if self.engine.env_adapter.go_out(destination):
                    result["action"] = f"외출_{destination}"
                    result["decision"] = "do"
                    self.history.append(result)
                    return result
            elif self.engine.env_adapter.decide_return_home():
                if self.engine.env_adapter.return_home():
                    result["action"] = "복귀"
                    result["decision"] = "do"
                    self.history.append(result)
                    return result

        # 4) Agency로 결정 (있으면)
        action = None
        decision_info = None
        if hasattr(self.engine, "agency_adapter") and self.engine.agency_adapter is not None:
            picked = self.engine.agency_adapter.best_action(candidates)
            if picked is not None:
                action, decision_info = picked
        else:
            # fallback: 첫 후보
            action = candidates[0]

        if action is None:
            result["decision"] = "hesitate"
            self.history.append(result)
            return result

        result["action"] = action
        result["decision"] = "do"

        # 5) ReasoningLoop으로 thought 생성 (silent 사고)
        thought_phrase = self._silent_thought(action)
        result["thought"] = thought_phrase

        # 6) emit — proactive_stream
        if emit and self._should_emit():
            response_chunks = self._emit_via_proactive(action, thought_phrase)
            if response_chunks:
                result["emitted"] = True
                result["response"] = " ".join(response_chunks)
                self.last_emit_step = self.step_count

        # 7) attribute (Agency 있으면)
        if (hasattr(self.engine, "agency_adapter")
                and self.engine.agency_adapter is not None and action):
            self.engine.agency_adapter.attribute(action, source="self")

        # 8) Curiosity — 자율 검색 시도 (호르몬 curiosity 높을 때만)
        if hasattr(self.engine, "curiosity_adapter") and self.engine.curiosity_adapter is not None:
            try:
                curio_result = self.engine.curiosity_adapter.explore_in_autonomous_step()
                if curio_result:
                    result["curiosity"] = curio_result
            except Exception:
                pass

        self.history.append(result)
        return result

    # ===== helpers =====

    def _internal_tick(self, dt: float):
        """SA/WM/hormone/DMN 등 내부 시간 진행."""
        eng = self.engine
        if eng.activation_adapter is not None:
            try:
                eng.activation_adapter.tick(dt=dt)
            except Exception:
                pass
        if eng.hormone_adapter is not None:
            try:
                eng.hormone_adapter.tick(dt=dt)
            except Exception:
                pass
        if eng.dmn_adapter is not None:
            try:
                eng.dmn_adapter.tick(dt=dt)
            except Exception:
                pass
        if eng.continual_adapter is not None:
            eng.continual_adapter.tick(dt=dt)
        if eng.allostatic_adapter is not None:
            eng.allostatic_adapter.tick(dt=dt)

    def _silent_thought(self, action: str) -> str | None:
        """ReasoningLoop 활용 — 행동 후보에 대한 짧은 생각."""
        if self.engine.reasoning_loop is None:
            return None
        # 가짜 meaning 만들어서 ReasoningLoop 돌림
        meaning = Meaning(
            intent="proactive",
            entities=[action],
            emotions={},
            context={"autonomous": True},
            raw_text=f"[자율: {action}]",
        )
        from cognition.meaning_graph import build_graph
        self.engine.workspace.update(build_graph(meaning), meaning)
        state = self.engine.workspace.get_state()
        thought = self.engine.reasoning_loop.think(state)
        return thought.content if thought else None

    def _should_emit(self) -> bool:
        """발화 조건 — autonomy cooldown 체크."""
        if self.engine.autonomy_adapter is None:
            return True
        return self.engine.autonomy_adapter.is_due()

    def _emit_via_proactive(self, action: str, thought: str | None) -> list[str]:
        """proactive_stream을 활용 (DMN 자발 발화 채널)."""
        if self.engine.dmn_adapter is None:
            return []
        # DMN이 자체 카테고리 결정 (action 기반은 아님)
        try:
            chunks = list(self.engine.proactive_stream(force=True))
        except Exception:
            chunks = []
        return chunks

    def get_state(self) -> dict:
        return {
            "step_count": self.step_count,
            "last_emit_step": self.last_emit_step,
            "recent_actions": [h["action"] for h in self.history[-5:] if h["action"]],
        }
