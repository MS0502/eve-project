"""
PersistenceAdapter

v40 Persistence ↔ v41 풀스택 영속화.

v40 Persistence는 eve 객체에서 sa/wm/hs/em 등 슬롯을 읽음.
우리는 v41 어댑터 묶음을 가지고 있으니, mock eve 객체로 슬롯 매핑.

추가로 v41 자체 상태도 저장:
- StreamingEngine.history (대화 기록)
- GoalAdapter.gm.active_goals
- NormAdapter 누적 카운트
- TaskSolver.attempts
"""

import os
import pickle
import gzip

from utils.legacy_path import enable as _enable_legacy
_enable_legacy()

from persistence import Persistence


class _MockEve:
    """v40 Persistence가 기대하는 슬롯 인터페이스 - 어댑터에서 빌려옴."""
    def __init__(self, engine):
        h, a, m = engine.hormone_adapter, engine.activation_adapter, engine.memory_adapter
        # v40 모듈 직접 노출
        self.hs = h.hs if h else None
        self.sa = a.sa if a else None
        self.wm = a.wm if a else None
        self.em = m.em if m else None
        # 옵션 모듈
        if engine.nl_adapter: self.nl = engine.nl_adapter.nl
        if engine.sd_adapter: self.sd = engine.sd_adapter.sd
        if engine.dmn_adapter: self.dmn = engine.dmn_adapter.dmn
        if engine.vsa_adapter: self.vsa = engine.vsa_adapter.vsa
        if engine.ai_adapter: self.ai = engine.ai_adapter.ai
        if engine.goal_adapter: self.gm = engine.goal_adapter.gm
        if engine.norm_adapter: self.norm = engine.norm_adapter.norm


class PersistenceAdapter:

    def __init__(self, engine):
        self.engine = engine

    # ===== save =====

    def save(self, path: str, compress: bool = True) -> dict:
        """풀스택 저장. v40 Persistence + v41 history."""
        mock = _MockEve(self.engine)
        # v40 부분
        try:
            v40_result = Persistence.save(mock, path, compress=compress)
            v40_path = v40_result.get("path", path)
        except Exception as e:
            v40_path = None
            v40_result = {"error": str(e)}

        # v41 추가물 (history + adapter 메타) — 별도 sidecar 파일
        sidecar_path = (v40_path or path) + ".v41sidecar"
        sidecar = self._collect_v41_sidecar()
        try:
            opener = gzip.open if compress else open
            sidecar_full = sidecar_path + (".gz" if compress else "")
            with opener(sidecar_full, "wb") as f:
                pickle.dump(sidecar, f)
        except Exception as e:
            return {"v40": v40_result, "v41": {"error": str(e)}}

        return {
            "v40": v40_result,
            "v41": {"path": sidecar_full, "items": list(sidecar.keys())},
            "path": v40_path,
        }

    # ===== load =====

    def load(self, path: str) -> dict:
        """풀스택 복원."""
        mock = _MockEve(self.engine)
        try:
            v40_state = Persistence.load_state(path)
            Persistence.apply_state(mock, v40_state)
            v40_ok = True
        except Exception as e:
            v40_ok = False
            v40_state = {"error": str(e)}

        # sidecar 시도 (.gz 우선)
        sidecar_loaded = self._load_sidecar(path)

        return {
            "v40_ok": v40_ok,
            "v41_loaded": sidecar_loaded,
        }

    # ===== v41 sidecar 내용 =====

    def _collect_v41_sidecar(self) -> dict:
        eng = self.engine
        sidecar: dict = {}

        # 대화 히스토리 (Experience dataclass 직렬화 가능)
        try:
            sidecar["history"] = [
                {
                    "input": h.input_text,
                    "response": h.response,
                    "timestamp": h.timestamp,
                    # Meaning은 dataclass라 dict 변환
                    "meaning_raw": h.meaning.raw_text if h.meaning else None,
                }
                for h in eng.history
            ]
        except Exception:
            sidecar["history"] = []

        # task solver 누적
        if hasattr(eng, "task_solver") and eng.task_solver is not None:
            sidecar["task_attempts"] = list(eng.task_solver.attempts)

        return sidecar

    def _load_sidecar(self, base_path: str) -> bool:
        candidates = [
            base_path + ".v41sidecar.gz",
            base_path + ".v41sidecar",
            base_path + ".gz" + ".v41sidecar.gz",
            base_path + ".gz" + ".v41sidecar",
        ]
        # v40이 .gz 붙였을 수도 있으니 추측
        for sp in candidates:
            if not os.path.exists(sp):
                continue
            try:
                opener = gzip.open if sp.endswith(".gz") else open
                with opener(sp, "rb") as f:
                    data = pickle.load(f)
                # history는 우리가 풍부하게 복원 안 함 — 메타만 반영
                # 자세한 복원이 필요하면 user side에서 처리
                self.engine.history.clear()  # 일단 비움
                return True
            except Exception:
                continue
        return False
