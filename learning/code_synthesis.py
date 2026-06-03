"""
TaskSolver — eve1.md 12절 코드 생성 루프 (LLM 없이).

while not solved:
    code = generate_step()      ← 도구 매칭
    result = execute(code)       ← 도구 실행
    if error:
        refine()                 ← 다른 도구로 fallback

결정론:
- 도구 순서대로 시도 (random 0)
- 같은 task → 같은 결과
- max_attempts로 무한 루프 방지
"""

from agent.tools import ToolRegistry, ToolError


class TaskSolver:

    def __init__(self, registry: ToolRegistry | None = None, max_attempts: int = 3):
        self.registry = registry if registry is not None else ToolRegistry()
        self.max_attempts = max_attempts
        # 학습/디버깅용 누적
        self.attempts: list[dict] = []

    def solve(self, task: dict) -> dict:
        """
        task → {ok: bool, result: str, tool: str, error: str, attempts: int}
        """
        last_error: str | None = None
        tried: set[str] = set()

        for attempt in range(self.max_attempts):
            tool = self.registry.find(task)
            if tool is None:
                last_error = "처리 가능한 도구 없음"
                break
            if tool.name in tried:
                # 같은 도구를 반복 매칭해서 무한루프 방지
                last_error = f"도구 {tool.name} 반복 매칭"
                break
            tried.add(tool.name)
            try:
                result = tool.run(task)
                self.attempts.append({"task": task, "tool": tool.name, "ok": True})
                return {
                    "ok": True,
                    "result": result,
                    "tool": tool.name,
                    "error": None,
                    "attempts": attempt + 1,
                }
            except ToolError as e:
                last_error = str(e)
                self.attempts.append({
                    "task": task, "tool": tool.name, "ok": False, "error": last_error,
                })
                # refine — 다음 attempt에서 다른 도구 찾을 기회
                # (현 단순 버전은 같은 task로 다른 도구 매칭 가능하면 시도)
                continue

        return {
            "ok": False,
            "result": None,
            "tool": None,
            "error": last_error or "unknown",
            "attempts": self.max_attempts,
        }
