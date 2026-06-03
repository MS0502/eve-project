"""
Tools — EVE가 task 해결할 때 호출하는 결정론적 도구.

v40 tool_use는 메타 레벨(도구 등록/효과 학습)만 있고 실제 도구 X.
v41 라운드 6는 진짜 도구 본체.

원칙:
- 모든 도구는 결정론적 (random 0)
- 실행 안전성 (eval 금지, AST 화이트리스트)
- 실패하면 ToolError raise (TaskSolver가 다른 도구 시도)
"""

import ast
import operator as op


class ToolError(Exception):
    """도구 실행 실패 (입력 형식, 안전 위반, 계산 오류 등)."""


class Tool:
    """베이스 인터페이스."""
    name: str = "tool"
    description: str = ""

    def can_handle(self, task: dict) -> bool:
        raise NotImplementedError

    def run(self, task: dict) -> str:
        """결과를 사람이 읽는 한국어 짧은 문장으로."""
        raise NotImplementedError


# =====================================================================
# Calculator — ast 기반 안전 산술
# =====================================================================

# 화이트리스트만. eval/exec 절대 안 씀.
_BIN_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_UNARY_OPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def safe_eval_arith(expr: str) -> float | int:
    """ast 화이트리스트로만 산술 평가. 다른 노드 나오면 ToolError."""
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError as e:
        raise ToolError(f"문법 오류: {e}")
    return _eval_node(node)


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ToolError(f"상수 타입 안 됨: {type(node.value).__name__}")
    if isinstance(node, ast.BinOp):
        opfn = _BIN_OPS.get(type(node.op))
        if opfn is None:
            raise ToolError(f"이항 연산 안 됨: {type(node.op).__name__}")
        try:
            return opfn(_eval_node(node.left), _eval_node(node.right))
        except ZeroDivisionError:
            raise ToolError("0으로 나누기")
    if isinstance(node, ast.UnaryOp):
        opfn = _UNARY_OPS.get(type(node.op))
        if opfn is None:
            raise ToolError(f"단항 연산 안 됨: {type(node.op).__name__}")
        return opfn(_eval_node(node.operand))
    raise ToolError(f"AST 노드 안 됨: {type(node).__name__}")


class Calculator(Tool):
    name = "calculator"
    description = "산술 계산 (+, -, *, /, //, %, **)"

    def can_handle(self, task: dict) -> bool:
        return task.get("type") == "arithmetic" and "expr" in task

    def run(self, task: dict) -> str:
        expr = task["expr"]
        result = safe_eval_arith(expr)
        # 정수면 정수로
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"그건 {result}이야"


# =====================================================================
# StringLength
# =====================================================================

class StringLength(Tool):
    name = "string_length"
    description = "문자열 길이"

    def can_handle(self, task: dict) -> bool:
        return task.get("type") == "string_length" and "target" in task

    def run(self, task: dict) -> str:
        target = task["target"]
        if not isinstance(target, str):
            raise ToolError("target이 문자열 아님")
        return f"'{target}' 글자 수는 {len(target)}개야"


# =====================================================================
# Registry
# =====================================================================

DEFAULT_TOOLS: list[Tool] = [
    Calculator(),
    StringLength(),
]


class ToolRegistry:

    def __init__(self, tools: list[Tool] | None = None):
        self.tools: list[Tool] = list(tools if tools is not None else DEFAULT_TOOLS)

    def find(self, task: dict) -> Tool | None:
        for t in self.tools:
            if t.can_handle(task):
                return t
        return None

    def register(self, tool: Tool):
        self.tools.append(tool)
