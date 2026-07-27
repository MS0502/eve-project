from types import SimpleNamespace

import pytest

from adapters.user_instruction_adapter import UserInstructionAdapter
from language.streaming import StreamingEngine


ACTUAL_STRESS_LOAD_INPUT_WITH_MODIFIER = "지금 떠오르는 생각 하나를 짧게 말해줘."
ACTUAL_STRESS_LOAD_FOLLOWUP_INPUT = "방금 말한 생각과 연결되는 이유를 한 가지 설명해줘."


class RoutedToLanguageUnderstanding(RuntimeError):
    pass


def _engine_with_route_probe():
    engine = StreamingEngine(
        delays={"stage1": 0, "core": 0, "expand": 0, "final": 0}
    )
    instruction = UserInstructionAdapter()
    engine.planner._engine_ref = SimpleNamespace(user_instruction=instruction)

    seen = []

    def route_probe(text):
        seen.append(text)
        raise RoutedToLanguageUnderstanding(text)

    engine.lu.parse = route_probe
    return engine, instruction, seen


def test_actual_content_bearing_short_modifier_reaches_normal_pipeline():
    engine, instruction, seen = _engine_with_route_probe()

    with pytest.raises(RoutedToLanguageUnderstanding):
        list(engine.chat_stream(ACTUAL_STRESS_LOAD_INPUT_WITH_MODIFIER))

    assert seen == [ACTUAL_STRESS_LOAD_INPUT_WITH_MODIFIER]
    assert instruction.is_short_mode() is True


def test_actual_followup_input_reaches_normal_pipeline():
    engine, instruction, seen = _engine_with_route_probe()

    with pytest.raises(RoutedToLanguageUnderstanding):
        list(engine.chat_stream(ACTUAL_STRESS_LOAD_FOLLOWUP_INPUT))

    assert seen == [ACTUAL_STRESS_LOAD_FOLLOWUP_INPUT]
    assert instruction.is_short_mode() is False


def test_meta_only_short_instruction_keeps_early_return():
    engine, instruction, seen = _engine_with_route_probe()

    chunks = list(engine.chat_stream("짧게 말해줘."))

    assert chunks == ["응. 짧게 답할게."]
    assert seen == []
    assert instruction.is_short_mode() is True
