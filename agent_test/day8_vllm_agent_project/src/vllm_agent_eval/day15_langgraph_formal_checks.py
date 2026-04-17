from __future__ import annotations

from typing import Any

from .assertion_matchers import evaluate_text_assertions


def _flatten_result_text(result: dict[str, Any]) -> str:
    try:
        return str(result)
    except Exception:
        return ""


def _tools_present(result: dict[str, Any], expected_tools: list[str]) -> bool:
    blob = _flatten_result_text(result)
    return all(tool in blob for tool in expected_tools)


def evaluate_langgraph_formal_turn(result: dict[str, Any], turn_spec: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    answer = result.get("final_answer", "")
    assertion_result = evaluate_text_assertions(
        answer=answer,
        turn_spec=turn_spec,
        failure_type="answer_error",
    )
    if not assertion_result.passed:
        return False, assertion_result.failure_type, assertion_result.failure_detail

    acceptable_tool_sequences = turn_spec.get("acceptable_tool_sequences")
    if acceptable_tool_sequences:
        matched = any(_tools_present(result, seq) for seq in acceptable_tool_sequences)
        if not matched:
            return False, "plan_error", f"acceptable_tool_sequences not matched: {acceptable_tool_sequences}"

    return True, None, None
