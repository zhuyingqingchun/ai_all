from __future__ import annotations

from typing import Any, TypedDict


class LangGraphPOCState(TypedDict, total=False):
    user_input: str
    plan: dict[str, Any]
    tool_results: list[dict[str, Any]]
    final_answer: str
    debug_trace: list[str]
