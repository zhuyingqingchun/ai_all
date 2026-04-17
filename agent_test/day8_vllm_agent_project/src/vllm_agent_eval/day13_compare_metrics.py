from __future__ import annotations

from typing import Any


def compute_langgraph_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    answer_nonempty_count = 0
    debug_trace_count = 0
    tool_mentions = {
        "get_weather": 0,
        "get_time": 0,
        "calculator": 0,
    }

    for item in results:
        result = item.get("result", {})
        answer = result.get("final_answer", "")
        if isinstance(answer, str) and answer.strip():
            answer_nonempty_count += 1

        debug_trace = result.get("debug_trace", [])
        if isinstance(debug_trace, list) and len(debug_trace) > 0:
            debug_trace_count += 1

        text_blob = ""
        try:
            text_blob = str(result)
        except Exception:
            text_blob = ""

        for key in tool_mentions:
            if key in text_blob:
                tool_mentions[key] += 1

    return {
        "langgraph_answer_nonempty_count": answer_nonempty_count,
        "langgraph_debug_trace_count": debug_trace_count,
        "heuristic_tool_mentions": tool_mentions,
    }
