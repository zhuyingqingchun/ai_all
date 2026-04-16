from __future__ import annotations

import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import VLLMChatClient
from .config import ANSWER_MAX_TOKENS, DEFAULT_RESULTS_JSON, DEFAULT_TRACE_JSONL, PLANNING_MAX_TOKENS
from .memory import maybe_summarize
from .prompts import FINAL_SYSTEM_PROMPT, TOOL_SYSTEM_PROMPT
from .routing import safe_parse_plan
from .schemas import ConversationMemory, ConversationState, TurnTrace
from .tools import LocalTools


def tool_match(actual_tool: str, turn_spec: Dict[str, Any]) -> bool:
    acceptable_tools = turn_spec.get("acceptable_tools")
    if acceptable_tools is not None:
        return actual_tool in acceptable_tools
    expected_tool = turn_spec.get("expected_tool")
    if expected_tool is not None:
        return actual_tool == expected_tool
    return True


def check_contains_all(answer: str, items: List[str]) -> bool:
    return all(item in answer for item in items)


def check_contains_any(answer: str, items: List[str]) -> bool:
    return any(item in answer for item in items)


def classify_failure(trace: TurnTrace, turn_spec: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    tool = trace.plan.get("tool")
    if not tool_match(tool, turn_spec):
        return False, "route_error", f"tool mismatch, actual_tool={tool}"
    if tool in {"get_weather", "get_time", "calculator"} and trace.tool_result is not None and trace.tool_result.get("ok") is False:
        contains_any = turn_spec.get("expected_contains_any")
        if contains_any is not None and check_contains_any(trace.answer, contains_any):
            return True, None, None
        return False, "tool_error", trace.tool_result.get("error")
    contains_all = turn_spec.get("expected_contains_all")
    if contains_all is not None and not check_contains_all(trace.answer, contains_all):
        return False, ("memory_error" if "memory" in turn_spec.get("tags", []) else "answer_error"), f"missing all keywords: {contains_all}"
    contains_any = turn_spec.get("expected_contains_any")
    if contains_any is not None and not check_contains_any(trace.answer, contains_any):
        return False, ("memory_error" if "memory" in turn_spec.get("tags", []) else "answer_error"), f"missing any keywords: {contains_any}"
    return True, None, None


def evaluate_trace(trace: TurnTrace, turn_spec: Dict[str, Any]) -> TurnTrace:
    trace.passed, trace.failure_type, trace.failure_detail = classify_failure(trace, turn_spec)
    return trace


class AgentEvaluator:
    def __init__(self, chat_client: VLLMChatClient, tools: Optional[LocalTools] = None, debug: bool = False):
        self.chat_client = chat_client
        self.tools = tools or LocalTools()
        self.debug = debug

    def run_turn(self, session_name: str, turn_index: int, turn_spec: Dict[str, Any], memory: ConversationMemory, state: ConversationState) -> TurnTrace:
        user_input = turn_spec["user"]
        trace = TurnTrace(session_name=session_name, turn_index=turn_index, user=user_input, tags=turn_spec.get("tags", []), expected_tool=turn_spec.get("expected_tool"), acceptable_tools=turn_spec.get("acceptable_tools"))
        context = memory.build_context()
        planning_messages = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}, *context, {"role": "user", "content": user_input}]
        trace.plan_raw, trace.planning_latency_ms = self.chat_client.chat(planning_messages, max_tokens=PLANNING_MAX_TOKENS)
        trace.plan = safe_parse_plan(trace.plan_raw, user_input, state)
        if self.debug:
            print(f"\n=== {session_name} / turn {turn_index} ===")
            print("PLAN RAW:", trace.plan_raw)
            print("PLAN PARSED:", json.dumps(trace.plan, ensure_ascii=False))
        if trace.plan["tool"] == "direct_answer":
            final_messages = [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT}, *context,
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"路由结果：{json.dumps(trace.plan, ensure_ascii=False)}"},
                {"role": "user", "content": "请基于当前问题、历史上下文和路由结果生成最终答复。"},
            ]
            trace.answer, trace.answer_latency_ms = self.chat_client.chat(final_messages, max_tokens=ANSWER_MAX_TOKENS)
            memory.add_turn(user_input, trace.answer)
            trace.summary_triggered, trace.summary_latency_ms = maybe_summarize(memory, self.chat_client)
            trace.memory_summary = memory.summary
            state.update(trace.plan)
            return trace
        trace.tool_result, trace.tool_latency_ms = self.tools.call(trace.plan["tool"], trace.plan.get("args", {}))
        final_messages = [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT}, *context,
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(trace.plan, ensure_ascii=False)},
            {"role": "user", "content": f"工具名：{trace.plan['tool']}\n工具参数：{json.dumps(trace.plan.get('args', {}), ensure_ascii=False)}\n工具返回：{json.dumps(trace.tool_result, ensure_ascii=False)}\n请基于上述信息回答用户。"},
        ]
        trace.answer, trace.answer_latency_ms = self.chat_client.chat(final_messages, max_tokens=ANSWER_MAX_TOKENS)
        memory.add_turn(user_input, trace.answer)
        trace.summary_triggered, trace.summary_latency_ms = maybe_summarize(memory, self.chat_client)
        trace.memory_summary = memory.summary
        state.update(trace.plan)
        return trace

    def run_dataset(self, dataset: Sequence[Dict[str, Any]]) -> List[TurnTrace]:
        traces: List[TurnTrace] = []
        for session in dataset:
            memory = ConversationMemory()
            state = ConversationState()
            for index, turn_spec in enumerate(session["turns"], start=1):
                traces.append(evaluate_trace(self.run_turn(session["session_name"], index, turn_spec, memory, state), turn_spec))
        return traces


def aggregate_stats(traces: List[TurnTrace]) -> Dict[str, Any]:
    total = len(traces)
    passed = sum(1 for t in traces if t.passed)
    by_tool: Dict[str, Dict[str, float]] = {}
    by_tag: Dict[str, Dict[str, float]] = {}
    by_failure_type: Dict[str, int] = {}
    by_route_source: Dict[str, Dict[str, float]] = {}
    for trace in traces:
        tool = trace.plan.get("tool", "unknown")
        by_tool.setdefault(tool, {"total": 0, "passed": 0})
        by_tool[tool]["total"] += 1
        by_tool[tool]["passed"] += int(trace.passed)
        route_source = trace.plan.get("route_source", "unknown")
        by_route_source.setdefault(route_source, {"total": 0, "passed": 0})
        by_route_source[route_source]["total"] += 1
        by_route_source[route_source]["passed"] += int(trace.passed)
        if trace.failure_type:
            by_failure_type[trace.failure_type] = by_failure_type.get(trace.failure_type, 0) + 1
        for tag in trace.tags:
            by_tag.setdefault(tag, {"total": 0, "passed": 0})
            by_tag[tag]["total"] += 1
            by_tag[tag]["passed"] += int(trace.passed)
    def with_rates(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        result = {}
        for key, value in stats.items():
            result[key] = {**value, "pass_rate": round(value["passed"] / value["total"], 4) if value["total"] else 0.0}
        return result
    return {
        "total_turns": total,
        "passed_turns": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_latency_ms": {
            "planning": round(statistics.mean(t.planning_latency_ms for t in traces), 2) if traces else 0.0,
            "tool": round(statistics.mean(t.tool_latency_ms for t in traces), 2) if traces else 0.0,
            "answer": round(statistics.mean(t.answer_latency_ms for t in traces), 2) if traces else 0.0,
            "summary": round(statistics.mean(t.summary_latency_ms for t in traces), 2) if traces else 0.0,
        },
        "by_tool": with_rates(by_tool),
        "by_tag": with_rates(by_tag),
        "by_failure_type": by_failure_type,
        "by_route_source": with_rates(by_route_source),
    }


def save_outputs(traces: List[TurnTrace], summary: Dict[str, Any], results_path: Path = DEFAULT_RESULTS_JSON, trace_path: Path = DEFAULT_TRACE_JSONL) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with trace_path.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")
