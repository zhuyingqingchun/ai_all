from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .client import VLLMChatClient
from .config import ANSWER_MAX_TOKENS, DEFAULT_RESULTS_JSON, DEFAULT_TRACE_JSONL, PLANNING_MAX_TOKENS
from .memory import maybe_summarize
from .multistep_patch_helpers import build_salvage_multistep_plan, classify_clause_to_step, split_multistep_clauses
from .multistep_prompts import MULTISTEP_FINAL_SYSTEM_PROMPT, MULTISTEP_PLANNER_SYSTEM_PROMPT
from .multistep_schemas import MultiStepTurnTrace
from .routing import extract_json, safe_parse_plan
from .schemas import ConversationMemory, ConversationState
from .tools import LocalTools


SUPPORTED_CITIES = {
    "东京": {"timezone": "Asia/Tokyo"},
    "北京": {"timezone": "Asia/Shanghai"},
    "上海": {"timezone": "Asia/Shanghai"},
    "广州": {"timezone": "Asia/Shanghai"},
    "深圳": {"timezone": "Asia/Shanghai"},
    "纽约": {"timezone": "America/New_York"},
    "洛杉矶": {"timezone": "America/Los_Angeles"},
    "旧金山": {"timezone": "America/Los_Angeles"},
}


def parse_multistep_json(plan_text: str) -> Dict[str, Any]:
    data = extract_json(plan_text)
    if not isinstance(data, dict):
        raise ValueError("planner output is not dict")
    steps = data.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("steps must be list")
    normalized_steps = []
    for step in steps[:3]:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool")
        args = step.get("args", {})
        if tool not in {"get_weather", "get_time", "calculator", "direct_answer"}:
            continue
        if not isinstance(args, dict):
            args = {}
        normalized_steps.append({
            "tool": tool,
            "args": args,
            "purpose": step.get("purpose", ""),
        })
    return {
        "steps": normalized_steps,
        "final_instruction": data.get("final_instruction", "") or "",
    }


def model_multistep_plan(chat_client: VLLMChatClient, context: List[Dict[str, str]], user_input: str) -> Tuple[Dict[str, Any], str, float]:
    planner_messages = [
        {"role": "system", "content": MULTISTEP_PLANNER_SYSTEM_PROMPT},
        *context,
        {"role": "user", "content": user_input},
    ]
    plan_raw, latency_ms = chat_client.chat(planner_messages, max_tokens=PLANNING_MAX_TOKENS)
    parsed = parse_multistep_json(plan_raw)
    parsed["route_source"] = "model_multistep_json"
    return parsed, plan_raw, latency_ms


def build_multistep_plan(chat_client: VLLMChatClient, context: List[Dict[str, str]], user_input: str, state: ConversationState) -> Tuple[Dict[str, Any], str, float]:
    weather_cities = SUPPORTED_CITIES
    time_cities = SUPPORTED_CITIES

    clauses, final_instruction = split_multistep_clauses(user_input)
    steps: List[Dict[str, Any]] = []

    for clause in clauses:
        step = classify_clause_to_step(clause, weather_cities, time_cities)
        if step is not None:
            steps.append(step)

    if steps:
        return {
            "steps": steps,
            "final_instruction": final_instruction or "请按顺序给出每一步结果。",
            "route_source": "heuristic_multistep_patch",
        }, json.dumps({"steps": steps, "final_instruction": final_instruction}, ensure_ascii=False), 0.0

    try:
        parsed, raw, latency = model_multistep_plan(chat_client, context, user_input)
        if parsed.get("steps"):
            return parsed, raw, latency
    except Exception:
        pass

    salvage_plan = build_salvage_multistep_plan(user_input, weather_cities, time_cities)
    if salvage_plan.get("steps"):
        return salvage_plan, json.dumps(salvage_plan, ensure_ascii=False), 0.0

    return {
        "steps": [],
        "final_instruction": user_input,
        "route_source": "fallback_single_direct_answer",
    }, json.dumps({"steps": [], "final_instruction": user_input}, ensure_ascii=False), 0.0


def actual_tool_sequence(trace: MultiStepTurnTrace) -> List[str]:
    return [step["tool"] for step in trace.plan.get("steps", [])]


def tool_sequence_match(actual: List[str], turn_spec: Dict[str, Any]) -> bool:
    acceptable = turn_spec.get("acceptable_tool_sequences")
    if acceptable is not None:
        return actual in acceptable
    expected = turn_spec.get("expected_tools_sequence")
    if expected is not None:
        return actual == expected
    return True


def check_contains_all(answer: str, items: List[str]) -> bool:
    return all(item in answer for item in items)


def check_contains_any(answer: str, items: List[str]) -> bool:
    return any(item in answer for item in items)


def classify_multistep_failure(trace: MultiStepTurnTrace, turn_spec: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
    actual_seq = actual_tool_sequence(trace)
    if not tool_sequence_match(actual_seq, turn_spec):
        return False, "plan_error", f"tool sequence mismatch: actual={actual_seq}"
    for step in trace.step_results:
        result = step.get("result")
        if result and result.get("ok") is False:
            if any(k in trace.answer for k in ["失败", "暂不支持", "无法", "错误"]):
                break
            return False, "step_error", result.get("error")
    contains_all = turn_spec.get("expected_contains_all")
    if contains_all is not None and not check_contains_all(trace.answer, contains_all):
        return False, ("memory_error" if "memory" in turn_spec.get("tags", []) else "answer_error"), f"missing all keywords: {contains_all}"
    contains_any = turn_spec.get("expected_contains_any")
    if contains_any is not None and not check_contains_any(trace.answer, contains_any):
        return False, ("memory_error" if "memory" in turn_spec.get("tags", []) else "answer_error"), f"missing any keywords: {contains_any}"
    return True, None, None


def evaluate_trace(trace: MultiStepTurnTrace, turn_spec: Dict[str, Any]) -> MultiStepTurnTrace:
    trace.passed, trace.failure_type, trace.failure_detail = classify_multistep_failure(trace, turn_spec)
    return trace


class MultiStepAgentEvaluator:
    def __init__(self, chat_client: VLLMChatClient, tools: Optional[LocalTools] = None, debug: bool = False):
        self.chat_client = chat_client
        self.tools = tools or LocalTools()
        self.debug = debug

    def run_turn(self, session_name: str, turn_index: int, turn_spec: Dict[str, Any], memory: ConversationMemory, state: ConversationState) -> MultiStepTurnTrace:
        user_input = turn_spec["user"]
        trace = MultiStepTurnTrace(
            session_name=session_name,
            turn_index=turn_index,
            user=user_input,
            tags=turn_spec.get("tags", []),
            expected_tools_sequence=turn_spec.get("expected_tools_sequence"),
            acceptable_tool_sequences=turn_spec.get("acceptable_tool_sequences"),
        )
        context = memory.build_context()
        plan, trace.plan_raw, trace.planning_latency_ms = build_multistep_plan(self.chat_client, context, user_input, state)
        trace.plan = plan

        if self.debug:
            print(f"\n=== {session_name} / turn {turn_index} ===")
            print("PLAN RAW:", trace.plan_raw)
            print("PLAN PARSED:", json.dumps(trace.plan, ensure_ascii=False))

        tool_steps = plan.get("steps", [])
        local_state = ConversationState(
            last_tool=state.last_tool,
            last_city=state.last_city,
            last_topic=state.last_topic,
            seen_cities=list(state.seen_cities),
            asked_topics=list(state.asked_topics),
        )
        for index, step in enumerate(tool_steps, start=1):
            result, latency_ms = self.tools.call(step["tool"], step.get("args", {}))
            trace.step_results.append({
                "step_index": index,
                "tool": step["tool"],
                "args": step.get("args", {}),
                "purpose": step.get("purpose", ""),
                "result": result,
                "latency_ms": latency_ms,
            })
            local_state.update({"tool": step["tool"], "args": step.get("args", {})})

        final_messages = [
            {"role": "system", "content": MULTISTEP_FINAL_SYSTEM_PROMPT},
            *context,
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": f"多步计划：{json.dumps(plan, ensure_ascii=False)}"},
            {"role": "user", "content": f"步骤执行结果：{json.dumps(trace.step_results, ensure_ascii=False)}\n请基于这些步骤结果给出最终答复。"},
        ]
        trace.answer, trace.answer_latency_ms = self.chat_client.chat(final_messages, max_tokens=ANSWER_MAX_TOKENS)
        memory.add_turn(user_input, trace.answer)
        trace.summary_triggered, trace.summary_latency_ms = maybe_summarize(memory, self.chat_client)
        trace.memory_summary = memory.summary

        for step in tool_steps:
            state.update({"tool": step["tool"], "args": step.get("args", {})})
        if not tool_steps:
            state.update({"tool": "direct_answer", "args": {}})
        return trace

    def run_dataset(self, dataset: Sequence[Dict[str, Any]]) -> List[MultiStepTurnTrace]:
        traces: List[MultiStepTurnTrace] = []
        for item in dataset:
            if "turns" in item:
                session_name = item.get("session_name", "unknown")
                memory = ConversationMemory()
                state = ConversationState()
                for index, turn_spec in enumerate(item["turns"], start=1):
                    trace = self.run_turn(session_name, index, turn_spec, memory, state)
                    traces.append(evaluate_trace(trace, turn_spec))
            else:
                session_name = item.get("session_name", "unknown")
                memory = ConversationMemory()
                state = ConversationState()
                turn_spec = {
                    "user": item["user"],
                    "tags": item.get("tags", []),
                    "expected_tools_sequence": item.get("expected_tools_sequence"),
                    "acceptable_tool_sequences": item.get("acceptable_tool_sequences"),
                }
                trace = self.run_turn(session_name, 1, turn_spec, memory, state)
                traces.append(evaluate_trace(trace, turn_spec))
        return traces


def aggregate_stats(traces: List[MultiStepTurnTrace]) -> Dict[str, Any]:
    total = len(traces)
    passed = sum(1 for t in traces if t.passed)
    by_step_tool: Dict[str, Dict[str, float]] = {}
    by_tag: Dict[str, Dict[str, float]] = {}
    by_failure_type: Dict[str, int] = {}
    by_route_source: Dict[str, Dict[str, float]] = {}
    total_steps = 0

    for trace in traces:
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

        for step in trace.step_results:
            total_steps += 1
            tool = step["tool"]
            by_step_tool.setdefault(tool, {"total": 0, "passed": 0})
            by_step_tool[tool]["total"] += 1
            by_step_tool[tool]["passed"] += int(step.get("result", {}).get("ok", False))

    def with_rates(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        return {
            key: {**value, "pass_rate": round(value["passed"] / value["total"], 4) if value["total"] else 0.0}
            for key, value in stats.items()
        }

    avg_planning = round(sum(t.planning_latency_ms for t in traces) / total, 2) if total else 0.0
    avg_answer = round(sum(t.answer_latency_ms for t in traces) / total, 2) if total else 0.0
    avg_summary = round(sum(t.summary_latency_ms for t in traces) / total, 2) if total else 0.0
    avg_steps = round(total_steps / total, 2) if total else 0.0

    return {
        "total_turns": total,
        "passed_turns": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_latency_ms": {
            "planning": avg_planning,
            "answer": avg_answer,
            "summary": avg_summary,
        },
        "avg_steps_per_turn": avg_steps,
        "by_step_tool": with_rates(by_step_tool),
        "by_tag": with_rates(by_tag),
        "by_failure_type": by_failure_type,
        "by_route_source": with_rates(by_route_source),
    }


def save_outputs(traces: List[MultiStepTurnTrace], summary: Dict[str, Any], results_json: Path, trace_jsonl: Path) -> None:
    results_json.parent.mkdir(parents=True, exist_ok=True)
    trace_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with results_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with trace_jsonl.open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(asdict(trace), ensure_ascii=False) + "\n")
