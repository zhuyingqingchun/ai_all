from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .client import VLLMChatClient
from .config import DEFAULT_API_KEY, DEFAULT_BASE_URL
from .langgraph_parity_runtime import LangGraphParityRuntime
from .memory import maybe_summarize
from .multistep_evaluator import (
    MultiStepAgentEvaluator,
    aggregate_stats,
    evaluate_trace,
    save_outputs,
)
from .multistep_schemas import MultiStepTurnTrace
from .schemas import ConversationMemory, ConversationState


def _load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


def _suite_gate(summary: Dict[str, Any], *, min_pass_rate: float, forbid_failure_types: List[str] | None = None) -> Dict[str, Any]:
    forbid_failure_types = forbid_failure_types or []
    pass_rate_ok = summary.get("pass_rate", 0.0) >= min_pass_rate
    failure_counts = summary.get("by_failure_type", {})
    forbidden_ok = all(failure_counts.get(name, 0) == 0 for name in forbid_failure_types)
    return {
        "min_pass_rate": min_pass_rate,
        "actual_pass_rate": summary.get("pass_rate", 0.0),
        "pass_rate_ok": pass_rate_ok,
        "forbid_failure_types": forbid_failure_types,
        "forbidden_failure_types_ok": forbidden_ok,
        "passed": pass_rate_ok and forbidden_ok,
    }


def run_baseline(
    *,
    dataset_path: Path,
    output_dir: Path,
    project_root: Path,
    base_url: str,
    api_key: str,
    model: str,
    debug: bool = False,
) -> Dict[str, Any]:
    data = _load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = VLLMChatClient(base_url=base_url, api_key=api_key, model=model)
    evaluator = MultiStepAgentEvaluator(chat_client=client, debug=debug)
    traces = evaluator.run_dataset(data)
    summary = aggregate_stats(traces)
    results_path = output_dir / "baseline_results.json"
    trace_path = output_dir / "baseline_traces.jsonl"
    save_outputs(traces, summary, results_path, trace_path)
    gate = _suite_gate(summary, min_pass_rate=1.0, forbid_failure_types=["plan_error"])
    return {
        "summary": summary,
        "gate": gate,
        "results_file": _display_path(results_path, project_root),
        "trace_file": _display_path(trace_path, project_root),
    }


def run_langgraph_parity(
    *,
    dataset_path: Path,
    output_dir: Path,
    project_root: Path,
    base_url: str,
    api_key: str,
    model: str,
    debug: bool = False,
) -> Dict[str, Any]:
    data = _load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = VLLMChatClient(base_url=base_url, api_key=api_key, model=model)
    runtime = LangGraphParityRuntime(chat_client=client, debug=debug)
    graph = runtime.build_graph()

    traces: List[MultiStepTurnTrace] = []
    for session in data:
        session_name = session.get("session_name", "unknown")
        memory = ConversationMemory()
        conversation_state = ConversationState()
        for index, turn_spec in enumerate(session.get("turns", []), start=1):
            user_input = turn_spec["user"]
            graph_result = graph.invoke(
                {
                    "user_input": user_input,
                    "context_messages": memory.build_context(),
                    "conversation_state": runtime.dump_conversation_state(conversation_state),
                    "debug_trace": [],
                }
            )

            trace = MultiStepTurnTrace(
                session_name=session_name,
                turn_index=index,
                user=user_input,
                tags=turn_spec.get("tags", []),
                expected_tools_sequence=turn_spec.get("expected_tools_sequence"),
                acceptable_tool_sequences=turn_spec.get("acceptable_tool_sequences"),
                plan_raw=graph_result.get("plan_raw", ""),
                plan=graph_result.get("plan", {}),
                step_results=graph_result.get("step_results", []),
                answer=graph_result.get("final_answer", ""),
                planning_latency_ms=graph_result.get("planning_latency_ms", 0.0),
                answer_latency_ms=graph_result.get("answer_latency_ms", 0.0),
            )
            memory.add_turn(user_input, trace.answer)
            trace.summary_triggered, trace.summary_latency_ms = maybe_summarize(memory, client)
            trace.memory_summary = memory.summary

            tool_steps = list(trace.plan.get("steps", []))
            for step in tool_steps:
                conversation_state.update({"tool": step["tool"], "args": step.get("args", {})})
            if not tool_steps:
                conversation_state.update({"tool": "direct_answer", "args": {}})

            traces.append(evaluate_trace(trace, turn_spec))

    summary = aggregate_stats(traces)
    results_path = output_dir / "langgraph_results.json"
    trace_path = output_dir / "langgraph_traces.jsonl"
    save_outputs(traces, summary, results_path, trace_path)
    gate = _suite_gate(summary, min_pass_rate=1.0, forbid_failure_types=["plan_error"])
    return {
        "summary": summary,
        "gate": gate,
        "results_file": _display_path(results_path, project_root),
        "trace_file": _display_path(trace_path, project_root),
    }


def run_day16_langgraph_parity(
    *,
    dataset_path: Path,
    output_dir: Path,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    model: str,
    debug: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baseline"
    langgraph_dir = output_dir / "langgraph"
    project_root = dataset_path.resolve().parents[1]

    baseline = run_baseline(
        dataset_path=dataset_path,
        output_dir=baseline_dir,
        project_root=project_root,
        base_url=base_url,
        api_key=api_key,
        model=model,
        debug=debug,
    )
    langgraph = run_langgraph_parity(
        dataset_path=dataset_path,
        output_dir=langgraph_dir,
        project_root=project_root,
        base_url=base_url,
        api_key=api_key,
        model=model,
        debug=debug,
    )

    baseline_summary = baseline["summary"]
    langgraph_summary = langgraph["summary"]
    parity = {
        "pass_rate_delta": round(langgraph_summary.get("pass_rate", 0.0) - baseline_summary.get("pass_rate", 0.0), 4),
        "avg_steps_delta": round(
            langgraph_summary.get("avg_steps_per_turn", 0.0) - baseline_summary.get("avg_steps_per_turn", 0.0),
            4,
        ),
        "overall_gate_passed": baseline["gate"]["passed"] and langgraph["gate"]["passed"],
    }

    summary = {
        "dataset": _display_path(dataset_path, project_root),
        "model": model,
        "baseline": baseline,
        "langgraph": langgraph,
        "parity": parity,
    }
    summary_path = output_dir / "day16_langgraph_parity_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
