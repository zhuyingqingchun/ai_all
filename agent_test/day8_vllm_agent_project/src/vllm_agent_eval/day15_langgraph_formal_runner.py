from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .day15_langgraph_formal_checks import evaluate_langgraph_formal_turn
from .langgraph_poc import build_langgraph_poc


def run_day15_langgraph_formal(dataset_path: Path, output_dir: Path, model_label: str = "next80b_fp8") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    graph = build_langgraph_poc()

    all_results: list[dict[str, Any]] = []
    passed_count = 0
    failure_types: dict[str, int] = {}

    for session in data:
        session_name = session.get("session_name", "unknown")
        for idx, turn in enumerate(session.get("turns", []), start=1):
            user_input = turn["user"]
            graph_result = graph.invoke({"user_input": user_input, "debug_trace": []})
            passed, failure_type, failure_detail = evaluate_langgraph_formal_turn(graph_result, turn)
            if passed:
                passed_count += 1
            else:
                failure_types[failure_type or "unknown"] = failure_types.get(failure_type or "unknown", 0) + 1

            all_results.append(
                {
                    "session_name": session_name,
                    "turn_index": idx,
                    "user": user_input,
                    "result": graph_result,
                    "passed": passed,
                    "failure_type": failure_type,
                    "failure_detail": failure_detail,
                }
            )

    results_path = output_dir / "day15_formal_results.json"
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    session_count = len(data)
    turn_count = sum(len(item.get("turns", [])) for item in data)
    summary = {
        "dataset": str(dataset_path),
        "model_label": model_label,
        "session_count": session_count,
        "turn_count": turn_count,
        "passed_count": passed_count,
        "pass_rate": round(passed_count / turn_count, 4) if turn_count else 0.0,
        "failure_types": failure_types,
        "results_file": str(results_path),
        "overall_passed": passed_count == turn_count,
    }

    summary_path = output_dir / "day15_formal_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
