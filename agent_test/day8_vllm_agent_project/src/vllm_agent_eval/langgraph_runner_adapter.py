from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .day13_compare_metrics import compute_langgraph_metrics
from .langgraph_poc import build_langgraph_poc


def run_langgraph_runner_adapter(dataset_path: Path, output_dir: Path, model_label: str = "next80b_fp8") -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    graph = build_langgraph_poc()

    all_results: list[dict[str, Any]] = []
    for session in data:
        session_name = session.get("session_name", "unknown")
        for idx, turn in enumerate(session.get("turns", []), start=1):
            user_input = turn["user"]
            result = graph.invoke({"user_input": user_input, "debug_trace": []})
            all_results.append(
                {
                    "session_name": session_name,
                    "turn_index": idx,
                    "user": user_input,
                    "result": result,
                }
            )

    results_path = output_dir / "langgraph_results.json"
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = compute_langgraph_metrics(all_results)
    session_count = len(data)
    turn_count = sum(len(item.get("turns", [])) for item in data)
    passed_count = sum(
        1
        for item in all_results
        if isinstance(item.get("result", {}).get("final_answer", ""), str)
        and item.get("result", {}).get("final_answer", "").strip()
    )

    summary = {
        "dataset": str(dataset_path),
        "model_label": model_label,
        "session_count": session_count,
        "turn_count": turn_count,
        "passed_count": passed_count,
        "results_file": str(results_path),
        **metrics,
        "overall_passed": passed_count == turn_count,
    }

    summary_path = output_dir / "day14_runner_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
