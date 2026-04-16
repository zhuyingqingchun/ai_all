from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .client import VLLMChatClient
from .config import DEFAULT_API_KEY, DEFAULT_BASE_URL
from .multistep_dataset import (
    load_multistep_dataset,
    load_regression_multistep_dataset,
    save_default_multistep_dataset,
    save_regression_multistep_dataset,
)
from .multistep_evaluator import MultiStepAgentEvaluator, aggregate_stats, save_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Day 8.2 multistep validation suite against a vLLM API service.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", required=True, help="Model id or alias exposed by vLLM.")
    parser.add_argument("--main-dataset", default=None, help="Optional main multistep dataset JSON path.")
    parser.add_argument("--regression-dataset", default=None, help="Optional regression dataset JSON path.")
    parser.add_argument("--out-dir", default="results/day8_2")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--write-main-dataset", default=None)
    parser.add_argument("--write-regression-dataset", default=None)
    return parser


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


def _run_one(evaluator: MultiStepAgentEvaluator, dataset: List[Dict[str, Any]], results_json: Path, trace_jsonl: Path) -> Dict[str, Any]:
    traces = evaluator.run_dataset(dataset)
    summary = aggregate_stats(traces)
    save_outputs(traces, summary, results_json, trace_jsonl)
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.write_main_dataset:
        save_default_multistep_dataset(args.write_main_dataset)
        print(f"主数据集已写入: {args.write_main_dataset}")
        return
    if args.write_regression_dataset:
        save_regression_multistep_dataset(args.write_regression_dataset)
        print(f"回归数据集已写入: {args.write_regression_dataset}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    main_dataset = load_multistep_dataset(args.main_dataset)
    regression_dataset = load_regression_multistep_dataset(args.regression_dataset)

    client = VLLMChatClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    evaluator = MultiStepAgentEvaluator(chat_client=client, debug=args.debug)

    main_summary = _run_one(
        evaluator,
        main_dataset,
        out_dir / "day8_2_main_eval_results.json",
        out_dir / "day8_2_main_eval_traces.jsonl",
    )
    regression_summary = _run_one(
        evaluator,
        regression_dataset,
        out_dir / "day8_2_regression_eval_results.json",
        out_dir / "day8_2_regression_eval_traces.jsonl",
    )

    gate_main = _suite_gate(main_summary, min_pass_rate=0.875, forbid_failure_types=["plan_error"])
    gate_regression = _suite_gate(regression_summary, min_pass_rate=1.0, forbid_failure_types=["plan_error"])

    combined = {
        "model": args.model,
        "main": main_summary,
        "regression": regression_summary,
        "gates": {
            "main": gate_main,
            "regression": gate_regression,
            "overall_passed": gate_main["passed"] and gate_regression["passed"],
        },
    }

    combined_path = out_dir / "day8_2_suite_summary.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)

    print(json.dumps(combined, ensure_ascii=False, indent=2))
    print(f"Day 8.2 总结已保存到: {combined_path}")


if __name__ == "__main__":
    main()
