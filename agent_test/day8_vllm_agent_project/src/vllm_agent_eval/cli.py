from __future__ import annotations

import argparse
import json
from pathlib import Path

from .client import VLLMChatClient
from .config import DEFAULT_API_KEY, DEFAULT_BASE_URL, DEFAULT_RESULTS_JSON, DEFAULT_TRACE_JSONL
from .dataset import load_dataset, save_default_dataset
from .evaluator import AgentEvaluator, aggregate_stats, save_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run modular engineering evaluation against a vLLM API service.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", required=True, help="Model id or alias exposed by vLLM.")
    parser.add_argument("--dataset", default=None, help="Optional dataset JSON path.")
    parser.add_argument("--results-json", default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--trace-jsonl", default=str(DEFAULT_TRACE_JSONL))
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--write-default-dataset", default=None, help="Write bundled dataset JSON to this path and exit.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.write_default_dataset:
        save_default_dataset(args.write_default_dataset)
        print(f"默认数据集已写入: {args.write_default_dataset}")
        return
    dataset = load_dataset(args.dataset)
    client = VLLMChatClient(base_url=args.base_url, api_key=args.api_key, model=args.model)
    evaluator = AgentEvaluator(chat_client=client, debug=args.debug)
    traces = evaluator.run_dataset(dataset)
    summary = aggregate_stats(traces)
    save_outputs(traces, summary, Path(args.results_json), Path(args.trace_jsonl))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果已保存到: {args.results_json}")
    print(f"trace 已保存到: {args.trace_jsonl}")


if __name__ == "__main__":
    main()
