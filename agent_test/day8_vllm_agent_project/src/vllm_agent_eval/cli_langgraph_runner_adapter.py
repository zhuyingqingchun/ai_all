from __future__ import annotations

import argparse
import json
from pathlib import Path

from .langgraph_runner_adapter import run_langgraph_runner_adapter


def main():
    parser = argparse.ArgumentParser(description="Day 14 LangGraph runner adapter")
    parser.add_argument("--dataset", default="datasets/day14_langgraph_migration_subset.json")
    parser.add_argument("--output-dir", default="results/day14_langgraph_runner")
    parser.add_argument("--model-label", default="next80b_fp8")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / args.dataset
    output_dir = project_root / args.output_dir

    summary = run_langgraph_runner_adapter(dataset_path=dataset_path, output_dir=output_dir, model_label=args.model_label)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
