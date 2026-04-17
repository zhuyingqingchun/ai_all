from __future__ import annotations

import argparse
import json
from pathlib import Path

from .day12_langgraph_compare import run_compare


def main():
    parser = argparse.ArgumentParser(description="Day 12 LangGraph compare runner")
    parser.add_argument("--model", default="next80b_fp8")
    parser.add_argument("--dataset", default="datasets/day12_langgraph_compare_subset.json")
    parser.add_argument("--output-dir", default="results/day12_compare")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / args.dataset
    output_dir = project_root / args.output_dir

    summary = run_compare(project_root=project_root, dataset_path=dataset_path, model=args.model, output_dir=output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
