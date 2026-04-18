from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_API_KEY, DEFAULT_BASE_URL
from .day16_langgraph_parity_runner import run_day16_langgraph_parity


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 16 LangGraph parity runner")
    parser.add_argument("--dataset", default="datasets/day16_langgraph_parity_subset.json")
    parser.add_argument("--output-dir", default="results/day16_langgraph_parity")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = project_root / args.dataset
    output_dir = project_root / args.output_dir

    summary = run_day16_langgraph_parity(
        dataset_path=dataset_path,
        output_dir=output_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        debug=args.debug,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
