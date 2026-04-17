from __future__ import annotations

import argparse
import json

from .langgraph_poc import build_langgraph_poc


def main():
    parser = argparse.ArgumentParser(description="LangGraph POC for day11")
    parser.add_argument("--input", required=True, help="User input")
    args = parser.parse_args()
    graph = build_langgraph_poc()
    result = graph.invoke({"user_input": args.input, "debug_trace": []})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
