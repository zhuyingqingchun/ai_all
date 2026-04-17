#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:-先告诉我东京现在几点，再算一下(23+7)*3。}"

PYTHONPATH=src python -m vllm_agent_eval.cli_langgraph_poc --input "${INPUT}"
