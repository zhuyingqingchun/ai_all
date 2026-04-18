#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-Qwen2.5-7B-Instruct}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
API_KEY="${API_KEY:-dummy}"

PYTHONPATH=src python -m vllm_agent_eval.cli_day16_langgraph_parity_runner \
  --model "${MODEL}" \
  --base-url "${BASE_URL}" \
  --api-key "${API_KEY}"
