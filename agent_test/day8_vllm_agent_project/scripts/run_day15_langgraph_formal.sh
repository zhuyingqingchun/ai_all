#!/usr/bin/env bash
set -euo pipefail

MODEL_LABEL="${1:-next80b_fp8}"

PYTHONPATH=src python -m vllm_agent_eval.cli_day15_langgraph_formal_runner --model-label "${MODEL_LABEL}"
