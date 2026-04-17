#!/usr/bin/env bash
set -euo pipefail

MODEL_LABEL="${1:-next80b_fp8}"

PYTHONPATH=src python -m vllm_agent_eval.cli_langgraph_runner_adapter --model-label "${MODEL_LABEL}"
