#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS_OR_ID="${1:-next80b_fp8}"
PYTHONPATH=src python -m vllm_agent_eval.cli_multistep --model "$MODEL_ALIAS_OR_ID" --dataset datasets/day8_multistep_dataset.json "$@"
