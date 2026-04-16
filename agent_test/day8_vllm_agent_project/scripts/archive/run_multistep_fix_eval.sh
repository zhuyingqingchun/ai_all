#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS_OR_ID="${1:-next80b_fp8}"
conda run -n swdtorch12 python -m vllm_agent_eval.cli_multistep --model "$MODEL_ALIAS_OR_ID" --dataset datasets/day8_multistep_regression_cases.json --results-json results/day8_multistep_fix_eval_results.json --trace-jsonl results/day8_multistep_fix_eval_traces.jsonl
