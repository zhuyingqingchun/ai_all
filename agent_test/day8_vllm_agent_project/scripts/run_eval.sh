#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m vllm_agent_eval.cli   --model "${1:-next80b_fp8}"   --results-json results/day8_vllm_agent_eval_results.json   --trace-jsonl results/day8_vllm_agent_eval_traces.jsonl   --debug
