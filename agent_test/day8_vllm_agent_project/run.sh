#!/usr/bin/env bash
set -euo pipefail

export ALL_PROXY=""
export HTTP_PROXY=""
export HTTPS_PROXY=""

cd /mnt/PRO6000_disk/swd/agent/agent_test/day8_vllm_agent_project
conda run -n swdtorch12 python run.py --model "${1:-next80b_fp8}" --results-json results/day8_vllm_agent_eval_results.json --trace-jsonl results/day8_vllm_agent_eval_traces.jsonl --debug
