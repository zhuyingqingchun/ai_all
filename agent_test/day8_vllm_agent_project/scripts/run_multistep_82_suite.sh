#!/usr/bin/env bash
set -euo pipefail

MODEL_ALIAS="${1:-next80b_fp8}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

PYTHONPATH=src python -m vllm_agent_eval.cli_multistep_suite --model "$MODEL_ALIAS"
