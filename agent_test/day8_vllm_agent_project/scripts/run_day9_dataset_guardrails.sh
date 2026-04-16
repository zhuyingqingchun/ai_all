#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[Day9] lint main multistep dataset"
python scripts/lint_multistep_dataset.py datasets/day8_multistep_dataset.json

echo "[Day9] lint regression suite"
python scripts/lint_multistep_dataset.py datasets/day8_multistep_regression_suite.json

echo "[Day9] dataset guardrails passed"
echo "[Day9] now you can run: bash scripts/run_multistep_82_suite.sh <model_alias>"
