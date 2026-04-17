# Day 15 LangGraph Formal Runner

## Goal

Validate whether LangGraph can carry a small subset of formal multistep samples with assertion-style checks.

## Install

```bash
pip install langgraph langchain-core
```

## Run

```bash
bash scripts/run_day15_langgraph_formal.sh
```

Or:

```bash
PYTHONPATH=src python -m vllm_agent_eval.cli_day15_langgraph_formal_runner --model-label next80b_fp8
```

