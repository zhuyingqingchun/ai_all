# Day 14 LangGraph Runner Adapter

## Goal

Validate whether LangGraph can serve as a small runner candidate for a subset of formal multistep tasks.

## Install

```bash
pip install langgraph langchain-core
```

## Run

```bash
bash scripts/run_day14_langgraph_runner.sh
```

Or:

```bash
PYTHONPATH=src python -m vllm_agent_eval.cli_langgraph_runner_adapter --model-label next80b_fp8
```

