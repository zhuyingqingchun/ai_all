# LangGraph POC

## Goal

Introduce a parallel LangGraph proof-of-concept without replacing the current agent framework.

## Install

```bash
pip install langgraph langchain-core
```

## Run

```bash
bash scripts/run_langgraph_poc.sh
```

Or:

```bash
PYTHONPATH=src python -m vllm_agent_eval.cli_langgraph_poc --input "先查北京天气，再告诉我上海现在几点。"
```

## Scope

This POC only validates graph structure and a minimal planner-executor-synthesizer loop.
