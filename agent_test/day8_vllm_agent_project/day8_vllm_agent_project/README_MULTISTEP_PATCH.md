# Day 8 多步任务补丁

这个补丁包是在现有 `day8_vllm_agent_project` 基础上增加“多步任务规划 + 顺序执行 + 结果整合”的能力。

## 新增模块

- `src/vllm_agent_eval/multistep_prompts.py`
- `src/vllm_agent_eval/multistep_schemas.py`
- `src/vllm_agent_eval/multistep_dataset.py`
- `src/vllm_agent_eval/multistep_evaluator.py`
- `src/vllm_agent_eval/cli_multistep.py`
- `datasets/day8_multistep_dataset.json`
- `scripts/run_multistep_eval.sh`

## 运行方式

```bash
PYTHONPATH=src python -m vllm_agent_eval.cli_multistep --model next80b_fp8 --debug
```

或者：

```bash
bash scripts/run_multistep_eval.sh next80b_fp8
```

## 设计原则

- 继续复用原工程的 `client / tools / memory / config`
- 优先使用启发式拆分“先…再…然后…”句式
- 无法稳定拆分时，再用模型输出多步 JSON 计划
- 工具步骤按顺序执行
- 最终由模型读取“原问题 + 多步计划 + 步骤结果 + 历史上下文”生成答案
- 部分步骤失败时，不直接中断；保留部分结果，并要求最终答案说明失败步骤
