# Day 8 多步任务模块补丁报告

## 目标

在现有 `day8_vllm_agent_project` 基础上，补充“多步任务”能力，而不是重写一套新工程。

本补丁聚焦：
- 多步规划（planner）
- 顺序执行（executor）
- 结果整合（final synthesizer）
- 多步数据集与评测入口

## 补丁内容

新增文件：

- `src/vllm_agent_eval/multistep_prompts.py`
- `src/vllm_agent_eval/multistep_schemas.py`
- `src/vllm_agent_eval/multistep_dataset.py`
- `src/vllm_agent_eval/multistep_evaluator.py`
- `src/vllm_agent_eval/cli_multistep.py`
- `datasets/day8_multistep_dataset.json`
- `scripts/run_multistep_eval.sh`
- `README_MULTISTEP_PATCH.md`

## 设计说明

### 1. 复用现有模块

补丁没有改写原来的：
- `client.py`
- `tools.py`
- `memory.py`
- `config.py`

因此可以直接叠加到现有 Day 8 工程上。

### 2. 规划策略

优先用启发式拆分：
- `先 … 再 … 然后 … 最后 …`

如果启发式无法形成有效步骤，再调用模型输出多步 JSON：

```json
{
  "steps": [
    {"tool": "get_time", "args": {"city": "东京"}, "purpose": "查询东京时间"},
    {"tool": "calculator", "args": {"expression": "(23+7)*3"}, "purpose": "计算表达式"}
  ],
  "final_instruction": "先回答东京时间，再回答计算结果"
}
```

最后兜底成单步计划或纯 direct answer。

### 3. 执行策略

- 按顺序执行每一步工具
- 保存每步结果与延迟
- 不因为单步失败直接中断
- 最终让模型根据“原问题 + 计划 + 步骤结果”生成答案

### 4. 评测重点

新增多步数据集，覆盖：
- 时间 + 计算
- 天气 + 时间 + 总结
- 双计算
- 历史回顾 + 新工具步骤
- 部分失败恢复

## 建议运行方式

在项目根目录执行：

```bash
bash scripts/run_multistep_eval.sh next80b_fp8
```

或：

```bash
PYTHONPATH=src python -m vllm_agent_eval.cli_multistep --model next80b_fp8 --debug
```

## 预期输出

结果文件建议保存到：
- `results/day8_multistep_eval_results.json`
- `results/day8_multistep_eval_traces.jsonl`

## 下一步建议

如果这个补丁跑通，下一步最值得做的是：

1. 把“多步 planner”从启发式为主，逐步过渡到“结构化状态 + 模型 planner”的混合模式
2. 增加 `partial_failure_recovery_rate`
3. 为多步任务补一个 `ablation`：
   - 关闭启发式拆分
   - 关闭 history summary
   - 对比 pass rate
4. 给多步数据集加入更多真实项目型任务，例如：
   - 先总结历史，再查询新城市
   - 连续两次计算再对比大小
   - 先查天气再给出是否适合出门的结论
