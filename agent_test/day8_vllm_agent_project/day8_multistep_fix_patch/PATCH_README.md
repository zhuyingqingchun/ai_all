# Day 8.1 多步任务补丁说明

这份补丁只针对 **本次 Day 8 多步任务实验** 的 3 个失败点：

1. `再算...` 被误归入 `final_instruction`
2. 启发式失败后直接 fallback 到 `direct_answer`
3. 多步句里 `unsupported_city` 误判吞掉整个计划

## 文件

- `src/vllm_agent_eval/multistep_patch_helpers.py`
  - 子句切分
  - step 分类
  - salvage fallback
- `datasets/day8_multistep_regression_cases.json`
  - 3 条回归样本

## 建议接入点

在你当前的 `multistep_evaluator.py` 中：

1. 先导入：

```python
from vllm_agent_eval.multistep_patch_helpers import (
    split_multistep_clauses,
    classify_clause_to_step,
    build_salvage_multistep_plan,
)
```

2. 把现有的多步启发式 planner 替换成：
   - 先 `split_multistep_clauses(user_input)`
   - 再对子句逐个 `classify_clause_to_step(...)`
   - 如果 step 为空，再走模型 JSON
   - 如果模型 JSON 失败，再走 `build_salvage_multistep_plan(...)`

## 目标

这份补丁的目标不是提升所有多步能力，而是把本次实验中：
- `plan_error`
- `fallback_single_direct_answer`
- `南京` 城市误抽取

先修掉一版。
