# Tool Integration Spec

## 目标

每新增一个本地 deterministic 工具，都要同时完成五层接入：

1. 工具函数可运行
2. planner prompt 知道该工具
3. heuristic clause 能识别该工具
4. evaluator / parser 允许该工具进入计划
5. 数据集与回归断言覆盖该工具

## 接入检查表

### 1. 代码层
- 在 `src/vllm_agent_eval/tools.py` 中实现工具函数
- 在 `LocalTools.call(...)` 中加入分发逻辑
- 如有常量表或 mock 数据，在 `config.py` 中集中管理

### 2. Planner 层
- 在 `multistep_prompts.py` 的 `MULTISTEP_PLANNER_SYSTEM_PROMPT` 中加入工具说明
- 明确参数格式与示例
- 明确何时应使用该工具

### 3. Heuristic 层
- 在 `multistep_patch_helpers.py` 中补充：
  - hint 关键词
  - clause 解析函数
  - `split_multistep_clauses(...)` 中的 step 收集逻辑
  - `classify_clause_to_step(...)` 中的工具映射

### 4. Evaluator 层
- 在 `multistep_evaluator.py` 的 `parse_multistep_json(...)` 中加入允许的工具名
- 确认失败类型与断言逻辑适用于该工具

### 5. 数据集层
至少新增三类样本：
- 单工具成功样本
- 与已有工具的多步组合样本
- 失败或边界样本

## 验收标准

- 工具单元 case 通过
- planner route 正确
- 多步组合 case 通过
- baseline / LangGraph 行为一致
- 回归套件不退化
