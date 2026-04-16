# Day 7：vLLM API 版 Agent 工程化评测说明

## 1. 这一步做什么

这一阶段的目标是：

1. 不再在评测脚本里重复加载本地模型。
2. 直接调用已经常驻运行的 vLLM OpenAI-compatible API。
3. 继续执行 Day 6 的 agent 工程化评测，包括：
   - 工具路由
   - 多轮上下文记忆
   - 历史摘要 memory
   - 边界输入处理
   - 路由错误分类和统计
4. 输出可落盘的 JSON 结果和 trace 结果。

---

## 2. 交付文件

### Python 脚本
- `day7_vllm_agent_engineering_eval.py`

### 输出结果
- `day7_vllm_agent_eval_results.json`
- `day7_vllm_agent_eval_traces.jsonl`

---

## 3. 运行前提

需要确保 vLLM 服务已经启动，并且可访问：

- Base URL: `http://127.0.0.1:8000/v1`
- `/v1/models` 可正常返回模型列表
- `/v1/chat/completions` 可正常对话

建议先用命令确认：

```bash
curl http://127.0.0.1:8000/v1/models
```

---

## 4. 安装依赖

最少需要：

```bash
pip install openai
```

如果环境里已经装过，就不需要重复安装。

---

## 5. 运行方式

### 默认运行

```bash
python day7_vllm_agent_engineering_eval.py
```

### 显式指定模型

```bash
python day7_vllm_agent_engineering_eval.py --model "Qwen3-Next-80B-A3B-Instruct-FP8"
```

### 指定输出文件

```bash
python day7_vllm_agent_engineering_eval.py \
  --model "Qwen3-Next-80B-A3B-Instruct-FP8" \
  --out day7_results.json \
  --trace-jsonl day7_traces.jsonl
```

### 打开调试模式

```bash
python day7_vllm_agent_engineering_eval.py --debug
```

调试模式会额外打印：
- 原始路由输出 `plan_raw`
- 工具返回 `tool_result`
- 摘要记忆 `summary_memory`

---

## 6. 脚本能力概览

### 6.1 路由策略

脚本采用：

- **规则优先**
- **模型 JSON 路由兜底**

具体包括：
- 数学表达式 → `calculator`
- 历史问题 → `direct_answer`
- 天气问题 → `get_weather`
- 时间问题 → `get_time`
- 省略句（如“那纽约呢”）→ 根据上一轮状态做启发式路由

这一步是为了解决 Day 6 中的主要问题：
- `route_error`
- `ellipsis`
- `boundary`
- `unsupported_city`

---

### 6.2 结构化状态

脚本内部维护：
- `last_tool`
- `last_city`
- `last_topic`
- `seen_cities`
- `asked_topics`

用于辅助处理省略句与指代。

---

### 6.3 历史摘要 memory

当最近原始消息超过阈值时，会自动把更早对话压缩为摘要，再继续参与后续推理。

当前阈值：
- `MAX_RECENT_MESSAGES = 8`
- `KEEP_RECENT_MESSAGES = 4`

---

### 6.4 工具层

本版本为了评测可重复性，默认使用**可重复 mock 工具**：

- `get_weather(city)` → 稳定返回伪天气数据
- `get_time(city)` → 稳定返回伪时间数据
- `calculator(expression)` → AST 安全求值

这样做的好处是：
- 不受外部 API 波动影响
- 便于版本对比
- 适合做服务级 agent 评测

---

## 7. 数据集内容

当前数据集覆盖：

1. 基础工具能力
   - 天气
   - 时间
   - 计算
2. 多轮记忆
3. 同义表达
4. 省略句
5. 不支持场景
6. 非法输入
7. 摘要记忆回溯

---

## 8. 输出结果怎么读

### 8.1 控制台

每轮会打印：
- USER
- PLAN
- ANSWER
- PASS
- 规划/工具/回答/摘要耗时

### 8.2 汇总 JSON

`day7_vllm_agent_eval_results.json` 包含：
- `summary`
- `traces`

其中 `summary` 里重点看：
- `pass_rate`
- `by_tool`
- `by_tag`
- `by_failure_type`
- `by_route_source`

### 8.3 trace JSONL

`day7_vllm_agent_eval_traces.jsonl` 每行是一条完整 trace，适合：
- grep
- pandas 分析
- 后续版本对比

---

## 9. 这一步为什么重要

这一步完成后，评测链路就从“本地脚本直接加载模型”升级成：

- **模型服务化**
- **评测脚本解耦**
- **多脚本可复用同一模型服务**
- **支持 A/B 对比不同模型版本**

这对后续继续做 agent 工程化非常关键。

---

## 10. 下一步建议

下一步建议做两件事：

### 10.1 A/B 对比
用同一份数据集，对比：
- 旧模型
- 新模型

重点比较：
- `pass_rate`
- `by_failure_type`
- `by_route_source`
- `memory` / `ellipsis` / `boundary` 标签表现

### 10.2 切换真实工具
当服务级评测链路稳定后，再把 mock 工具替换成真实 API：
- 真实天气 API
- 真实时间查询
- 更多工具组合

建议顺序：
1. 先把 mock 评测跑稳
2. 再切真实工具
3. 再做更复杂的多步 agent

---

## 11. 给低级小模型的任务单

你现在的任务是：

1. 确认 vLLM 服务已启动。
2. 运行：
   ```bash
   python day7_vllm_agent_engineering_eval.py --debug
   ```
3. 查看控制台输出是否正常。
4. 检查是否生成：
   - `day7_vllm_agent_eval_results.json`
   - `day7_vllm_agent_eval_traces.jsonl`
5. 从结果中总结：
   - 总通过率
   - 失败类型
   - 哪些标签最弱
   - 哪些路由来源最有效
6. 输出一份新的评测总结报告。

如果模型名不是自动识别的那个，就显式加：

```bash
python day7_vllm_agent_engineering_eval.py --model "你的模型ID"
```

---

## 12. 当前定位

这不是“继续做 demo”，而是进入真正的 agent 工程化阶段：

- 有服务
- 有评测
- 有 trace
- 有统计
- 有版本对比基础

这条线继续往下走，就可以做：
- A/B 对比
- ablation test
- 多工具多步任务
- 更复杂的文档/代码 agent
