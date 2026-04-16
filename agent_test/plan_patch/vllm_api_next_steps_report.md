# vLLM 本地模型启动后的下一步报告

## 1. 当前状态

你已经完成了最关键的一步：

- 本地模型已经通过 vLLM 启动为常驻服务
- 模型只在服务启动时加载一次
- 后续 Python 脚本可以直接通过 HTTP 调用这个本地 API
- 不需要在每个测试脚本里重复加载模型

这意味着接下来的工作重点已经从“部署模型”切换到“验证服务是否稳定、验证模型在任务上的表现、把原来的 agent/评测脚本迁移到 API 调用方式”。

---

## 2. 本次给出的脚本

本次提供的脚本是：

- `vllm_api_noninteractive_eval.py`

这个脚本的目标不是做完整 agent，而是做 **服务级非交互评测**，用来快速检查：

1. `/v1/models` 是否正常返回
2. `/v1/chat/completions` 是否正常返回
3. 模型的基础中文能力是否正常
4. 模型的 JSON 输出能力是否正常
5. 单请求多轮上下文记忆能力是否正常
6. 边界输入处理倾向是否正常
7. 长指令遵循是否正常

---

## 3. 这个脚本是做什么的

脚本内部内置了一小组测试样例：

- 基础中文解释
- 只输出 JSON 的路由式输出
- 单请求中的记忆引用
- 非法数学表达式
- 不支持城市/异常对象
- 长指令格式遵循

运行后它会：

- 自动请求 `GET /v1/models`
- 自动选择模型名（或使用你手动指定的模型名）
- 逐条执行非交互测试
- 打印每一条的延迟、响应、是否通过
- 最后输出一个总体报告
- 同时把完整结果保存成 JSON 文件

---

## 4. 运行方式

### 4.1 最简单运行

```bash
python /mnt/data/vllm_api_noninteractive_eval.py
```

默认会调用：

- `base_url = http://127.0.0.1:8000/v1`
- `api_key = dummy`

如果服务是本机默认端口，这一条就够了。

### 4.2 指定模型名运行

```bash
python /mnt/data/vllm_api_noninteractive_eval.py \
  --model /mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
```

或者如果 `/v1/models` 返回的是别名 ID，就把 `--model` 改成那个 ID。

### 4.3 指定结果输出路径

```bash
python /mnt/data/vllm_api_noninteractive_eval.py \
  --out /mnt/PRO6000_disk/swd/agent/results/qwen3next_eval.json
```

---

## 5. 输出结果怎么看

终端会看到两部分：

### 5.1 逐条结果

每条都会打印：

- 用例名
- 描述
- 延迟
- 是否通过
- 模型输出文本
- 缺失的关键词（如果有）

### 5.2 最终汇总

最终会输出：

- `total_cases`
- `passed_cases`
- `pass_rate`
- `avg_latency_ms`

同时还会保存完整 JSON，便于后续做版本对比。

---

## 6. 建议的下一步执行顺序

### 第一步：先跑这个服务级评测脚本

目的：

- 验证 vLLM 服务本身是稳定的
- 验证模型名是否正确
- 验证基础调用是否正常
- 验证新模型在格式化输出、记忆、边界输入上的基本能力

这一步通过之后，再进入下一步。

### 第二步：把 Day 6 / Day 7 的评测脚本改成 API 调用版

也就是把原来本地 `transformers` 直接推理的部分，替换成：

- 向本地 vLLM 的 `/v1/chat/completions` 发请求
- 保留原有的数据集、规则、打分逻辑、trace 逻辑

这样你就能直接做：

- 同一套数据集
- 同一套评测指标
- 对比旧模型和新模型

### 第三步：再继续 agent 工程化

后续可以继续做：

- router 硬化
- structured state
- flexible eval
- 多工具链路
- API 化的 dataset regression test

---

## 7. 这一步的定位

这一步的定位不是“最终 agent 脚本”，而是：

> 先把模型服务验证好，再把整个 agent 工程迁移到 API 架构。

这样以后：

- 模型升级不需要改评测框架
- agent 脚本不需要重复加载模型
- 多个测试脚本都能复用同一个本地服务
- 更适合做持续迭代和版本对比

---

## 8. 给低级小模型的执行任务单

你现在的任务只有三件事：

1. 运行 `/mnt/data/vllm_api_noninteractive_eval.py`
2. 确认 `/v1/models` 能返回模型列表，并且评测能跑完
3. 保存结果 JSON，汇报：
   - 总通过率
   - 平均延迟
   - 哪些 case 失败了

不要一上来就继续改 agent 主框架。
先把服务调用和基础评测跑稳。

---

## 9. 当前最优先目标

当前最优先目标是：

**完成“本地 vLLM 服务 + Python 非交互评测脚本”的闭环。**

只要这一步稳定，后面把 Day 6/Day 7 迁移成 API 版本就会很顺。
