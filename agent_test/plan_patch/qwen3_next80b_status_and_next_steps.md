# Qwen3-Next-80B-A3B-Instruct-FP8 本地部署阶段报告与下一步工作单

## 1. 本次阶段目标

本阶段的目标是：

1. 确认更高能力本地模型是否已经成功下载到指定目录。
2. 为后续的 Agent 工程化测试准备更强的基础模型。
3. 为后续服务化部署、API 调用测试、A/B 对比评测做好交接记录。

---

## 2. 当前完成情况

### 2.1 模型下载完成

已完成模型下载：

- **模型名称**：`Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- **本地目录**：`/mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- **缓存目录**：`/mnt/PRO6000_disk/modelscope_cache`

### 2.2 关键文件检查通过

以下关键文件已经确认存在：

- `config.json` ✅
- `tokenizer_config.json` ✅
- `generation_config.json` ✅

这说明模型目录基本完整，可以进入下一步部署验证。

### 2.3 可用模型别名

当前已整理出的模型别名如下：

- `next80b_fp8 -> Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- `qwen235b_fp8 -> Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`

其中当前优先使用的是：

- **`next80b_fp8`**

原因：

1. 相比当前的小模型，更适合做 Agent 路由、长上下文、总结记忆、结构化回答。
2. 规模比 235B 更容易先落地测试。
3. 更适合作为现阶段本地 API 服务主模型。

---

## 3. 当前环境判断

### 3.1 GPU 条件

当前机器拥有 4 张大显存 GPU，可用于多卡部署。

### 3.2 CUDA 条件

在当前 Python 环境中已经确认：

```python
import torch
print(torch.version.cuda)
```

输出为：

```text
12.8
```

这说明当前 PyTorch CUDA 版本与后续 vLLM 在 Blackwell 环境下的部署方向是匹配的。

---

## 4. 到目前为止的结论

### 4.1 不是继续写本地直接加载脚本

现在不建议继续写“每次运行脚本都重新加载模型”的测试方式。

原因：

1. `Qwen3-Next-80B-A3B-Instruct-FP8` 体量较大。
2. 每次测试都重新加载模型会非常慢。
3. 不利于后续批量评测、A/B 对比、非交互数据集测试。

### 4.2 正确做法：服务化

下一步应该把该模型做成 **常驻本地服务**：

- 模型只在服务启动时加载一次
- 后续测试脚本通过 HTTP API 调用
- 评测脚本不再直接加载模型
- 更适合 Day 6 / Day 7 这种工程化评测方式

---

## 5. 下一步总任务

下一步的核心任务可以概括为一句话：

> **把 `Qwen3-Next-80B-A3B-Instruct-FP8` 部署成常驻本地 API 服务，并让后续测试脚本通过该 API 做非交互评测。**

---

## 6. 下一步工作报告（写给低级小模型）

下面这部分是可以直接交给低能力模型执行的任务描述。

---

# 任务名称

部署 `Qwen3-Next-80B-A3B-Instruct-FP8` 为本地 OpenAI 兼容 API 服务，并准备非交互测试。

# 已知条件

1. 模型已经下载完成。
2. 模型目录为：

```text
/mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8
```

3. 当前 Python 环境中 `torch.version.cuda == 12.8`
4. 当前机器有 4 张大显存 GPU。

# 目标

完成以下 4 个子任务：

## 子任务 1：安装服务化依赖

优先尝试安装：

```bash
pip install -U vllm openai
```

如果已有环境冲突，不要乱删环境，优先记录报错并单独建立服务环境。

## 子任务 2：启动 vLLM 服务

使用本地模型目录启动常驻服务，建议命令如下：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --enable-prefix-caching
```

如果后续要测试函数调用，再使用增强版参数：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve /mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --enable-prefix-caching \
  --tool-call-parser hermes \
  --enable-auto-tool-choice
```

## 子任务 3：验证 API 是否启动成功

先用 curl 验证：

```bash
curl http://127.0.0.1:8000/v1/models
```

如果能返回模型列表，说明服务启动成功。

## 子任务 4：编写最小非交互测试脚本

使用 OpenAI 兼容写法，发一条最小请求验证服务：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="dummy"
)

resp = client.chat.completions.create(
    model="/mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    messages=[
        {"role": "system", "content": "你是一个简洁的中文助手。"},
        {"role": "user", "content": "请用一句话解释什么是 agent。"}
    ],
    temperature=0,
    max_tokens=128,
)

print(resp.choices[0].message.content)
```

如果该脚本能正常返回文本，说明后续可以继续接入批量评测。

---

## 7. 下一阶段建议顺序

建议严格按照下面顺序推进，不要跳步：

### 第 1 步：先服务化

先确认 vLLM 服务能起来。

### 第 2 步：再做最小 API smoke test

确认服务可调用。

### 第 3 步：再把 Day 6 / Day 7 评测脚本改成 API 调用版

不要一开始就把所有评测代码一起改掉。

### 第 4 步：最后做新旧模型 A/B 对比

对比对象：

- 旧模型：`Qwen2.5-7B-Instruct`
- 新模型：`Qwen3-Next-80B-A3B-Instruct-FP8`

重点看：

1. `pass_rate`
2. `route_error` 是否下降
3. `model_json` 路由成功率是否提升
4. `memory / summary / ellipsis / boundary` 标签表现是否改善

---

## 8. 风险提醒

### 风险 1：vLLM 安装失败

如果 `vllm` 安装失败，不要立刻改回本地 Transformers 直跑。

优先做法：

1. 记录完整报错
2. 确认当前 Python / CUDA / torch 版本
3. 单独建立服务环境重试

### 风险 2：服务能启动但请求失败

优先检查：

1. 端口是否被占用
2. `model` 字段是否填写为服务可识别的模型名
3. 是否真的返回了 `/v1/models`

### 风险 3：多卡张量并行异常

如果 `--tensor-parallel-size 4` 启动失败，可以临时：

1. 检查 4 张卡是否可见
2. 检查 NCCL / 驱动问题
3. 记录日志，不要盲目继续改脚本

---

## 9. 本次阶段的最终结论

本次阶段已经完成：

- 更强模型下载
- 本地目录准备完成
- 关键配置文件确认完整
- 下一步部署路线明确

下一步不再重复研究模型下载，而是进入：

> **本地服务化部署 + API 非交互测试 + 工程化 A/B 评测**

