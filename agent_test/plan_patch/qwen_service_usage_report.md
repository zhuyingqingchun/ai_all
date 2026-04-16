# qwen_service 项目学习与使用报告

## 1. 项目定位

这个项目是一个 **Qwen 本地模型服务化工具**。核心目标不是训练模型，也不是做前端页面，而是：

- 读取本地模型路径和服务配置
- 用 `vLLM` 将模型启动为 **OpenAI 兼容 API 服务**
- 提供启动、停止、状态检查、健康自检、接口测试等一整套脚本
- 方便后续其他 Python 脚本通过 HTTP/OpenAI SDK 调用模型，而不用每次重复加载模型

一句话概括：

> 这是一个“把本地 Qwen 模型包装成常驻 API 服务”的脚手架项目。

---

## 2. 项目结构与作用

解压后的核心文件如下：

- `run_server.py`：启动服务
- `stop_server.py`：停止服务
- `status_server.py`：查看服务状态
- `self_check.py`：做环境/模型/端口/健康检查
- `test_openai_client.py`：用 OpenAI 客户端做最小调用测试
- `config.py`：加载并校验 `service_config.json`
- `utils.py`：端口、PID、健康检查、启动命令、日志等工具函数
- `service_config.json`：服务配置文件
- `logs/qwen_service.log`：运行日志

这套结构很适合后续让“小模型”继续维护，因为职责比较清晰，全部是 Python 脚本，没有额外 shell 框架依赖。

---

## 3. 代码工作流理解

项目的主流程是：

1. `run_server.py` 读取 `service_config.json`
2. 检查：
   - Python 解释器是否存在
   - 模型目录和关键文件是否存在
   - `vllm`、`requests` 是否已安装
   - 目标端口是否空闲
3. 调用 `utils.generate_start_command()` 拼接启动命令：
   - `python -m vllm.entrypoints.openai.api_server ...`
4. 后台启动进程，写 PID 到 `qwen_service.pid`
5. 循环探测 `http://host:port/v1/models` 是否返回 200
6. 成功后即可被其他脚本通过 OpenAI SDK 调用

停止流程：

1. `stop_server.py` 读取 PID 文件
2. 先发 `SIGTERM`
3. 超时不退出再 `SIGKILL`
4. 删除 PID 文件

状态检查流程：

1. `status_server.py` 读取 PID
2. 判断进程是否存在
3. 判断端口是否监听
4. 请求 `/v1/models` 检查接口是否正常

---

## 4. 实际学习结论：这个项目已经具备什么能力

### 已具备的能力

- 本地模型服务启动/停止/状态检查
- OpenAI 兼容接口暴露
- PID 管理
- 健康检查
- 最小接口 smoke test
- 环境自检

### 当前更适合的使用场景

- 你本地把大模型加载为常驻服务
- 后续评测脚本、agent 测试脚本、批量问答集脚本都直接调 API
- 避免每个脚本里重复加载大模型

这正适合你当前的目标：

> “模型只加载一次，后续非交互脚本直接调本地 API 进行测试。”

---

## 5. 从日志看到的实际运行情况

项目日志 `logs/qwen_service.log` 显示：

### 成功事实

- vLLM 已成功启动
- 模型路径为：
  `/mnt/PRO6000_disk/models/Qwen/Qwen3-Next-80B-A3B-Instruct-FP8`
- 服务成功监听：
  `http://127.0.0.1:8000`
- `/v1/models` 返回 200
- `/v1/chat/completions` 已经被正常调用多次
- engine 初始化成功，日志显示 warmup/model init 完成

### 可确认的结论

这个项目 **不是停留在“写好了脚本”阶段**，而是已经至少成功启动并跑通过接口测试。

---

## 6. 当前配置与机器条件的匹配情况

### 当前配置文件内容

`service_config.json` 关键项：

- `model_path`: 已正确指向你的 80B FP8 模型
- `served_model_name`: `next80b_fp8`
- `host`: `127.0.0.1`
- `port`: `8000`
- `tensor_parallel_size`: **1**
- `max_model_len`: **8192**
- `dtype`: `auto`

### 和你机器的关系

你现在有 **4 张 RTX PRO 6000 Blackwell**，每张约 96GB 显存。

因此从工程角度看：

#### 当前配置能不能用？
能用，日志也表明已经成功运行。

#### 当前配置是不是最优？
不是。

### 我对当前配置的判断

1. **`tensor_parallel_size=1` 偏保守**
   - 对 80B FP8 来说，单卡模式不是最适合你的机器资源的用法
   - 更合理的是尝试 `tensor_parallel_size=4`

2. **`max_model_len=8192` 偏小**
   - 对 Qwen3-Next-80B-A3B-Instruct 来说，这只是保守起步值
   - 如果后续要做长上下文 agent / 文档问答 /摘要记忆，后面可以逐步上调

3. **项目目前是“基础服务版”**
   - 启动命令里还没有加：
     - `--enable-prefix-caching`
     - tool calling 相关参数
     - 更细的日志/性能开关
   - 但作为第一版常驻服务脚手架，这样已经够用了

---

## 7. 已发现的注意点与潜在问题

### 7.1 404 不是主问题
日志里有：

- `GET /v1/v1/models -> 404`
- `GET /v1 -> 404`

这更像是某次手动或脚本调用时把 base_url 拼错了，不代表服务本身异常。因为正确的：

- `/v1/models` 已多次返回 200
- `/v1/chat/completions` 已多次返回 200

### 7.2 generation_config 覆盖提示
日志提示：

- 模型自带 `generation_config.json` 覆盖了 vLLM 默认采样参数

这不是故障，只是说明默认温度/top_p/top_k 取的是模型自己的值。如果你后续测试需要强可复现性，建议在客户端显式传采样参数。

### 7.3 输入 shape warning
日志中还有类似：

- `Input tensor shape suggests potential format mismatch`

目前它没有阻止服务工作，因为实际接口请求已经成功返回结果。现阶段可作为观察项，不作为阻塞问题。

### 7.4 配置里没有多卡并行开关优化
当前代码的 `generate_start_command()` 只拼了最基础参数，没有额外优化项。后续如果做正式长期服务，建议扩展。

---

## 8. 怎么使用这个项目

### 启动服务
```bash
cd qwen_service
python run_server.py
```

### 查看状态
```bash
python status_server.py
```

### 测试接口
```bash
python test_openai_client.py
```

### 停止服务
```bash
python stop_server.py
```

### 环境自检
```bash
python self_check.py
```

---

## 9. 适合你的实际使用方式

你当前最适合按下面方式使用：

1. 用这个项目先把模型服务常驻起来
2. 确保 `status_server.py` 和 `test_openai_client.py` 通过
3. 再写你自己的：
   - 非交互问答集测试脚本
   - agent 路由评测脚本
   - A/B 模型对比脚本
   - 长上下文/记忆测试脚本
4. 这些脚本统一走：
   - `http://127.0.0.1:8000/v1`
   - OpenAI SDK 或 requests

这样模型只加载一次，最省时间，也最适合大模型评测。

---

## 10. 简要使用报告

### 结论

这个项目是一个 **Qwen 本地模型 API 服务化脚手架**，主要用途是把本地模型启动成常驻 OpenAI 兼容服务，便于后续各种测试脚本直接调用。

### 当前状态

- 项目结构清晰
- 配置方式简单
- 核心功能完整
- 日志表明服务已经真实启动成功过
- 可以满足“模型加载一次、后续脚本反复调用”的需求

### 当前优点

- 纯 Python 核心脚本，容易维护
- 不依赖 `conda activate`
- 生命周期管理完整
- 很适合作为你后续 agent 工程化测试的底座

### 当前不足

- 当前配置还偏保守，没有充分利用 4 张 GPU
- 启动参数还可以继续增强
- 当前版本更像“基础服务版”，不是“高性能调优版”

---

## 11. 下一步建议

### 优先级 1：先确认基础服务稳定
建议顺序：

1. `python self_check.py`
2. `python run_server.py`
3. `python status_server.py`
4. `python test_openai_client.py`

### 优先级 2：再做参数升级
建议后续修改：

- `tensor_parallel_size: 4`
- 适当增大 `max_model_len`
- 后续可加 prefix caching 参数

### 优先级 3：再接你自己的评测脚本
例如：

- Day6/Day7 的 agent 工程化评测脚本改为 API 调用版
- 批量问答集测试
- 路由器能力评测
- 记忆/摘要能力评测

---

## 12. 给低级小模型的执行说明

你可以把下面这段直接交给低能力模型：

### 任务目标
把本地 Qwen 大模型作为常驻 API 服务运行，并验证服务可被后续 Python 脚本调用。

### 执行步骤
1. 进入项目目录 `qwen_service`
2. 检查 `service_config.json` 是否存在且路径正确
3. 运行 `python self_check.py`
4. 如果通过，运行 `python run_server.py`
5. 服务启动后运行 `python status_server.py`
6. 再运行 `python test_openai_client.py`
7. 如果一切正常，记录：
   - 服务地址
   - 模型名
   - 端口
   - 日志文件路径
8. 如果失败，优先检查：
   - 模型路径
   - python_bin
   - 端口是否被占用
   - vllm 是否已安装

### 当前建议修改项
- 把 `tensor_parallel_size` 从 `1` 调整为 `4`
- 后续根据实际需要再调整 `max_model_len`

---

## 13. 最终判断

这个项目 **值得继续用**，因为它已经具备你当前最需要的能力：

> 把本地大模型变成常驻服务，让后续 agent 和评测脚本直接调用 API。

如果你下一步继续做 agent 工程化，这个项目完全可以作为底座使用。
