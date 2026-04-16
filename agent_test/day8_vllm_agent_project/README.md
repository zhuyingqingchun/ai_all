# Day 8 vLLM Agent Engineering Project

这是把 `day7_vllm_agent_engineering_eval.py` 拆分后的工程化版本，目标是：

- 保留 Day 7 已验证通过的单步 agent 评测链路
- 提升模块复用性与可维护性
- 为 Day 8/Day 9 的多步任务、A/B 评测、真实工具接入提供干净底座

## 运行

先确认本地 vLLM 服务已启动：

```bash
curl http://127.0.0.1:8000/v1/models
```

再运行评测：

```bash
PYTHONPATH=src python -m vllm_agent_eval.cli --model next80b_fp8 --debug
```

或者安装后：

```bash
pip install -e .
vllm-agent-eval --model next80b_fp8 --debug
```
