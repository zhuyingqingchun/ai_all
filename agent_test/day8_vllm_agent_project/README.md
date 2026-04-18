# vLLM Agent Eval - 多步评估与 LangGraph 迁移实验

## 项目简介

本项目是一个针对 vLLM Agent 的多步评估框架，当前主线已经从 Day 8 推进到 Day 16.1。
核心目标：
1. 构建多步工具调用评估链路（planner → executor → synthesizer）
2. 探索 LangGraph 作为替代执行后端的可行性
3. 在同一数据集与同一 gate 下完成 baseline 与 LangGraph parity 对齐

## 环境信息

### 系统环境
| 项目 | 版本 |
|------|------|
| 操作系统 | Ubuntu 25.10 (Questing Quokka) |
| Python | 3.13.11 |
| Conda 环境 | `swdtorch12` |

### GPU 环境
| 项目 | 规格 |
|------|------|
| GPU 型号 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition × 4 |
| 单卡显存 | 96 GB |
| 驱动版本 | 580.126.09 |

## 目录结构

```text
agent_test/day8_vllm_agent_project/
├── datasets/                         # 实验数据集
├── results/                          # 实验结果输出
├── scripts/                          # 运行脚本
└── src/vllm_agent_eval/              # 评测框架源码
```

## 快速开始

### 激活环境
```bash
conda activate swdtorch12
```

### 运行多步评估
```bash
cd agent_test/day8_vllm_agent_project
bash scripts/run_day8_multistep.sh
```

### 运行 LangGraph 正式实验
```bash
bash scripts/run_day15_langgraph_formal.sh next80b_fp8
```

### 运行 Day 16 parity 对齐实验
```bash
bash scripts/run_day16_langgraph_parity.sh next80b_fp8
```

## 开发里程碑

| 阶段 | 内容 | 状态 |
|------|------|------|
| Day 8 | 多步评估框架基础 | ✅ |
| Day 9 | 数据集护栏 | ✅ |
| Day 10 | 断言评估 | ✅ |
| Day 12 | LangGraph 并行对比（3 样本） | ✅ |
| Day 13 | LangGraph 扩展对比（8 样本 + 结构化指标） | ✅ |
| Day 14 | LangGraph Runner 适配器（4 样本） | ✅ |
| Day 15 | LangGraph 正式迁移实验（4 样本 + 断言检查） | ✅ |
| Day 16 | LangGraph parity migration 与 baseline 能力对齐 | ✅ |
| Day 16.1 | planner 表达式清理、README 恢复、路径相对化修复 | ✅ |

## Day 16.1 修复说明

本轮修复包含三项内容：
1. 修复数学 step 与 final-only hint 混入同一 clause 时的表达式污染问题
2. 恢复项目 README，避免远程仓库缺少说明文档
3. 将 Day 16 parity summary 中的本地绝对路径改为相对路径，提升可移植性
