# vLLM Agent Eval - 多步评估与 LangGraph 迁移实验

## 项目简介

本项目是一个针对 vLLM Agent 的多步评估框架，包含从 Day 8 到 Day 15 的渐进式开发实验。核心目标：
1. 构建多步工具调用评估链路（planner → executor → synthesizer）
2. 探索 LangGraph 作为潜在替代方案的可行性（Day 12-15）

## 环境信息

### 系统环境
| 项目 | 版本 |
|------|------|
| 操作系统 | Ubuntu 25.10 (Questing Quokka) |
| Python | 3.13.11 |
| Conda 环境 | `swdtorch12` (`/home/a/miniconda3/envs/swdtorch12`) |

### GPU 环境
| 项目 | 规格 |
|------|------|
| GPU 型号 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition × 4 |
| 单卡显存 | 96 GB (97887 MiB) |
| 驱动版本 | 580.126.09 |

## 目录结构

项目主目录位于 `agent_test/day8_vllm_agent_project/`：

```
agent_test/day8_vllm_agent_project/
├── archive/                          # 归档目录
│   ├── docs/                         # 历史文档
│   └── patches/                      # 历史补丁
├── datasets/                         # 测试数据集
│   ├── day8_multistep_dataset.json          # Day 8 多步数据集
│   ├── day8_multistep_regression_cases.json # Day 8 回归测试用例
│   ├── day8_multistep_regression_suite.json # Day 8 回归测试套件
│   ├── default_day8_dataset.json            # 默认数据集
│   ├── day10_1_assertion_dataset.json       # Day 10 断言数据集
│   ├── day12_langgraph_compare_subset.json  # Day 12 LangGraph 对比子集
│   ├── day13_langgraph_compare_extended.json# Day 13 LangGraph 扩展对比
│   ├── day14_langgraph_migration_subset.json# Day 14 LangGraph 迁移子集
│   └── day15_langgraph_formal_subset.json   # Day 15 LangGraph 正式实验
├── results/                          # 运行结果输出目录
│   ├── day8_2/                       # Day 8 结果
│   ├── day10_1/                      # Day 10 结果
│   ├── day12_compare/                # Day 12 对比结果
│   ├── day13_compare/                # Day 13 对比结果
│   ├── day14_langgraph_runner/       # Day 14 LangGraph runner 结果
│   └── day15_langgraph_formal/       # Day 15 正式实验结果
├── scripts/                          # 运行脚本
│   ├── archive/                      # 归档脚本
│   ├── run_day8_multistep.sh         # Day 8 多步运行脚本
│   ├── run_day9_dataset_guardrails.sh# Day 9 数据集护栏脚本
│   ├── run_day10_1_assertion.sh      # Day 10 断言运行脚本
│   ├── run_day12_langgraph_compare.sh# Day 12 对比运行脚本
│   ├── run_day13_langgraph_compare.sh# Day 13 扩展对比运行脚本
│   ├── run_day14_langgraph_runner.sh # Day 14 LangGraph runner 脚本
│   ├── run_day15_langgraph_formal.sh # Day 15 正式实验脚本
│   └── ...                           # 其他脚本
└── src/vllm_agent_eval/              # 源代码
    ├── __init__.py                   # 包初始化
    ├── cli.py                        # CLI 入口
    ├── cli_multistep.py              # 多步 CLI
    ├── cli_multistep_suite.py        # 多步套件 CLI
    ├── cli_langgraph_poc.py          # LangGraph PoC CLI
    ├── cli_langgraph_compare.py      # LangGraph 对比 CLI
    ├── cli_day13_langgraph_compare.py# Day 13 对比 CLI
    ├── cli_langgraph_runner_adapter.py# LangGraph runner 适配器 CLI
    ├── cli_day15_langgraph_formal_runner.py # Day 15 正式运行 CLI
    ├── client.py                     # 客户端
    ├── config.py                     # 配置
    ├── dataset.py                    # 数据集处理
    ├── evaluator.py                  # 评估器
    ├── tools.py                      # 工具定义
    ├── routing.py                    # 路由逻辑
    ├── schemas.py                    # 数据结构定义
    ├── prompts.py                    # 提示词
    ├── memory.py                     # 记忆模块
    ├── assertion_matchers.py         # 断言匹配器
    ├── assertion_guardrails.py       # 断言护栏
    ├── multistep_dataset.py          # 多步数据集
    ├── multistep_evaluator.py        # 多步评估器
    ├── multistep_schemas.py          # 多步数据结构
    ├── multistep_prompts.py          # 多步提示词
    ├── multistep_patch_helpers.py    # 多步补丁辅助
    ├── langgraph_poc.py              # LangGraph PoC
    ├── langgraph_nodes.py            # LangGraph 节点
    ├── langgraph_state.py            # LangGraph 状态
    ├── langgraph_runner_adapter.py   # LangGraph runner 适配器
    ├── day12_langgraph_compare.py    # Day 12 对比逻辑
    ├── day13_langgraph_compare_extended.py # Day 13 扩展对比
    ├── day13_compare_metrics.py      # Day 13 指标计算
    ├── day15_langgraph_formal_checks.py    # Day 15 正式检查
    └── day15_langgraph_formal_runner.py    # Day 15 正式运行
```

## 快速开始

### 激活环境
```bash
conda activate swdtorch12
```

### 进入项目目录
```bash
cd agent_test/day8_vllm_agent_project/
```

### 运行多步评估
```bash
bash scripts/run_day8_multistep.sh
```

### 运行 LangGraph 实验
```bash
# Day 12: 并行对比
bash scripts/run_day12_langgraph_compare.sh

# Day 13: 扩展对比
bash scripts/run_day13_langgraph_compare.sh

# Day 14: Runner 适配器
bash scripts/run_day14_langgraph_runner.sh

# Day 15: 正式迁移实验
bash scripts/run_day15_langgraph_formal.sh next80b_fp8
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
