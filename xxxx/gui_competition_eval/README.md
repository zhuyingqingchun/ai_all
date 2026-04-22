# GUI 智能体竞赛项目

## 项目概述

本项目是 GUI 智能体竞赛的参赛代码，实现了基于视觉语言模型（VLM）的安卓手机 GUI 自动化操作 Agent。

## 目录结构

```
xxxx/                           # 项目根目录
├── gui_competition_eval/       # GUI 竞赛主目录
│   ├── code-for-student/       # 主要开发和测试目录
│   │   ├── agent.py            # Agent 主实现（参赛者修改）
│   │   ├── agent_base.py       # Agent 基类（主办方提供）
│   │   ├── test_runner.py      # 测试运行器（主办方提供）
│   │   ├── run_quick_eval.py   # 快速回归测试脚本
│   │   ├── local_quick_eval_config.json   # 快速测试配置
│   │   ├── local_model_config.py          # 本地模型配置
│   │   ├── .env.quick.example  # 环境变量模板
│   │   ├── .env.quick          # 本地环境配置（gitignored）
│   │   ├── requirements.txt    # Python 依赖
│   │   ├── model.txt           # 可用模型列表
│   │   ├── test_data/          # 测试数据集
│   │   │   └── offline/        # 离线测试数据
│   │   │       ├── step_meituan_onekey_0001/   # 美团测试用例
│   │   │       ├── step_baidumap_onekey_0008/  # 百度地图测试用例
│   │   │       └── ...         # 其他测试用例
│   │   ├── utils/              # 工具模块
│   │   │   ├── action_parser.py    # 动作解析器（支持容错解析）
│   │   │   ├── state_manager.py    # 状态管理器（含任务先验）
│   │   │   ├── cache_store.py      # 缓存存储
│   │   │   ├── checkpoint_store.py # 检查点存储
│   │   │   ├── observability.py    # 可观测性（trace/metrics）
│   │   │   ├── image_utils.py      # 图像工具
│   │   │   ├── visualize_ref.py    # 可视化工具
│   │   │   └── __init__.py
│   │   ├── output/             # 完整测试结果输出
│   │   │   ├── result.xlsx
│   │   │   └── test_run.log
│   │   └── output_quick/       # 快速测试结果输出
│   │       ├── trace.jsonl     # 执行轨迹日志
│   │       ├── metrics.json    # 指标统计
│   │       ├── checkpoints/    # 步骤检查点
│   │       ├── cache/          # 模型响应缓存
│   │       ├── visualization/  # 可视化报告
│   │       └── result.xlsx
│   │
│   ├── submission/             # 最终提交目录
│   │   ├── src/                # 提交源代码
│   │   │   ├── agent.py
│   │   │   ├── agent_base.py
│   │   │   ├── utils/
│   │   │   └── requirements.txt
│   │   └── doc/                # 文档
│   │       └── 算法设计说明文档.md
│   │
│   └── README.md               # 本文件
│
└── patch/                      # 开发补丁记录
    ├── session_12_envfile_quick_eval.patch
    ├── session_18_gui_action_system_prompt_v2.txt
    ├── session_21_engineered_agent_patch_git.patch
    ├── session_27_engineering_suite_report.md
    ├── session_28_dataset_aligned_prompt.txt
    ├── session_34_engineering_fix.patch
    ├── session_34_engineering_fix_report.md
    ├── session_38_task_specific_prior_patch.patch
    ├── session_38_task_specific_prior_report.md
    └── glm官方/
        └── session_36_glm46v_minimal_patch.patch
```

## 快速开始

### 1. 环境配置

```bash
cd code-for-student

# 创建环境文件
cp .env.quick.example .env.quick

# 编辑 .env.quick，填入你的 API Key
# DEBUG_MODEL_ID=doubao-1-5-vision-pro-32k-250115
# VLM_API_KEY=your-api-key-here
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 快速测试

```bash
# 运行单个样本快速测试
python run_quick_eval.py

# 或指定环境文件
python run_quick_eval.py --env-file .env.quick
```

### 4. 完整测试

```bash
# 运行完整测试集
python test_runner.py
```

## 核心文件说明

### agent.py
Agent 主实现，包含：
- `_build_system_prompt()`: 构建系统提示词
- `generate_messages()`: 生成 LLM 消息
- `act()`: 执行动作决策

### utils/action_parser.py
动作解析器，支持多种输出格式：
- JSON 格式: `{"action":"CLICK","parameters":{"point":[x,y]}}`
- 简洁格式: `CLICK:[[x,y]]`
- 其他变体格式

### utils/state_manager.py
状态管理器，负责：
- 历史步骤记录
- 动作去重
- 首步启发式规则

## 测试数据

测试数据位于 `code-for-student/test_data/offline/`，包含多个测试用例：
- `step_meituan_onekey_0001`: 美团一键下单
- 其他测试用例...

## 输出结果

测试结果保存在：
- `output/result.xlsx`: 完整测试结果
- `output_quick/result.xlsx`: 快速测试结果
- `output_quick/visualization/`: 可视化报告

## 模型配置

当前使用的模型：
- **模型 ID**: `doubao-1-5-vision-pro-32k-250115`
- **API 地址**: `https://ark.cn-beijing.volces.com/api/v3`
- **支持模态**: text, image

可用模型列表见 `model.txt`。

## 提交要求

1. 将 `code-for-student/agent.py` 复制到 `submission/src/agent.py`
2. 确保 `submission/src/utils/` 下的工具模块完整
3. 填写 `submission/doc/算法设计说明文档.md`
4. 提交整个 `submission/` 目录

## 注意事项

1. **不要修改** `agent_base.py` 和 `test_runner.py`（主办方会替换）
2. **可以修改** `agent.py` 和 `utils/` 下的工具模块
3. 本地测试时使用自己的 API Key，提交后使用主办方统一配置
4. 保持代码兼容 Python 3.10+

## 开发记录

| 轮次 | 日期 | 主要内容 | 关键改进 |
|:----:|:-----|:---------|:---------|
| 12 | 2025-04-21 | 环境配置优化 | 添加 `.env.quick` 自动加载，避免手动 export |
| 18 | 2025-04-21 | Prompt 优化 | 强化决策约束，禁止过早 COMPLETE |
| 21 | 2025-04-21 | Agent 工程化 | 状态管理、动作去重、首步启发式 |
| 27 | 2025-04-21 | 可观测性套件 | trace.jsonl、metrics.json、cache、checkpoint |
| 28 | 2025-04-21 | Prompt 对齐 | 与官方数据集对齐的 System Prompt |
| 34 | 2025-04-21 | 工程修复 | Parser 容错、禁缓存评测、TYPE 约束、page-aware fallback |
| 38 | 2025-04-21 | 任务专项优化 | 美团/百度地图先验锚点、分阶段子目标、自动 COMPLETE |

### 性能演进

| 阶段 | 步骤准确率 | 主要问题 |
|:-----|:----------:|:---------|
| 初始版本 | 21.4% | 过早 COMPLETE、坐标偏差 |
| 第34轮 | 35.7% | Parser 容错、TYPE 约束 |
| 第38轮 | **57.14%** | 任务先验、分阶段优化 |

### 当前瓶颈

- **坐标精度**: 占 85% 错误，模型无法精确定位 UI 元素
- **页面阶段误判**: Step 3/4 重复点击
- **动作序列偏差**: 与官方轨迹存在时序差异

## 联系方式

如有问题，请参考主办方提供的文档或联系竞赛组委会。
