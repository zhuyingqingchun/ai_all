# GUI 智能体竞赛项目

## 项目概述

本项目是 GUI 智能体竞赛的参赛代码，实现了基于视觉语言模型（VLM）的安卓手机 GUI 自动化操作 Agent。

## 目录结构

```
gui_competition_eval/
├── code-for-student/          # 主要开发和测试目录
│   ├── agent.py               # Agent 主实现（参赛者修改）
│   ├── agent_base.py          # Agent 基类（主办方提供）
│   ├── test_runner.py         # 测试运行器（主办方提供）
│   ├── run_quick_eval.py      # 快速回归测试脚本
│   ├── local_quick_eval_config.json  # 快速测试配置
│   ├── .env.quick.example     # 环境变量模板
│   ├── .env.quick             # 本地环境配置（gitignored）
│   ├── requirements.txt       # Python 依赖
│   ├── model.txt              # 可用模型列表
│   ├── test_data/             # 测试数据集
│   │   └── offline/           # 离线测试数据
│   ├── utils/                 # 工具模块
│   │   ├── action_parser.py   # 动作解析器
│   │   ├── state_manager.py   # 状态管理器
│   │   ├── image_utils.py     # 图像工具
│   │   ├── visualize_ref.py   # 可视化工具
│   │   └── __init__.py
│   ├── output/                # 完整测试结果输出
│   └── output_quick/          # 快速测试结果输出
│
├── submission/                # 最终提交目录
│   ├── src/                   # 提交源代码
│   │   ├── agent.py           # Agent 实现
│   │   ├── agent_base.py      # 基类
│   │   ├── utils/             # 工具模块
│   │   └── requirements.txt   # 依赖
│   └── doc/                   # 文档
│       └── 算法设计说明文档.md
│
└── README.md                  # 本文件
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

- **第12轮**: 添加环境配置文件自动加载
- **第17轮**: 优化 System Prompt 格式
- **第18轮**: 强化决策约束（禁止过早 COMPLETE、优化坐标精度）

## 联系方式

如有问题，请参考主办方提供的文档或联系竞赛组委会。
