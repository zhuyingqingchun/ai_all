# GLM-4.7-20B Agent

基于GLM-4.7-20B模型的智能Agent，支持工具调用、多步规划和反思闭环。

## 功能特性
- 🤖 基于GLM-4.7-20B大语言模型
- 🔧 支持多种工具调用（搜索、计算、天气查询）
- 📋 多步规划能力
- 🔄 反思闭环机制
- 💾 对话记忆功能
- 🚀 本地一键运行

## 环境要求
- Python 3.10+
- CUDA 11.7+
- 至少32GB显存（推荐64GB以上）

## 快速开始

### 1. 克隆仓库
```bash
git clone <repository-url>
cd GLM-4.7-20B-Agent
```

### 2. 创建并激活conda环境
```bash
conda env create -f environment.yml
conda activate glm4-agent
```

### 3. 运行Agent
```bash
python main.py
```

## 项目结构
```
.
├── main.py          # 主脚本
├── environment.yml  # conda环境配置
├── .gitignore       # git忽略文件
└── README.md        # 项目说明
```

## 使用示例

```bash
用户: 北京今天的天气怎么样？
助手: 正在查询北京的天气...
北京 天气:
温度: 15°C
天气: 晴
湿度: 45%
风速: 10 km/h

用户: 计算 12345 * 67890
助手: 正在计算...
计算结果: 12345 * 67890 = 838102050

用户: 搜索最新的人工智能技术
助手: 正在搜索...
搜索结果 for '最新的人工智能技术':
1. 2024年十大人工智能技术趋势 - 知乎
   https://www.zhihu.com/question/650000000
2. 2024年人工智能技术发展报告
   https://www.example.com/report
3. 最新AI技术突破：生成式AI、大语言模型等
   https://www.example.com/ai-breakthroughs
```

## 模型说明
- 模型: GLM-4.7-20B
- 来源: THUDM
- 下载方式: 自动从国内镜像下载

## 注意事项
1. 首次运行会自动下载模型，可能需要较长时间
2. 确保有足够的显存空间
3. 工具调用可能会受到网络限制
4. 对话历史会保存在内存中，重启程序后会清空

## 许可证
本项目基于MIT许可证开源。
