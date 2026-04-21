# 第7轮：xxxx 目录下比赛内容本地测试操作步骤

## 目标
在 `/mnt/PRO6000_disk/swd/agent/xxxx/` 目录下，使用：
1. 官方参考附件 `1776137151181_bf4cdr2t_6915.zip`
2. 已生成的提交包 `submission_ark_api_small.zip`
完成一次本地离线测试。

## 目录判断
- `xxxx/` 里放的是比赛相关压缩包与校验文件。
- 真正的多步工具实验项目在 `agent_test/day8_vllm_agent_project/`，不是这次 GUI 比赛本地测试入口。
- GUI 比赛本地测试应该以官方附件中的 `code-for-student/test_runner.py` 为准。

## 最短测试路径
1. 进入 `xxxx/`
2. 解压官方附件到一个工作目录
3. 解压 submission 包
4. 用 submission 里的 `src/` 覆盖官方 `code-for-student/` 对应位置
5. 安装依赖
6. 设置 `VLM_API_KEY`
7. 运行 `test_runner.py`
8. 查看 `output/` 中结果

## 建议命令
```bash
cd /mnt/PRO6000_disk/swd/agent/xxxx
mkdir -p gui_competition_eval
cd gui_competition_eval
unzip ../1776137151181_bf4cdr2t_6915.zip
unzip ../submission_ark_api_small.zip
cp -f submission/src/agent.py code-for-student/
cp -f submission/src/requirements.txt code-for-student/
cp -f submission/src/utils/action_parser.py code-for-student/utils/
cp -f submission/src/utils/state_manager.py code-for-student/utils/
```

## 依赖安装
```bash
cd /mnt/PRO6000_disk/swd/agent/xxxx/gui_competition_eval/code-for-student
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 设置 API Key
```bash
export VLM_API_KEY=你的火山ARK密钥
```

## 运行离线测试
```bash
python test_runner.py
```

## 输出位置
- 日志：`code-for-student/output/test_run.log`
- 汇总：`code-for-student/output/result.xlsx`
- 明细：`code-for-student/output/results.json`

## 快速自检
1. `agent.py` 已覆盖到 `code-for-student/agent.py`
2. `utils/action_parser.py` 与 `utils/state_manager.py` 已复制
3. `VLM_API_KEY` 已设置
4. 当前目录是 `code-for-student/`
5. Python 版本尽量用 3.10.12

## 常见错误
- 在 `agent_test/day8_vllm_agent_project/` 里跑：这是另一条实验线，不是 GUI 比赛入口。
- 忘记解压 `submission_ark_api_small.zip`：会没有 `agent.py`。
- 没有设置 `VLM_API_KEY`：模型调用会失败。
- 没有进入 `code-for-student/`：`test_runner.py` 的相对路径会找不到 `./test_data/offline`。

## 补充
如果你只是想先测接口链路，不想消耗真实 API，可以把之前做的 fake API 调试方案单独接进去；但真正接近比赛环境的本地测试，还是应当直接使用官方附件 + 官方 `test_runner.py` + 官方默认 ARK 接口。
