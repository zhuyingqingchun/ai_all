# 第12轮：环境配置文件自动加载补丁

## 目标
把快速测试从手动 `export` 改成环境配置文件自动加载。

## 修改内容
- `run_quick_eval.py`：新增 `.env` 风格文件读取逻辑，并在导入 `agent` 之前加载环境变量。
- `local_quick_eval_config.json`：新增 `env_file` 字段，默认指向 `.env.quick`。
- `.env.quick`：本地调试默认环境文件。
- `.env.quick.example`：环境文件模板。

## 使用方法
编辑项目根目录下的 `.env.quick`：

```bash
DEBUG_MODEL_ID=Doubao-1-5-vision-pro-32k
VLM_API_KEY=你的真实Key
# DEBUG_API_URL=https://ark.cn-beijing.volces.com/api/v3
```

然后直接执行：

```bash
python run_quick_eval.py
```

或：

```bash
python run_quick_eval.py --env-file .env.quick
```

## 兼容性说明
这次只改了本地快速回归脚本，不改 `agent_base.py` 的受保护逻辑；`agent.py` 仍通过 `BaseAgent._call_api()` 调模型，因此不会和主办方替换后的 `agent_base.py` 冲突。

## 校验结果
- `git apply --check`：通过
- `git apply`：通过
- `py_compile`：通过
- `python run_quick_eval.py --help`：通过

## help 输出摘录
```text
usage: run_quick_eval.py [-h] [--config CONFIG] [--data_dir DATA_DIR]
                         [--env-file ENV_FILE]

仅跑 1~2 个样本的本地快速回归脚本

options:
  -h, --help           show this help message and exit
  --config CONFIG      快速回归配置文件
  --data_dir DATA_DIR  完整离线数据集目录
  --env-file ENV_FILE  可选：显式指定环境变量文件路径
```
