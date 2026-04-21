#!/usr/bin/env python3
"""仅跑 1~2 个样本的本地快速回归脚本"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def _load_env_file(env_path: str | None) -> None:
    if not env_path:
        return
    path = Path(env_path).expanduser()
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description="仅跑 1~2 个样本的本地快速回归脚本")
    parser.add_argument("--config", default="local_quick_eval_config.json", help="快速回归配置文件")
    parser.add_argument("--data_dir", default="./test_data/offline", help="完整离线数据集目录")
    parser.add_argument("--env-file", default=None, help="可选：显式指定环境变量文件路径")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    env_file = args.env_file or cfg.get("env_file") or ".env.quick"
    _load_env_file(env_file)

    from agent import Agent
    from test_runner import TestRunner

    selected_cases = cfg.get("selected_cases") or []
    output_dir = cfg.get("output_dir", "./output_quick")
    debug_test = bool(cfg.get("debug_test", True))

    # 创建临时目录作为工作目录
    work_dir = tempfile.mkdtemp(prefix="quick_eval_")
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 创建 Agent 实例
        agent = Agent()

        # 创建 TestRunner
        runner = TestRunner(
            agent=agent,
            data_dir=args.data_dir,
            output_dir=output_dir,
            work_dir=work_dir,
            debug_test=debug_test
        )

        # 运行选中的用例
        results = runner.run_selected_cases(selected_cases)

    finally:
        # 清理临时目录
        shutil.rmtree(work_dir, ignore_errors=True)

    print("\n" + "=" * 50)
    print("快速回归完成")
    print(f"env_file: {env_file}")
    print(f"selected_cases: {selected_cases}")
    print(f"DEBUG_MODEL_ID: {os.environ.get('DEBUG_MODEL_ID', '')}")
    print(f"total_cases: {results['total_cases']}")
    print(f"passed_cases: {results['passed_cases']}")
    print(f"case_accuracy: {results['case_accuracy']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
