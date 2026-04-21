from __future__ import annotations

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="仅跑 1~2 个样本的本地快速回归脚本")
    parser.add_argument("--config", default="local_quick_eval_config.json", help="快速回归配置文件")
    parser.add_argument("--data_dir", default="./test_data/offline", help="完整离线数据集目录")
    parser.add_argument("--env-file", default=None, help="可选：显式指定环境变量文件路径")
    return parser.parse_args()


def main() -> None:
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

    if not selected_cases:
        raise ValueError("selected_cases 不能为空")
    if len(selected_cases) > 2:
        raise ValueError("这个快速脚本只建议跑 1~2 个样本")

    src_root = Path(args.data_dir).resolve()
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="gui_quick_eval_") as temp_dir:
        temp_root = Path(temp_dir)
        for case_name in selected_cases:
            src_case = src_root / case_name
            if not src_case.exists():
                raise FileNotFoundError(f"样本不存在: {src_case}")
            shutil.copytree(src_case, temp_root / case_name)

        agent = Agent()
        # 关键点：当前官方 test_runner.py 的构造函数只接受 (agent, debug_test)
        runner = TestRunner(agent, debug_test=debug_test)
        results = runner.run_all_tasks(data_dir=str(temp_root), output_dir=output_dir)

    print("\n" + "=" * 50)
    print("快速回归完成")
    print(f"env_file: {env_file}")
    print(f"selected_cases: {selected_cases}")
    print(f"DEBUG_MODEL_ID: {os.environ.get('DEBUG_MODEL_ID', '')}")
    print(f"total_cases: {results['total_cases']}")
    print(f"passed_cases: {results['passed_cases']}")
    print(f"case_accuracy: {results['case_accuracy']:.2%}")
    if debug_test and "step_accuracy" in results:
        print(f"step_accuracy: {results['step_accuracy']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
