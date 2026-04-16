#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def find_dynamic_literals_in_dataset(dataset_path: Path) -> dict[str, bool]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    has_regex_any = False
    has_regex_all = False
    has_forbidden = False

    if isinstance(data, list):
        for item in data:
            turns = item.get("turns", [])
            for turn in turns:
                if "expected_regex_any" in turn:
                    has_regex_any = True
                if "expected_regex_all" in turn:
                    has_regex_all = True
                if "forbidden_contains_any" in turn:
                    has_forbidden = True

    return {
        "expected_regex_any_present": has_regex_any,
        "expected_regex_all_present": has_regex_all,
        "forbidden_contains_any_present": has_forbidden,
    }


def count_sessions_and_turns(dataset_path: Path) -> tuple[int, int]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    session_count = 0
    turn_count = 0

    if isinstance(data, list):
        for item in data:
            session_count += 1
            turns = item.get("turns", [])
            turn_count += len(turns)

    return session_count, turn_count


def run_guardrails(project_root: Path) -> tuple[int, str, str]:
    script_path = project_root / "scripts" / "run_day9_dataset_guardrails.sh"
    if not script_path.exists():
        return 1, "", "guardrails script not found"

    result = subprocess.run(
        ["bash", str(script_path)],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def run_suite(project_root: Path, model: str) -> tuple[int, str, str]:
    script_path = project_root / "scripts" / "run_multistep_82_suite.sh"
    if not script_path.exists():
        return 1, "", "suite script not found"

    result = subprocess.run(
        ["bash", str(script_path), model],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    return result.returncode, result.stdout, result.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description="Day 10.1 验证脚本")
    parser.add_argument("--project-root", type=str, default=".", help="项目根目录")
    parser.add_argument("--dataset", type=str, default="datasets/day10_1_assertion_dataset.json", help="数据集路径")
    parser.add_argument("--model", type=str, default="next80b_fp8", help="模型别名")
    parser.add_argument("--output-dir", type=str, default="results/day10_1", help="输出目录")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    dataset_path = project_root / args.dataset
    output_dir = project_root / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Day10.1] 验证开始")
    print(f"[Day10.1] 项目根目录: {project_root}")
    print(f"[Day10.1] 数据集: {dataset_path}")
    print(f"[Day10.1] 模型: {args.model}")
    print(f"[Day10.1] 输出目录: {output_dir}")
    print()

    # 1. 检查数据集
    print("[Day10.1] Step 1: 检查数据集...")
    if not dataset_path.exists():
        print(f"[Day10.1] ERROR: 数据集不存在: {dataset_path}")
        return 1

    session_count, turn_count = count_sessions_and_turns(dataset_path)
    print(f"[Day10.1]   sessions: {session_count}")
    print(f"[Day10.1]   turns: {turn_count}")

    assertion_coverage = find_dynamic_literals_in_dataset(dataset_path)
    print(f"[Day10.1]   expected_regex_any: {assertion_coverage['expected_regex_any_present']}")
    print(f"[Day10.1]   expected_regex_all: {assertion_coverage['expected_regex_all_present']}")
    print(f"[Day10.1]   forbidden_contains_any: {assertion_coverage['forbidden_contains_any_present']}")
    print()

    # 2. 运行 guardrails
    print("[Day10.1] Step 2: 运行 guardrails...")
    guardrails_returncode, guardrails_stdout, guardrails_stderr = run_guardrails(project_root)
    guardrails_passed = guardrails_returncode == 0

    with (output_dir / "guardrails.stdout.txt").open("w", encoding="utf-8") as f:
        f.write(guardrails_stdout)
    with (output_dir / "guardrails.stderr.txt").open("w", encoding="utf-8") as f:
        f.write(guardrails_stderr)

    print(f"[Day10.1]   returncode: {guardrails_returncode}")
    print(f"[Day10.1]   passed: {guardrails_passed}")
    print()

    if not guardrails_passed:
        print("[Day10.1] ERROR: guardrails 失败")
        return 1

    # 3. 运行 suite
    print("[Day10.1] Step 3: 运行 suite...")
    suite_returncode, suite_stdout, suite_stderr = run_suite(project_root, args.model)
    suite_passed = suite_returncode == 0

    with (output_dir / "suite.stdout.txt").open("w", encoding="utf-8") as f:
        f.write(suite_stdout)
    with (output_dir / "suite.stderr.txt").open("w", encoding="utf-8") as f:
        f.write(suite_stderr)

    print(f"[Day10.1]   returncode: {suite_returncode}")
    print(f"[Day10.1]   passed: {suite_passed}")
    print()

    if not suite_passed:
        print("[Day10.1] ERROR: suite 失败")
        return 1

    # 4. 解析 suite 结果
    summary_path = output_dir / "day10_1_validation_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "project_root": str(project_root),
        "dataset": str(dataset_path),
        "model": args.model,
        "session_count": session_count,
        "turn_count": turn_count,
        "assertion_coverage": assertion_coverage,
        "guardrails": {
            "returncode": guardrails_returncode,
            "passed": guardrails_passed,
        },
        "suite": {
            "returncode": suite_returncode,
            "passed": suite_passed,
        },
        "overall_passed": guardrails_passed and suite_passed,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[Day10.1] Step 4: 生成总结报告...")
    print(f"[Day10.1]   总结文件: {summary_path}")
    print()

    # 5. 最终结论
    print("=" * 60)
    print("[Day10.1] 验证结论")
    print("=" * 60)

    if summary["overall_passed"]:
        print("[Day10.1] ✅ Day 10.1 验证通过")
        print()
        print("判断依据:")
        print(f"  - 数据集 JSON 合法: ✅")
        print(f"  - guardrails 通过: ✅")
        print(f"  - suite 通过: ✅")
        print(f"  - expected_regex_any 使用: {'✅' if assertion_coverage['expected_regex_any_present'] else '❌'}")
        print(f"  - expected_regex_all 使用: {'✅' if assertion_coverage['expected_regex_all_present'] else '❌'}")
        print(f"  - forbidden_contains_any 使用: {'✅' if assertion_coverage['forbidden_contains_any_present'] else '❌'}")
        print()
        print("最终结论: Day 10.1 验证闭环完成，可以进入 Day 10.2")
    else:
        print("[Day10.1] ❌ Day 10.1 验证失败")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
