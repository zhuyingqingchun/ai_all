from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .langgraph_poc import build_langgraph_poc


def _run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    return proc.returncode, proc.stdout, proc.stderr


def run_baseline(project_root: Path, dataset_path: Path, model: str, output_dir: Path) -> dict:
    baseline_dir = output_dir / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "env",
        f"PYTHONPATH={project_root / 'src'}",
        sys.executable,
        "-m",
        "vllm_agent_eval.cli_multistep_suite",
        "--model",
        model,
        "--main-dataset",
        str(dataset_path),
        "--out-dir",
        str(baseline_dir),
    ]
    code, out, err = _run_cmd(cmd, project_root)
    (baseline_dir / "stdout.txt").write_text(out, encoding="utf-8")
    (baseline_dir / "stderr.txt").write_text(err, encoding="utf-8")

    return {
        "returncode": code,
        "passed": code == 0,
        "stdout_file": str(baseline_dir / "stdout.txt"),
        "stderr_file": str(baseline_dir / "stderr.txt"),
        "output_dir": str(baseline_dir),
    }


def run_langgraph(project_root: Path, dataset_path: Path, output_dir: Path) -> dict:
    langgraph_dir = output_dir / "langgraph"
    langgraph_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    graph = build_langgraph_poc()

    all_results = []
    for session in data:
        session_name = session.get("session_name", "unknown")
        for idx, turn in enumerate(session.get("turns", []), start=1):
            user_input = turn["user"]
            result = graph.invoke({"user_input": user_input, "debug_trace": []})
            all_results.append(
                {
                    "session_name": session_name,
                    "turn_index": idx,
                    "user": user_input,
                    "result": result,
                }
            )

    results_path = langgraph_dir / "results.json"
    results_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "passed": True,
        "count": len(all_results),
        "results_file": str(results_path),
        "output_dir": str(langgraph_dir),
    }


def run_compare(project_root: Path, dataset_path: Path, model: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = run_baseline(project_root, dataset_path, model, output_dir)
    langgraph = run_langgraph(project_root, dataset_path, output_dir)

    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    session_count = len(data)
    turn_count = sum(len(item.get("turns", [])) for item in data)

    summary = {
        "dataset": str(dataset_path),
        "model": model,
        "session_count": session_count,
        "turn_count": turn_count,
        "baseline": baseline,
        "langgraph": langgraph,
        "overall_passed": baseline["passed"] and langgraph["passed"],
    }

    summary_path = output_dir / "day12_compare_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
