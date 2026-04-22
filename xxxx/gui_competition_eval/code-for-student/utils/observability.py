from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


class TraceLogger:
    def __init__(self, trace_path: str | Path = "output_quick/trace.jsonl", append: bool = False) -> None:
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.append = append
        # 非追加模式下，首次写入前清空文件
        if not append and self.trace_path.exists():
            self.trace_path.write_text("", encoding="utf-8")

    def log_step(
        self,
        run_id: str,
        step: int,
        page_guess: str,
        subgoal: str,
        raw_model_output: str,
        parsed_action: Optional[Dict[str, Any]],
        final_action: Dict[str, Any],
        cache_hit: bool,
        guardrail_reason: str = "",
        failure_type: str = "",
    ) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "step": step,
            "page_guess": page_guess,
            "subgoal": subgoal,
            "raw_model_output": raw_model_output,
            "parsed_action": parsed_action,
            "final_action": final_action,
            "cache_hit": cache_hit,
            "guardrail_reason": guardrail_reason,
            "failure_type": failure_type,
        }
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class MetricsCollector:
    def __init__(self, metrics_path: str | Path = "output_quick/metrics.json") -> None:
        self.metrics_path = Path(metrics_path)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self.counters: Dict[str, int] = defaultdict(int)
        self.observations: Dict[str, list[float]] = defaultdict(list)
        self._persist()

    def bump(self, key: str, value: int = 1) -> None:
        self.counters[key] += value
        self._persist()

    def observe(self, key: str, value: float) -> None:
        self.observations[key].append(float(value))
        self._persist()

    def snapshot(self) -> Dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "observations": {
                k: {
                    "count": len(v),
                    "avg": (sum(v) / len(v)) if v else 0.0,
                    "max": max(v) if v else 0.0,
                    "min": min(v) if v else 0.0,
                }
                for k, v in self.observations.items()
            },
        }

    def _persist(self) -> None:
        with self.metrics_path.open("w", encoding="utf-8") as f:
            json.dump(self.snapshot(), f, ensure_ascii=False, indent=2)
