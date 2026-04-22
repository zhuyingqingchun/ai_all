from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class CheckpointStore:
    def __init__(self, base_dir: str | Path = "output_quick/checkpoints") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, step: int, payload: Dict[str, Any]) -> None:
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"step_{step:03d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
