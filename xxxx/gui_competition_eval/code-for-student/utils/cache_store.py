from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Optional


class TTLCache:
    def __init__(self, ttl_seconds: int = 300, namespace: str = "default", base_dir: str | Path = "output_quick/cache") -> None:
        self.ttl_seconds = ttl_seconds
        self.namespace = namespace
        self.base_dir = Path(base_dir) / namespace
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if time.time() - payload.get("ts", 0) > self.ttl_seconds:
            return None
        return payload.get("value")

    def set(self, key: str, value: Any) -> None:
        path = self._path_for(key)
        payload = {
            "ts": time.time(),
            "value": value,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def _path_for(self, key: str) -> Path:
        safe = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.base_dir / f"{safe}.json"
