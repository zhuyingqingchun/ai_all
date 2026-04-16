#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vllm_agent_eval.assertion_guardrails import (  # noqa: E402
    has_blocking_issues,
    lint_dataset,
    print_report,
)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/lint_multistep_dataset.py <dataset.json>")
        return 2

    path = Path(sys.argv[1]).resolve()
    if not path.exists():
        print(f"dataset not found: {path}")
        return 2

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    errors, warnings = lint_dataset(data)
    print_report(errors, warnings)

    if has_blocking_issues(errors):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
