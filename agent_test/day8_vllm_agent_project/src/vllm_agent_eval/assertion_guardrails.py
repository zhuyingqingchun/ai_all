from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple


TIME_LITERAL_RE = re.compile(r"\b([01]?\d|2[0-3]):[0-5]\d\b")
DATE_LITERAL_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
TEMP_LITERAL_RE = re.compile(r"\b-?\d+(?:\.\d+)?\s*(?:°C|℃)\b")
WINDSPEED_LITERAL_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:km/h|公里/小时|m/s|米/秒)\b")


def _iter_turns(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, dict):
        if "turns" in data and isinstance(data["turns"], list):
            for turn in data["turns"]:
                if isinstance(turn, dict):
                    yield turn
        elif "sessions" in data and isinstance(data["sessions"], list):
            for session in data["sessions"]:
                if not isinstance(session, dict):
                    continue
                for turn in session.get("turns", []):
                    if isinstance(turn, dict):
                        yield turn
        elif isinstance(data.get("dataset"), list):
            for item in data["dataset"]:
                if isinstance(item, dict):
                    yield item
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                if "turns" in item and isinstance(item["turns"], list):
                    for turn in item["turns"]:
                        if isinstance(turn, dict):
                            yield turn
                elif "user" in item:
                    yield item


def _find_fragile_literals(values: Iterable[str]) -> List[str]:
    fragile: List[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        if TIME_LITERAL_RE.search(value):
            fragile.append(f"time_literal:{value}")
        if DATE_LITERAL_RE.search(value):
            fragile.append(f"date_literal:{value}")
        if TEMP_LITERAL_RE.search(value):
            fragile.append(f"temp_literal:{value}")
        if WINDSPEED_LITERAL_RE.search(value):
            fragile.append(f"windspeed_literal:{value}")
    return fragile


def lint_turn(turn: Dict[str, Any], turn_index: int) -> List[str]:
    warnings: List[str] = []
    for key in ("expected_contains_all", "expected_contains_any"):
        values = turn.get(key, [])
        if not isinstance(values, list):
            warnings.append(f"turn[{turn_index}] {key} is not a list")
            continue
        fragile = _find_fragile_literals(values)
        for item in fragile:
            warnings.append(
                f"turn[{turn_index}] fragile assertion in {key}: {item} | "
                f"suggest using expected_regex_any / semantic assertion"
            )
    return warnings


def lint_dataset(data: Any) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    turns = list(_iter_turns(data))
    if not turns:
        errors.append("dataset contains no readable turns")
        return errors, warnings

    for idx, turn in enumerate(turns, start=1):
        if "user" not in turn:
            errors.append(f"turn[{idx}] missing required field: user")
        warnings.extend(lint_turn(turn, idx))

    return errors, warnings


def print_report(errors: List[str], warnings: List[str]) -> None:
    if errors:
        print("[dataset-guardrails] ERRORS")
        for err in errors:
            print(f"  - {err}")
    else:
        print("[dataset-guardrails] ERRORS: none")

    if warnings:
        print("[dataset-guardrails] WARNINGS")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("[dataset-guardrails] WARNINGS: none")


def has_blocking_issues(errors: List[str]) -> bool:
    return len(errors) > 0
