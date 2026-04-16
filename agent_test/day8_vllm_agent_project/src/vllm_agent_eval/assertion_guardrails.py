from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

from .assertion_matchers import find_dynamic_literals


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
        flagged = find_dynamic_literals(values)
        for item in flagged:
            warnings.append(
                f"turn[{turn_index}] {key} contains dynamic literal {item} | "
                f"suggest using expected_regex_any / expected_regex_all for time/date/temp/wind values"
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
