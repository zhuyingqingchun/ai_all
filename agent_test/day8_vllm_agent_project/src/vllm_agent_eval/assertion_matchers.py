from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


TIME_LITERAL_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
DATE_LITERAL_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
TEMP_LITERAL_RE = re.compile(r"\b-?\d+(?:\.\d+)?\s?(?:°C|℃)\b", re.IGNORECASE)
WIND_LITERAL_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:km/h|m/s)\b", re.IGNORECASE)


@dataclass
class AssertionResult:
    passed: bool
    failure_type: str | None = None
    failure_detail: str | None = None


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def _contains_all(answer: str, patterns: Iterable[str]) -> tuple[bool, list[str]]:
    patterns = list(patterns)
    missing = [p for p in patterns if p not in answer]
    return len(missing) == 0, missing


def _contains_any(answer: str, patterns: Iterable[str]) -> tuple[bool, list[str]]:
    patterns = list(patterns)
    hits = [p for p in patterns if p in answer]
    return len(hits) > 0, hits


def _regex_all(answer: str, regexes: Iterable[str]) -> tuple[bool, list[str]]:
    regexes = list(regexes)
    missing = [r for r in regexes if re.search(r, answer) is None]
    return len(missing) == 0, missing


def _regex_any(answer: str, regexes: Iterable[str]) -> tuple[bool, list[str]]:
    regexes = list(regexes)
    hits = [r for r in regexes if re.search(r, answer) is not None]
    return len(hits) > 0, hits


def _forbidden_contains_any(answer: str, patterns: Iterable[str]) -> tuple[bool, list[str]]:
    patterns = list(patterns)
    hits = [p for p in patterns if p in answer]
    return len(hits) == 0, hits


def find_dynamic_literals(values: Iterable[str]) -> list[str]:
    flagged: list[str] = []
    for value in values:
        if (
            TIME_LITERAL_RE.search(value)
            or DATE_LITERAL_RE.search(value)
            or TEMP_LITERAL_RE.search(value)
            or WIND_LITERAL_RE.search(value)
        ):
            flagged.append(value)
    return flagged


def evaluate_text_assertions(answer: str, turn_spec: dict[str, Any], failure_type: str = "answer_error") -> AssertionResult:
    expected_contains_all = _as_list(turn_spec.get("expected_contains_all"))
    expected_contains_any = _as_list(turn_spec.get("expected_contains_any"))
    expected_regex_all = _as_list(turn_spec.get("expected_regex_all"))
    expected_regex_any = _as_list(turn_spec.get("expected_regex_any"))
    forbidden_contains_any = _as_list(turn_spec.get("forbidden_contains_any"))

    ok, missing = _contains_all(answer, expected_contains_all)
    if not ok:
        return AssertionResult(
            passed=False,
            failure_type=failure_type,
            failure_detail=f"missing all keywords: {missing}",
        )

    if expected_contains_any:
        ok, hits = _contains_any(answer, expected_contains_any)
        if not ok:
            return AssertionResult(
                passed=False,
                failure_type=failure_type,
                failure_detail=f"missing any keywords: {expected_contains_any}",
            )

    if expected_regex_all:
        ok, missing_regex = _regex_all(answer, expected_regex_all)
        if not ok:
            return AssertionResult(
                passed=False,
                failure_type=failure_type,
                failure_detail=f"missing regex(all): {missing_regex}",
            )

    if expected_regex_any:
        ok, hits = _regex_any(answer, expected_regex_any)
        if not ok:
            return AssertionResult(
                passed=False,
                failure_type=failure_type,
                failure_detail=f"missing regex(any): {expected_regex_any}",
            )

    if forbidden_contains_any:
        ok, hits = _forbidden_contains_any(answer, forbidden_contains_any)
        if not ok:
            return AssertionResult(
                passed=False,
                failure_type=failure_type,
                failure_detail=f"forbidden keywords present: {hits}",
            )

    return AssertionResult(passed=True)
