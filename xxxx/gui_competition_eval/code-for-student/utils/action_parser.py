from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


class ActionParseError(ValueError):
    pass


@dataclass
class ParsedAction:
    action: str
    parameters: Dict[str, Any]


class ActionParser:
    ACTION_RE = re.compile(r"\b(OPEN|CLICK|TYPE|SCROLL|COMPLETE)\b", re.IGNORECASE)
    CLICK_ID_RE = re.compile("CLICK\s*:\s*\[\s*['\"]?(C\d+)['\"]?\s*\]", re.IGNORECASE)
    CLICK_RE = re.compile(r"CLICK\s*:\s*(\[[^\n]+\])", re.IGNORECASE)
    TYPE_RE = re.compile(r"TYPE\s*:\s*(\[[^\n]+\])", re.IGNORECASE)
    OPEN_RE = re.compile(r"OPEN\s*:\s*(\[[^\n]+\])", re.IGNORECASE)
    SCROLL_RE = re.compile(r"SCROLL\s*:\s*(\[[^\n]+\])", re.IGNORECASE)

    def parse(self, text: str) -> ParsedAction:
        text = (text or "").strip()
        if not text:
            raise ActionParseError("empty output")
        parsed = self._parse_explicit(text) or self._parse_linewise(text)
        if parsed is None:
            raise ActionParseError(f"cannot parse action from: {text!r}")
        return ParsedAction(action=parsed[0], parameters=parsed[1])

    def _parse_explicit(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if re.search(r"\bCOMPLETE\b", text, re.IGNORECASE) and not re.search(r"\b(CLICK|TYPE|SCROLL|OPEN)\b", text, re.IGNORECASE):
            return "COMPLETE", {}

        m = self.CLICK_ID_RE.search(text)
        if m:
            return "CLICK", {"candidate_id": m.group(1).upper()}

        m = self.CLICK_RE.search(text)
        if m:
            point = self._extract_click_target(m.group(1))
            if point is not None:
                return "CLICK", {"point": point}

        m = self.TYPE_RE.search(text)
        if m:
            content = self._extract_single_string(m.group(1))
            return "TYPE", {"text": content}

        m = self.OPEN_RE.search(text)
        if m:
            app_name = self._extract_single_string(m.group(1))
            return "OPEN", {"app_name": app_name}

        m = self.SCROLL_RE.search(text)
        if m:
            nums = self._extract_numbers(m.group(1))
            if len(nums) >= 4:
                return "SCROLL", {
                    "start_point": self._ensure_point([nums[0], nums[1]]),
                    "end_point": self._ensure_point([nums[2], nums[3]]),
                }
        return None

    def _parse_linewise(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            m = self.ACTION_RE.search(line)
            if not m:
                continue
            action = m.group(1).upper()
            if action == "COMPLETE":
                return "COMPLETE", {}
            if action == "CLICK":
                m2 = self.CLICK_ID_RE.search(line)
                if m2:
                    return "CLICK", {"candidate_id": m2.group(1).upper()}
                point = self._extract_click_target(line)
                if point is not None:
                    return "CLICK", {"point": point}
            if action == "TYPE":
                quoted = self._extract_first_quoted(line)
                if quoted is not None:
                    return "TYPE", {"text": quoted}
            if action == "OPEN":
                quoted = self._extract_first_quoted(line)
                if quoted is not None:
                    return "OPEN", {"app_name": quoted}
            if action == "SCROLL":
                nums = self._extract_numbers(line)
                if len(nums) >= 4:
                    return "SCROLL", {
                        "start_point": self._ensure_point([nums[0], nums[1]]),
                        "end_point": self._ensure_point([nums[2], nums[3]]),
                    }
        return None

    def _extract_click_target(self, text: str) -> Optional[List[int]]:
        nums = self._extract_numbers(text)
        if len(nums) >= 4:
            x1, y1, x2, y2 = map(float, nums[:4])
            return self._ensure_point([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        if len(nums) >= 2:
            return self._ensure_point([nums[0], nums[1]])
        return None

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", text)

    @staticmethod
    def _extract_first_quoted(text: str) -> Optional[str]:
        m = re.search(r"['\"]([^'\"]+)['\"]", text)
        return m.group(1) if m else None

    def _extract_single_string(self, list_like: str) -> str:
        try:
            value = ast.literal_eval(list_like)
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str):
                return value
        except Exception:
            pass
        quoted = self._extract_first_quoted(list_like)
        return quoted or ""

    @staticmethod
    def _ensure_point(point: List[Any]) -> List[int]:
        out: List[int] = []
        for v in point[:2]:
            try:
                iv = int(round(float(v)))
            except Exception:
                iv = 500
            out.append(max(0, min(1000, iv)))
        while len(out) < 2:
            out.append(500)
        return out
