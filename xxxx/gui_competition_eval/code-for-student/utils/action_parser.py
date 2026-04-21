from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

VALID_ACTIONS = {"CLICK", "TYPE", "SCROLL", "OPEN", "COMPLETE"}


class ActionParseError(ValueError):
    pass


@dataclass
class ParsedAction:
    action: str
    parameters: Dict[str, Any]
    raw_output: str = ""


class ActionParser:
    """鲁棒动作解析器。

    目标：兼容 GUI Agent 常见的多种输出风格，并统一转成标准格式：
    - CLICK  -> {"point": [x, y]}
    - TYPE   -> {"text": "..."}
    - SCROLL -> {"start_point": [x1, y1], "end_point": [x2, y2]}
    - OPEN   -> {"app_name": "..."}
    - COMPLETE -> {}
    """

    ACTION_RE = re.compile(r"\b(CLICK|TYPE|SCROLL|OPEN|COMPLETE)\b", re.IGNORECASE)
    JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
    JSON_OBJECT_RE = re.compile(r"(\{[\s\S]*\})")
    POINT_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")
    OPEN_RE = re.compile(r"OPEN\s*[:=]\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
    TYPE_RE = re.compile(r"TYPE\s*[:=]\s*\[(.*?)\]", re.IGNORECASE | re.DOTALL)
    CLICK_RE = re.compile(r"CLICK\s*[:=]\s*(\[[\s\S]*?\])", re.IGNORECASE)
    SCROLL_RE = re.compile(r"SCROLL\s*[:=]\s*(\[[\s\S]*?\])", re.IGNORECASE)

    def parse(self, raw_output: str) -> ParsedAction:
        text = (raw_output or "").strip()
        if not text:
            raise ActionParseError("empty model output")

        parsed = self._parse_json_style(text)
        if parsed is not None:
            return ParsedAction(parsed[0], parsed[1], raw_output=text)

        parsed = self._parse_explicit_syntax(text)
        if parsed is not None:
            return ParsedAction(parsed[0], parsed[1], raw_output=text)

        parsed = self._parse_linewise(text)
        if parsed is not None:
            return ParsedAction(parsed[0], parsed[1], raw_output=text)

        raise ActionParseError(f"unable to parse action from output: {text[:240]}")

    def normalize(self, action: str, parameters: Dict[str, Any]) -> ParsedAction:
        action = (action or "").strip().upper()
        if action not in VALID_ACTIONS:
            raise ActionParseError(f"invalid action: {action}")

        if action == "CLICK":
            point = self._ensure_point(parameters.get("point") or parameters.get("coord") or parameters.get("coords"))
            return ParsedAction(action, {"point": point})
        if action == "SCROLL":
            start = self._ensure_point(parameters.get("start_point") or parameters.get("start") or parameters.get("from"))
            end = self._ensure_point(parameters.get("end_point") or parameters.get("end") or parameters.get("to"))
            return ParsedAction(action, {"start_point": start, "end_point": end})
        if action == "TYPE":
            text = parameters.get("text")
            if text is None:
                text = parameters.get("content", "")
            return ParsedAction(action, {"text": str(text)})
        if action == "OPEN":
            app_name = parameters.get("app_name")
            if app_name is None:
                app_name = parameters.get("app", "")
            return ParsedAction(action, {"app_name": str(app_name).strip()})
        return ParsedAction("COMPLETE", {})

    def _parse_json_style(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        json_candidates: List[str] = []
        json_candidates.extend(self.JSON_BLOCK_RE.findall(text))
        if text.strip().startswith("{") and text.strip().endswith("}"):
            json_candidates.append(text.strip())
        else:
            obj_match = self.JSON_OBJECT_RE.search(text)
            if obj_match:
                json_candidates.append(obj_match.group(1))

        for candidate in json_candidates:
            candidate = candidate.strip()
            for loader in (json.loads, self._safe_literal_json):
                try:
                    obj = loader(candidate)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                action = obj.get("action") or obj.get("Action") or obj.get("act")
                params = obj.get("parameters") or obj.get("params") or obj.get("arguments") or {}
                if action:
                    normalized = self.normalize(str(action), params if isinstance(params, dict) else {})
                    return normalized.action, normalized.parameters
        return None

    def _parse_explicit_syntax(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        if re.search(r"\bCOMPLETE\b", text, re.IGNORECASE):
            if not re.search(r"\b(CLICK|TYPE|SCROLL|OPEN)\b", text, re.IGNORECASE):
                return "COMPLETE", {}

        click_match = self.CLICK_RE.search(text)
        if click_match:
            points = self.POINT_RE.findall(click_match.group(1))
            if points:
                point = self._ensure_point([points[0][0], points[0][1]])
                return "CLICK", {"point": point}

        scroll_match = self.SCROLL_RE.search(text)
        if scroll_match:
            points = self.POINT_RE.findall(scroll_match.group(1))
            if len(points) >= 2:
                start = self._ensure_point([points[0][0], points[0][1]])
                end = self._ensure_point([points[1][0], points[1][1]])
                return "SCROLL", {"start_point": start, "end_point": end}

        type_match = self.TYPE_RE.search(text)
        if type_match:
            content = self._extract_single_string(type_match.group(1))
            return "TYPE", {"text": content}

        open_match = self.OPEN_RE.search(text)
        if open_match:
            app_name = self._extract_single_string(open_match.group(1))
            return "OPEN", {"app_name": app_name}

        return None

    def _parse_linewise(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            action_match = self.ACTION_RE.search(line)
            if not action_match:
                continue
            action = action_match.group(1).upper()
            if action == "COMPLETE":
                return "COMPLETE", {}
            if action == "CLICK":
                points = self.POINT_RE.findall(line)
                if points:
                    return "CLICK", {"point": self._ensure_point([points[0][0], points[0][1]])}
            elif action == "SCROLL":
                points = self.POINT_RE.findall(line)
                if len(points) >= 2:
                    return "SCROLL", {
                        "start_point": self._ensure_point([points[0][0], points[0][1]]),
                        "end_point": self._ensure_point([points[1][0], points[1][1]]),
                    }
            elif action == "TYPE":
                quoted = self._extract_first_quoted(line)
                if quoted is not None:
                    return "TYPE", {"text": quoted}
            elif action == "OPEN":
                quoted = self._extract_first_quoted(line)
                if quoted is not None:
                    return "OPEN", {"app_name": quoted}
        return None

    def _ensure_point(self, value: Any) -> List[int]:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                pass
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ActionParseError(f"invalid point: {value}")
        x = self._clamp_coord(value[0])
        y = self._clamp_coord(value[1])
        return [x, y]

    @staticmethod
    def _clamp_coord(value: Any) -> int:
        try:
            v = int(round(float(value)))
        except Exception as exc:
            raise ActionParseError(f"invalid coord: {value}") from exc
        return max(0, min(1000, v))

    @staticmethod
    def _safe_literal_json(candidate: str) -> Any:
        return ast.literal_eval(candidate)

    @staticmethod
    def _extract_single_string(text: str) -> str:
        quoted = ActionParser._extract_first_quoted(text)
        if quoted is not None:
            return quoted
        value = text.strip().strip("[](){} ")
        value = value.strip("'\" ")
        return value

    @staticmethod
    def _extract_first_quoted(text: str) -> Optional[str]:
        match = re.search(r"['\"](.*?)['\"]", text, re.DOTALL)
        if match:
            return match.group(1)
        return None
