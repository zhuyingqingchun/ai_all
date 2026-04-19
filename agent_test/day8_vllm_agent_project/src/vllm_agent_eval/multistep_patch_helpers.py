from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .config import NOTE_LOOKUP_TABLE, STRUCTURED_LOOKUP_TABLE

# 这份 helper 只解决 Day 8 多步实验的 planner 问题：
# 1) 子句切分
# 2) "再算..." 强制识别为 calculator step
# 3) 多步 fallback salvage
# 4) 多步里的 unsupported city 不再直接吞掉整个计划

STEP_CONNECTORS = ["先", "再", "然后", "接着", "随后", "最后"]
FINAL_ONLY_HINTS = ["总结", "概括", "用两行", "两行", "说明哪一步失败", "告诉我两个结果", "告诉我结果", "输出结果"]
MATH_HINTS = ["算", "计算", "sqrt", "log", "sin", "cos", "tan", "乘", "加", "减", "除"]
TIME_HINTS = ["几点", "时间", "现在几点", "当地时间", "几时"]
WEATHER_HINTS = ["天气", "气温", "温度", "风速", "下雨", "冷不冷", "热不热"]
UNIT_HINTS = ["转成", "换算", "换成", "转换成", "转换为", "转为"]
LOOKUP_HINTS = ["配置值", "配置", "参数", "上下文长度", "context"]
DATETIME_CALC_HINTS = ["小时后", "小时前", "天后", "天前", "几号", "日期"]
SEARCH_HINTS = ["搜索", "检索", "搜一下", "搜一搜", "搜索一下", "检索一下", "在笔记里找", "查找笔记"]
NOTE_HINTS = ["笔记", "note", "记录"]
UNIT_PATTERN = re.compile(
    r"(?P<value>-?\d+(?:\.\d+)?)\s*(?P<from_unit>km/h|m/s|kg|g|cm|m|℃|°C|°F|C|F)\s*(?:转成|换算成|换成|转换成|转换为|转为)\s*(?P<to_unit>km/h|m/s|kg|g|cm|m|℃|°C|°F|C|F)",
    flags=re.IGNORECASE,
)
FINAL_SUFFIX_RE = re.compile(
    r"(?:[，,、 ]*(?:并|并且)?[，,、 ]*)?(说明哪一步失败.*|告诉我两个结果.*|告诉我结果.*|输出结果.*|用两行.*|两行.*|总结.*|概括.*)$"
)


def normalize_clause_text(text: str) -> str:
    text = text.strip().strip("，。；; ")
    text = re.sub(r"^(请|帮我|麻烦|先帮我|请先)", "", text)
    text = re.sub(r"^(先|再|然后|接着|随后|最后)", "", text)
    return text.strip("，。；; ")


def split_math_clause_and_final_hint(text: str) -> Tuple[str, str]:
    """把形如“算一下 12*12，并说明哪一步失败了”拆成 step 与 final hint。"""
    normalized = normalize_clause_text(text)
    if not normalized:
        return "", ""

    match = FINAL_SUFFIX_RE.search(normalized)
    if not match:
        return normalized, ""

    step_part = normalized[: match.start()].strip(" ，。；;、")
    final_part = match.group(1).strip(" ，。；;、")
    if step_part and looks_like_math_clause(step_part):
        return step_part, final_part
    return normalized, ""


def split_multistep_clauses(user_input: str) -> Tuple[List[str], str]:
    """
    把多步任务拆成 step clauses 和 final_instruction。
    关键修复：
    - "再算一下 ..." 一定保留为 step，不允许误归入 final_instruction
    - "最后告诉我两个结果 / 用两行总结 / 并说明哪一步失败了" 这类归入 final_instruction
    - 数学 step 后缀中的 final-only hint 要和表达式本身拆开
    """
    text = user_input.strip()
    if not text:
        return [], ""

    normalized = text.replace("；", "，").replace(";", "，")
    pieces = re.split(r"(?=(?:先|再|然后|接着|随后|最后))", normalized)
    pieces = [p.strip("， ") for p in pieces if p.strip("， ")]

    step_clauses: List[str] = []
    final_parts: List[str] = []

    for raw_piece in pieces:
        piece = raw_piece.strip()
        normalized_piece = normalize_clause_text(piece)
        if not normalized_piece:
            continue

        math_step_part, math_final_part = split_math_clause_and_final_hint(normalized_piece)
        if looks_like_math_clause(math_step_part):
            step_clauses.append(math_step_part)
            if math_final_part:
                final_parts.append(math_final_part)
            continue

        if parse_unit_convert_clause(normalized_piece) is not None:
            step_clauses.append(normalized_piece)
            continue

        if parse_note_lookup_clause(normalized_piece) is not None:
            step_clauses.append(normalized_piece)
            continue

        if parse_text_search_clause(normalized_piece) is not None:
            step_clauses.append(normalized_piece)
            continue

        if parse_structured_lookup_clause(normalized_piece) is not None:
            step_clauses.append(normalized_piece)
            continue

        if has_any(normalized_piece, DATETIME_CALC_HINTS) and re.search(
            r"(东京|北京|上海|深圳|广州|纽约|洛杉矶|旧金山)", normalized_piece
        ):
            step_clauses.append(normalized_piece)
            continue

        if has_any(normalized_piece, WEATHER_HINTS) or has_any(normalized_piece, TIME_HINTS):
            step_clauses.append(normalized_piece)
            continue

        if is_final_only_clause(normalized_piece):
            final_parts.append(normalized_piece)
            continue

        if re.search(r"(东京|北京|上海|深圳|广州|纽约|洛杉矶|旧金山).*(时间|几点)", normalized_piece):
            step_clauses.append(normalized_piece)
            continue
        if re.search(r"(东京|北京|上海|深圳|广州|纽约|洛杉矶|旧金山).*(天气|气温|温度)", normalized_piece):
            step_clauses.append(normalized_piece)
            continue

        final_parts.append(normalized_piece)

    return step_clauses, "；".join(final_parts).strip("； ")


def has_any(text: str, hints: List[str]) -> bool:
    return any(h in text for h in hints)


def is_final_only_clause(text: str) -> bool:
    if looks_like_math_clause(text):
        return False
    if has_any(text, WEATHER_HINTS) or has_any(text, TIME_HINTS):
        return False
    return has_any(text, FINAL_ONLY_HINTS)


def looks_like_math_expression(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    t = t.replace("多少", "").replace("等于多少", "").replace("等于几", "")
    t = t.replace("等于", "").replace("=?", "").replace("＝", "=").replace("=", "")
    t = t.strip()
    return bool(re.fullmatch(r"[0-9\.\+\-\*/%\(\)\s]+", t))


def extract_math_expression(text: str) -> str:
    expr = text
    expr = expr.replace("算一下", "").replace("算", "").replace("计算", "")
    expr = expr.replace("多少", "").replace("等于多少", "").replace("等于几", "")
    expr = expr.replace("等于", "").replace("=?", "").replace("＝", "=").replace("=", "")
    expr = FINAL_SUFFIX_RE.sub("", expr)
    expr = expr.strip(" ，。；;、`")
    return expr


def looks_like_math_clause(text: str) -> bool:
    if looks_like_math_expression(text):
        return True
    if any(h in text for h in MATH_HINTS):
        return True
    if re.search(r"[0-9].*[\+\-\*/%].*[0-9]", text):
        return True
    return False


def extract_city_from_clause(clause: str, supported_cities: Dict[str, Any]) -> Optional[str]:
    for city in sorted(supported_cities.keys(), key=len, reverse=True):
        if city in clause:
            return city

    m = re.search(r"(?:查|告诉我|问一下|看一下)?([\u4e00-\u9fa5]{2,4})(?:天气|气温|温度|风速|时间|几点)", clause)
    if m:
        return m.group(1)

    m = re.search(r"([\u4e00-\u9fa5]{2,4})(?:现在几点|几点|时间)", clause)
    if m:
        return m.group(1)

    return None


def parse_unit_convert_clause(clause: str) -> Optional[Dict[str, Any]]:
    match = UNIT_PATTERN.search(clause)
    if not match:
        return None
    return {
        "tool": "unit_convert",
        "args": {
            "value": float(match.group("value")),
            "from_unit": match.group("from_unit"),
            "to_unit": match.group("to_unit"),
        },
        "route_source": "heuristic_multistep_clause_unit_convert",
    }


def parse_text_search_clause(clause: str) -> Optional[Dict[str, Any]]:
    if not has_any(clause, SEARCH_HINTS):
        return None
    for prefix in ["搜索一下", "搜索", "检索一下", "检索", "搜一下", "搜一搜", "在笔记里找", "查找笔记"]:
        if prefix in clause:
            query = clause.split(prefix, 1)[1].strip(" ：:，。；;、")
            query = FINAL_SUFFIX_RE.sub("", query).strip(" ，。；;、")
            if query:
                return {
                    "tool": "text_search",
                    "args": {"query": query, "top_k": 3},
                    "route_source": "heuristic_multistep_clause_text_search",
                }
    return None


def parse_note_lookup_clause(clause: str) -> Optional[Dict[str, Any]]:
    if not (has_any(clause, NOTE_HINTS) or "_note" in clause):
        return None
    for key in sorted(NOTE_LOOKUP_TABLE.keys(), key=len, reverse=True):
        if key in clause:
            return {
                "tool": "note_lookup",
                "args": {"note_id": key},
                "route_source": "heuristic_multistep_clause_note_lookup",
            }
    token_match = re.search(r"([A-Za-z0-9._-]+_note)", clause)
    if token_match:
        return {
            "tool": "note_lookup",
            "args": {"note_id": token_match.group(1)},
            "route_source": "heuristic_multistep_clause_note_lookup",
        }
    return None


def parse_date_time_calc_clause(clause: str, time_cities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    city = extract_city_from_clause(clause, time_cities)
    match = re.search(r"(\d+)\s*(小时|天)\s*(后|前)", clause)
    if not city or not match:
        return None
    if not has_any(clause, DATETIME_CALC_HINTS):
        return None
    delta = int(match.group(1))
    if match.group(3) == "前":
        delta = -delta
    unit = "hour" if match.group(2) == "小时" else "day"
    return {
        "tool": "date_time_calc",
        "args": {"city": city, "delta": delta, "unit": unit},
        "route_source": "heuristic_multistep_clause_datetime_calc",
    }


def parse_structured_lookup_clause(clause: str) -> Optional[Dict[str, Any]]:
    if not has_any(clause, LOOKUP_HINTS):
        return None
    for key in sorted(STRUCTURED_LOOKUP_TABLE.keys(), key=len, reverse=True):
        if key in clause:
            return {
                "tool": "structured_lookup",
                "args": {"key": key},
                "route_source": "heuristic_multistep_clause_structured_lookup",
            }
    token_match = re.search(r"([A-Za-z0-9._-]+)", clause)
    if token_match:
        return {
            "tool": "structured_lookup",
            "args": {"key": token_match.group(1)},
            "route_source": "heuristic_multistep_clause_structured_lookup",
        }
    return None


def classify_clause_to_step(clause: str, weather_cities: Dict[str, Any], time_cities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    clause = normalize_clause_text(clause)
    if not clause:
        return None

    note_step = parse_note_lookup_clause(clause)
    if note_step is not None:
        return note_step

    search_step = parse_text_search_clause(clause)
    if search_step is not None:
        return search_step

    unit_step = parse_unit_convert_clause(clause)
    if unit_step is not None:
        return unit_step

    lookup_step = parse_structured_lookup_clause(clause)
    if lookup_step is not None:
        return lookup_step

    datetime_step = parse_date_time_calc_clause(clause, time_cities)
    if datetime_step is not None:
        return datetime_step

    if looks_like_math_clause(clause):
        return {
            "tool": "calculator",
            "args": {"expression": extract_math_expression(clause)},
            "route_source": "heuristic_multistep_clause_math",
        }

    if has_any(clause, WEATHER_HINTS):
        city = extract_city_from_clause(clause, weather_cities)
        if city:
            return {
                "tool": "get_weather",
                "args": {"city": city},
                "route_source": "heuristic_multistep_clause_weather",
            }

    if has_any(clause, TIME_HINTS):
        city = extract_city_from_clause(clause, time_cities)
        if city:
            return {
                "tool": "get_time",
                "args": {"city": city},
                "route_source": "heuristic_multistep_clause_time",
            }

    return None


def build_salvage_multistep_plan(user_input: str, weather_cities: Dict[str, Any], time_cities: Dict[str, Any]) -> Dict[str, Any]:
    clauses, final_instruction = split_multistep_clauses(user_input)

    steps: List[Dict[str, Any]] = []
    for clause in clauses:
        step = classify_clause_to_step(clause, weather_cities, time_cities)
        if step is not None:
            steps.append(step)

    if steps:
        return {
            "steps": steps,
            "final_instruction": final_instruction or "请按顺序给出每一步结果，并在最后简要总结。",
            "route_source": "fallback_multistep_salvage",
        }

    return {
        "steps": [],
        "final_instruction": final_instruction or "请直接回答用户。",
        "route_source": "fallback_single_direct_answer",
    }


def expected_tools_from_steps(steps: List[Dict[str, Any]]) -> List[str]:
    return [s.get("tool", "") for s in steps if s.get("tool")]
