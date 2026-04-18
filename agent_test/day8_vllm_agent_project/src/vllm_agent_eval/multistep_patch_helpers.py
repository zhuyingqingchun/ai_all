from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

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


def classify_clause_to_step(clause: str, weather_cities: Dict[str, Any], time_cities: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    clause = normalize_clause_text(clause)
    if not clause:
        return None

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
