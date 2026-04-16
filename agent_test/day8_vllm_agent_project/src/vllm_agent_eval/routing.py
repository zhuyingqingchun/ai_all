from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from .config import CITY_COORDS, CITY_TIMEZONES
from .schemas import ConversationState
from .tools import safe_eval_expr


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"未找到 JSON。模型输出为:\n{text}")
    return json.loads(match.group(0))


def looks_like_math_expression(text: str) -> bool:
    candidate = text.strip()
    if not candidate:
        return False
    for token in ["多少", "等于多少", "等于几", "等于", "=?", "＝", "="]:
        candidate = candidate.replace(token, "")
    candidate = candidate.strip()
    return bool(re.fullmatch(r"[0-9\.\+\-\*/%\(\)\s]+", candidate))


def extract_math_expression(text: str) -> str:
    expr = text
    for token in ["多少", "等于多少", "等于几", "等于", "=?", "＝", "="]:
        expr = expr.replace(token, "")
    return expr.strip()


def has_weather_intent(text: str) -> bool:
    return any(k in text for k in ["天气", "气温", "温度", "下雨", "风速", "冷不冷", "热不热"])


def has_time_intent(text: str) -> bool:
    return any(k in text for k in ["几点", "时间", "现在几点", "当地时间", "现在几时"])


def has_history_intent(text: str) -> bool:
    keywords = ["我问过", "我们刚才聊了什么", "我们聊到哪了", "还记得吗", "你刚才说过", "前面聊了什么", "总结一下我们前面", "回顾一下", "刚才那个问题"]
    return any(k in text for k in keywords)


def is_ellipsis_query(text: str) -> bool:
    patterns = [r"^那(.+?)呢$", r"^那(.+?)怎么样$", r"^那(.+?)如何$", r"^(.+?)那个问题再说一遍$"]
    return any(re.match(pattern, text.strip()) for pattern in patterns)


def find_supported_city(text: str, city_map: Dict[str, Any]) -> Optional[str]:
    for city in sorted(city_map.keys(), key=len, reverse=True):
        if city in text:
            return city
    return None


def route_by_heuristics(user_input: str, state: ConversationState) -> Optional[Dict[str, Any]]:
    if looks_like_math_expression(user_input):
        expr = extract_math_expression(user_input)
        try:
            safe_eval_expr(expr)
            return {"tool": "calculator", "args": {"expression": expr}, "route_source": "heuristic_math"}
        except Exception:
            return {"tool": "direct_answer", "args": {"mode": "invalid_math", "expression": expr}, "route_source": "heuristic_invalid_math"}
    if has_history_intent(user_input):
        return {"tool": "direct_answer", "args": {"mode": "history_answer"}, "route_source": "heuristic_history"}
    if is_ellipsis_query(user_input):
        city = find_supported_city(user_input, CITY_COORDS)
        if city and state.last_tool == "get_weather":
            return {"tool": "get_weather", "args": {"city": city}, "route_source": "heuristic_ellipsis_weather"}
        city = find_supported_city(user_input, CITY_TIMEZONES)
        if city and state.last_tool == "get_time":
            return {"tool": "get_time", "args": {"city": city}, "route_source": "heuristic_ellipsis_time"}
        return {"tool": "direct_answer", "args": {"mode": "ellipsis_fallback"}, "route_source": "heuristic_ellipsis_fallback"}
    if has_weather_intent(user_input):
        city = find_supported_city(user_input, CITY_COORDS)
        if city:
            return {"tool": "get_weather", "args": {"city": city}, "route_source": "heuristic_weather"}
        match = re.search(r"([\u4e00-\u9fa5]{2,4})天气", user_input)
        if match:
            return {"tool": "direct_answer", "args": {"mode": "unsupported_city", "city": match.group(1), "topic": "weather"}, "route_source": "heuristic_unsupported_city"}
    if has_time_intent(user_input):
        city = find_supported_city(user_input, CITY_TIMEZONES)
        if city:
            return {"tool": "get_time", "args": {"city": city}, "route_source": "heuristic_time"}
        match = re.search(r"([\u4e00-\u9fa5]{2,4})(现在几点|几点|时间)", user_input)
        if match:
            return {"tool": "direct_answer", "args": {"mode": "unsupported_city", "city": match.group(1), "topic": "time"}, "route_source": "heuristic_unsupported_city"}
    return None


def safe_parse_plan(plan_text: str, user_input: str, state: ConversationState) -> Dict[str, Any]:
    heuristic_plan = route_by_heuristics(user_input, state)
    if heuristic_plan is not None:
        return heuristic_plan
    try:
        plan = extract_json(plan_text)
        if not isinstance(plan, dict):
            raise ValueError("plan is not dict")
        if "tool" not in plan:
            raise ValueError("missing tool field")
        if "args" not in plan or not isinstance(plan["args"], dict):
            plan["args"] = {}
        if plan["tool"] not in {"get_weather", "get_time", "calculator", "direct_answer"}:
            raise ValueError("unknown tool")
        plan["route_source"] = "model_json"
        return plan
    except Exception:
        return {"tool": "direct_answer", "args": {"mode": "fallback_parse_error"}, "route_source": "fallback_parse_error"}
