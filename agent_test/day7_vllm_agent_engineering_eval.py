#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import operator
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_RESULTS_JSON = "day7_vllm_agent_eval_results.json"
DEFAULT_TRACE_JSONL = "day7_vllm_agent_eval_traces.jsonl"

MAX_RECENT_MESSAGES = 8
KEEP_RECENT_MESSAGES = 4

CITY_COORDS = {
    "上海": (31.23, 121.47),
    "北京": (39.90, 116.40),
    "深圳": (22.54, 114.06),
    "广州": (23.13, 113.27),
    "东京": (35.68, 139.69),
    "纽约": (40.71, -74.00),
    "洛杉矶": (34.05, -118.24),
    "旧金山": (37.77, -122.42),
}

CITY_TIMEZONES = {
    "上海": "Asia/Shanghai",
    "北京": "Asia/Shanghai",
    "深圳": "Asia/Shanghai",
    "广州": "Asia/Shanghai",
    "东京": "Asia/Tokyo",
    "纽约": "America/New_York",
    "洛杉矶": "America/Los_Angeles",
    "旧金山": "America/Los_Angeles",
}

TOOL_SYSTEM_PROMPT = """你是一个严格的工具路由助手。

你的职责只有一个：判断当前问题应该使用哪个工具。
你必须只输出一个 JSON 对象，不能输出任何解释、前缀、后缀、markdown。

可选工具只有 4 个：

1. get_weather
用途：查询城市天气
格式：
{"tool":"get_weather","args":{"city":"北京"}}

2. get_time
用途：查询城市当前时间
格式：
{"tool":"get_time","args":{"city":"东京"}}

3. calculator
用途：计算数学表达式
格式：
{"tool":"calculator","args":{"expression":"(23+7)*3"}}

4. direct_answer
用途：不需要外部工具、但需要结合对话历史或常识直接回答
格式：
{"tool":"direct_answer","args":{}}

严格规则：
- 包含“天气 / 温度 / 气温 / 下雨 / 风速”等 → 必须用 get_weather
- 包含“几点 / 时间 / 现在几点 / 当地时间”等 → 必须用 get_time
- 只要用户输入主要是在做数学计算，哪怕只是 45454+87987987，也必须用 calculator
- 包含“我问过什么 / 我们刚才聊了什么 / 还记得吗 / 你刚才说过 / 总结前面”等依赖对话历史的问题 → 必须用 direct_answer
- 常识问答、闲聊 → direct_answer
- direct_answer 时不要生成答案内容，只输出 {"tool":"direct_answer","args":{}}
- 只能输出 JSON 对象
"""

FINAL_SYSTEM_PROMPT = """你是一个简洁、自然、可靠的中文助手。

规则：
- 请结合当前问题、对话历史、历史摘要、路由提示、工具返回来回答
- 如果工具返回 ok=true，只根据工具结果回答，不要编造
- 如果工具返回 ok=false，直接告诉用户失败原因
- 如果用户问“我刚才问过什么”“我们聊到哪了”“还记得吗”，请基于对话历史和历史摘要准确回答
- 如果用户输入是纯数学表达式，也可以直接给出自然中文答案
- 用自然、简洁、纯中文回答
"""

SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要助手。

请把更早的对话内容压缩成一段中文摘要，要求：
1. 只保留后续回答真正有用的信息
2. 保留用户问过的主题、已经查过的城市/时间/天气、做过的计算、模型给过的关键答复
3. 保留未解决的问题或偏好（如果有）
4. 不要逐字复述对话
5. 输出 80~180 字中文摘要
"""

DATASET = [
    {
        "session_name": "basic_tools_and_memory",
        "turns": [
            {
                "user": "北京天气如何",
                "expected_tool": "get_weather",
                "expected_contains_any": ["北京", "温度", "气温", "℃"],
                "tags": ["weather", "basic"],
            },
            {
                "user": "纽约现在几点",
                "expected_tool": "get_time",
                "expected_contains_any": ["纽约", "时间", "点", "星期"],
                "tags": ["time", "basic"],
            },
            {
                "user": "45454+87987987",
                "expected_tool": "calculator",
                "expected_contains_any": ["88033441", "计算结果"],
                "tags": ["calculator", "basic"],
            },
            {
                "user": "我问过你哪些问题",
                "expected_tool": "direct_answer",
                "expected_contains_all": ["北京", "纽约"],
                "tags": ["memory", "history"],
            },
            {
                "user": "还记得我刚才问过纽约的问题吗",
                "expected_tool": "direct_answer",
                "expected_contains_all": ["纽约"],
                "expected_contains_any": ["时间", "几点"],
                "tags": ["memory", "history", "reference"],
            },
        ],
    },
    {
        "session_name": "synonym_and_ellipsis",
        "turns": [
            {
                "user": "北京冷不冷",
                "acceptable_tools": ["get_weather", "direct_answer"],
                "expected_contains_any": ["北京", "温度", "气温", "℃", "冷"],
                "tags": ["weather", "synonym"],
            },
            {
                "user": "那纽约呢",
                "acceptable_tools": ["get_weather", "get_time", "direct_answer"],
                "expected_contains_all": ["纽约"],
                "tags": ["memory", "ellipsis", "reference"],
            },
            {
                "user": "23*7",
                "expected_tool": "calculator",
                "expected_contains_any": ["161", "计算结果"],
                "tags": ["calculator", "synonym"],
            },
            {
                "user": "前面我们查过哪些城市",
                "expected_tool": "direct_answer",
                "expected_contains_all": ["北京", "纽约"],
                "tags": ["memory", "history"],
            },
        ],
    },
    {
        "session_name": "tool_errors_and_boundaries",
        "turns": [
            {
                "user": "南京天气如何",
                "acceptable_tools": ["direct_answer", "get_weather"],
                "expected_contains_any": ["南京", "暂不支持", "抱歉"],
                "tags": ["weather", "unsupported_city", "boundary"],
            },
            {
                "user": "火星天气怎么样",
                "expected_tool": "direct_answer",
                "expected_contains_any": ["无法", "不支持", "抱歉"],
                "tags": ["weather", "boundary"],
            },
            {
                "user": "2/**3",
                "acceptable_tools": ["direct_answer", "calculator"],
                "expected_contains_any": ["抱歉", "换一种说法", "无法", "格式有误", "计算失败"],
                "tags": ["calculator", "boundary"],
            },
        ],
    },
    {
        "session_name": "summary_memory_rollover",
        "turns": [
            {
                "user": "上海天气如何",
                "expected_tool": "get_weather",
                "expected_contains_any": ["上海", "温度", "气温", "℃"],
                "tags": ["weather", "summary"],
            },
            {
                "user": "东京现在几点",
                "expected_tool": "get_time",
                "expected_contains_any": ["东京", "时间", "点", "星期"],
                "tags": ["time", "summary"],
            },
            {
                "user": "1+1",
                "expected_tool": "calculator",
                "expected_contains_any": ["2", "计算结果"],
                "tags": ["calculator", "summary"],
            },
            {
                "user": "深圳天气如何",
                "expected_tool": "get_weather",
                "expected_contains_any": ["深圳", "温度", "气温", "℃"],
                "tags": ["weather", "summary"],
            },
            {
                "user": "旧金山现在几点",
                "expected_tool": "get_time",
                "expected_contains_any": ["旧金山", "时间", "点", "星期"],
                "tags": ["time", "summary"],
            },
            {
                "user": "总结一下我们前面聊过哪些城市和任务",
                "expected_tool": "direct_answer",
                "expected_contains_all": ["上海", "东京", "深圳"],
                "expected_contains_any": ["天气", "时间", "计算"],
                "tags": ["memory", "summary", "history"],
            },
        ],
    },
]


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
    t = text.strip()
    if not t:
        return False
    t = t.replace("多少", "").replace("等于多少", "").replace("等于几", "").replace("等于", "")
    t = t.replace("=?", "").replace("＝", "=").replace("=", "").strip()
    return bool(re.fullmatch(r"[0-9\.\+\-\*/%\(\)\s]+", t))


def extract_math_expression(text: str) -> str:
    expr = text
    for s in ["多少", "等于多少", "等于几", "等于", "=?", "＝", "="]:
        expr = expr.replace(s, "")
    return expr.strip()


def has_weather_intent(text: str) -> bool:
    return any(k in text for k in ["天气", "气温", "温度", "下雨", "风速", "冷不冷", "热不热"])


def has_time_intent(text: str) -> bool:
    return any(k in text for k in ["几点", "时间", "现在几点", "当地时间", "现在几时"])


def has_history_intent(text: str) -> bool:
    kws = ["我问过", "我们刚才聊了什么", "我们聊到哪了", "还记得吗", "你刚才说过", "前面聊了什么", "总结一下我们前面", "回顾一下", "刚才那个问题"]
    return any(k in text for k in kws)


def is_ellipsis_query(text: str) -> bool:
    patterns = [r"^那(.+?)呢$", r"^那(.+?)怎么样$", r"^那(.+?)如何$", r"^(.+?)那个问题再说一遍$"]
    text = text.strip()
    return any(re.match(p, text) for p in patterns)


def find_supported_city(text: str, city_map: Dict[str, Any]) -> Optional[str]:
    for city in sorted(city_map.keys(), key=len, reverse=True):
        if city in text:
            return city
    return None


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
}
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


def safe_eval_expr(expr: str):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("只允许数字常量")
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_BIN_OPS:
                raise ValueError(f"不支持的运算符：{op_type.__name__}")
            return _ALLOWED_BIN_OPS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY_OPS:
                raise ValueError(f"不支持的一元运算符：{op_type.__name__}")
            return _ALLOWED_UNARY_OPS[op_type](_eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("只允许简单函数名调用")
            func_name = node.func.id
            if func_name not in _ALLOWED_FUNCS:
                raise ValueError(f"不支持的函数：{func_name}")
            args = [_eval(arg) for arg in node.args]
            return _ALLOWED_FUNCS[func_name](*args)
        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"未知变量：{node.id}")
        raise ValueError(f"不支持的表达式节点：{type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def route_by_heuristics(user_input: str, state: "ConversationState") -> Optional[Dict[str, Any]]:
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
        m = re.search(r"([\u4e00-\u9fa5]{2,4})天气", user_input)
        if m:
            return {"tool": "direct_answer", "args": {"mode": "unsupported_city", "city": m.group(1), "topic": "weather"}, "route_source": "heuristic_unsupported_city"}

    if has_time_intent(user_input):
        city = find_supported_city(user_input, CITY_TIMEZONES)
        if city:
            return {"tool": "get_time", "args": {"city": city}, "route_source": "heuristic_time"}
        m = re.search(r"([\u4e00-\u9fa5]{2,4})(现在几点|几点|时间)", user_input)
        if m:
            return {"tool": "direct_answer", "args": {"mode": "unsupported_city", "city": m.group(1), "topic": "time"}, "route_source": "heuristic_unsupported_city"}

    return None


def safe_parse_plan(plan_text: str, user_input: str, state: "ConversationState") -> Dict[str, Any]:
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


@dataclass
class ConversationState:
    last_tool: Optional[str] = None
    last_city: Optional[str] = None
    last_topic: Optional[str] = None
    seen_cities: List[str] = field(default_factory=list)
    asked_topics: List[str] = field(default_factory=list)

    def update(self, plan: dict):
        tool = plan.get("tool")
        args = plan.get("args", {})
        self.last_tool = tool
        if tool == "get_weather":
            self.last_topic = "weather"
            city = args.get("city")
            if city:
                self.last_city = city
                if city not in self.seen_cities:
                    self.seen_cities.append(city)
        elif tool == "get_time":
            self.last_topic = "time"
            city = args.get("city")
            if city:
                self.last_city = city
                if city not in self.seen_cities:
                    self.seen_cities.append(city)
        elif tool == "calculator":
            self.last_topic = "calculator"
        elif tool == "direct_answer":
            self.last_topic = "chat"
        if self.last_topic and self.last_topic not in self.asked_topics:
            self.asked_topics.append(self.last_topic)


@dataclass
class ConversationMemory:
    summary: str = ""
    recent_messages: List[Dict[str, str]] = field(default_factory=list)

    def build_context(self) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = []
        if self.summary.strip():
            msgs.append({"role": "system", "content": f"以下是更早对话的摘要记忆：\n{self.summary.strip()}"})
        msgs.extend(self.recent_messages)
        return msgs

    def add_turn(self, user_text: str, assistant_text: str):
        self.recent_messages.append({"role": "user", "content": user_text})
        self.recent_messages.append({"role": "assistant", "content": assistant_text})


@dataclass
class TurnTrace:
    session_name: str
    turn_index: int
    user: str
    tags: List[str]
    expected_tool: Optional[str]
    acceptable_tools: Optional[List[str]]

    plan_raw: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    tool_result: Optional[Dict[str, Any]] = None
    memory_summary: str = ""

    planning_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    answer_latency_ms: float = 0.0
    summary_latency_ms: float = 0.0
    summary_triggered: bool = False

    passed: bool = False
    failure_type: Optional[str] = None
    failure_detail: Optional[str] = None


class VLLMChatClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float = 0.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def list_models(self) -> List[str]:
        resp = self.client.models.list()
        return [m.id for m in resp.data]

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 160) -> Tuple[str, float]:
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        content = resp.choices[0].message.content or ""
        return content.strip(), latency_ms


class LocalTools:
    def get_weather(self, city: str):
        if not city or not city.strip():
            return {"ok": False, "data": None, "error": "city 不能为空"}
        city = city.strip()
        if city not in CITY_COORDS:
            return {"ok": False, "data": None, "error": f"暂不支持城市：{city}"}
        # 非交互测试版本仍保留 mock 数据，保证可重复性
        temp_seed = sum(ord(ch) for ch in city) % 12
        wind_seed = sum(ord(ch) for ch in city) % 15
        return {
            "ok": True,
            "data": {
                "city": city,
                "temperature_c": round(16 + temp_seed * 0.8, 1),
                "windspeed_kmh": round(5 + wind_seed * 1.1, 1),
                "source": "mock_repeatable_weather",
            },
            "error": None,
        }

    def get_time(self, city: str):
        if not city or not city.strip():
            return {"ok": False, "data": None, "error": "city 不能为空"}
        city = city.strip()
        if city not in CITY_TIMEZONES:
            return {"ok": False, "data": None, "error": f"暂不支持城市时区：{city}"}
        fake_time = {
            "上海": "2026-04-15 10:30:00",
            "北京": "2026-04-15 10:30:00",
            "深圳": "2026-04-15 10:30:00",
            "广州": "2026-04-15 10:30:00",
            "东京": "2026-04-15 11:30:00",
            "纽约": "2026-04-14 22:30:00",
            "洛杉矶": "2026-04-14 19:30:00",
            "旧金山": "2026-04-14 19:30:00",
        }[city]
        return {
            "ok": True,
            "data": {
                "city": city,
                "timezone": CITY_TIMEZONES[city],
                "time": fake_time,
                "weekday": "Wednesday" if city in {"上海", "北京", "深圳", "广州", "东京"} else "Tuesday",
                "source": "mock_repeatable_time",
            },
            "error": None,
        }

    def calculator(self, expression: str):
        if not expression or not expression.strip():
            return {"ok": False, "data": None, "error": "expression 不能为空"}
        try:
            value = safe_eval_expr(expression.strip())
            return {"ok": True, "data": {"expression": expression.strip(), "result": value}, "error": None}
        except Exception as e:
            return {"ok": False, "data": None, "error": f"计算失败：{e}"}

    def call(self, tool_name: str, args: dict) -> Tuple[Dict[str, Any], float]:
        t0 = time.perf_counter()
        if tool_name == "get_weather":
            result = self.get_weather(args.get("city", ""))
        elif tool_name == "get_time":
            result = self.get_time(args.get("city", ""))
        elif tool_name == "calculator":
            result = self.calculator(args.get("expression", ""))
        else:
            result = {"ok": False, "data": None, "error": f"unknown tool: {tool_name}"}
        return result, round((time.perf_counter() - t0) * 1000, 2)


class APIDatasetAgent:
    def __init__(self, llm: VLLMChatClient, tools: LocalTools):
        self.llm = llm
        self.tools = tools

    def summarize_if_needed(self, memory: ConversationMemory) -> Tuple[bool, float]:
        if len(memory.recent_messages) <= MAX_RECENT_MESSAGES:
            return False, 0.0
        old_chunk = memory.recent_messages[:-KEEP_RECENT_MESSAGES]
        memory.recent_messages = memory.recent_messages[-KEEP_RECENT_MESSAGES:]
        transcript = "\n".join(("用户" if m["role"] == "user" else "助手") + f"：{m['content']}" for m in old_chunk)
        existing_summary = memory.summary.strip() if memory.summary.strip() else "无"
        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"已有摘要：\n{existing_summary}\n\n需要压缩的新对话：\n{transcript}\n\n请输出更新后的摘要。"},
        ]
        new_summary, latency_ms = self.llm.chat(messages, max_tokens=180)
        if new_summary:
            memory.summary = new_summary
        return True, latency_ms

    def run_turn(self, session_name: str, turn_index: int, turn_spec: Dict[str, Any], memory: ConversationMemory, state: ConversationState) -> TurnTrace:
        user_input = turn_spec["user"]
        trace = TurnTrace(
            session_name=session_name,
            turn_index=turn_index,
            user=user_input,
            tags=turn_spec.get("tags", []),
            expected_tool=turn_spec.get("expected_tool"),
            acceptable_tools=turn_spec.get("acceptable_tools"),
        )

        context = memory.build_context()
        planning_messages = [{"role": "system", "content": TOOL_SYSTEM_PROMPT}, *context, {"role": "user", "content": user_input}]
        plan_raw, planning_latency_ms = self.llm.chat(planning_messages, max_tokens=96)
        trace.plan_raw = plan_raw
        trace.planning_latency_ms = planning_latency_ms
        plan = safe_parse_plan(plan_raw, user_input, state)
        trace.plan = plan

        if plan["tool"] == "direct_answer":
            final_messages = [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                *context,
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": f"路由结果：{json.dumps(plan, ensure_ascii=False)}"},
                {"role": "user", "content": "请基于当前问题、历史上下文和路由结果生成最终答复。"},
            ]
            answer, answer_latency_ms = self.llm.chat(final_messages, max_tokens=180)
            trace.answer = answer
            trace.answer_latency_ms = answer_latency_ms
            memory.add_turn(user_input, answer)
            state.update(plan)
            trace.summary_triggered, trace.summary_latency_ms = self.summarize_if_needed(memory)
            trace.memory_summary = memory.summary
            return trace

        tool_result, tool_latency_ms = self.tools.call(plan["tool"], plan.get("args", {}))
        trace.tool_result = tool_result
        trace.tool_latency_ms = tool_latency_ms
        final_messages = [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            *context,
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
            {"role": "user", "content": f"工具名：{plan['tool']}\n工具参数：{json.dumps(plan.get('args', {}), ensure_ascii=False)}\n工具返回：{json.dumps(tool_result, ensure_ascii=False)}\n请基于上述信息回答用户。"},
        ]
        answer, answer_latency_ms = self.llm.chat(final_messages, max_tokens=180)
        trace.answer = answer
        trace.answer_latency_ms = answer_latency_ms
        memory.add_turn(user_input, answer)
        state.update(plan)
        trace.summary_triggered, trace.summary_latency_ms = self.summarize_if_needed(memory)
        trace.memory_summary = memory.summary
        return trace


def tool_match(actual_tool: str, turn_spec: Dict[str, Any]) -> bool:
    if "acceptable_tools" in turn_spec:
        return actual_tool in turn_spec["acceptable_tools"]
    if "expected_tool" in turn_spec:
        return actual_tool == turn_spec["expected_tool"]
    return True


def check_contains_all(answer: str, items: List[str]) -> bool:
    return all(x in answer for x in items)


def check_contains_any(answer: str, items: List[str]) -> bool:
    return any(x in answer for x in items)


def evaluate_trace(trace: TurnTrace, turn_spec: Dict[str, Any]) -> TurnTrace:
    tool = trace.plan.get("tool", "unknown")
    if not tool_match(tool, turn_spec):
        trace.passed = False
        trace.failure_type = "route_error"
        trace.failure_detail = f"tool mismatch, actual_tool={tool}"
        return trace

    if tool in {"get_weather", "get_time", "calculator"} and trace.tool_result is not None and trace.tool_result.get("ok") is False:
        trace.passed = False
        trace.failure_type = "tool_error"
        trace.failure_detail = trace.tool_result.get("error")
        return trace

    if "expected_contains_all" in turn_spec and not check_contains_all(trace.answer, turn_spec["expected_contains_all"]):
        trace.passed = False
        trace.failure_type = "memory_error" if "memory" in trace.tags else "answer_error"
        trace.failure_detail = f"missing all keywords: {turn_spec['expected_contains_all']}"
        return trace

    if "expected_contains_any" in turn_spec and not check_contains_any(trace.answer, turn_spec["expected_contains_any"]):
        trace.passed = False
        trace.failure_type = "memory_error" if "memory" in trace.tags else "answer_error"
        trace.failure_detail = f"missing any keywords: {turn_spec['expected_contains_any']}"
        return trace

    trace.passed = True
    return trace


def aggregate_stats(traces: List[TurnTrace]) -> Dict[str, Any]:
    total = len(traces)
    passed = sum(1 for t in traces if t.passed)
    by_tool: Dict[str, Dict[str, Any]] = {}
    by_tag: Dict[str, Dict[str, Any]] = {}
    by_failure: Dict[str, int] = {}
    by_route_source: Dict[str, Dict[str, Any]] = {}
    for t in traces:
        tool = t.plan.get("tool", "unknown")
        by_tool.setdefault(tool, {"total": 0, "passed": 0})
        by_tool[tool]["total"] += 1
        by_tool[tool]["passed"] += int(t.passed)
        route_source = t.plan.get("route_source", "unknown")
        by_route_source.setdefault(route_source, {"total": 0, "passed": 0})
        by_route_source[route_source]["total"] += 1
        by_route_source[route_source]["passed"] += int(t.passed)
        if t.failure_type:
            by_failure[t.failure_type] = by_failure.get(t.failure_type, 0) + 1
        for tag in t.tags:
            by_tag.setdefault(tag, {"total": 0, "passed": 0})
            by_tag[tag]["total"] += 1
            by_tag[tag]["passed"] += int(t.passed)

    def with_rate(d: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for k, v in d.items():
            out[k] = {**v, "pass_rate": round(v["passed"] / v["total"], 4) if v["total"] else 0.0}
        return out

    return {
        "total_turns": total,
        "passed_turns": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "avg_latency_ms": {
            "planning": round(statistics.mean([t.planning_latency_ms for t in traces]), 2) if traces else 0.0,
            "tool": round(statistics.mean([t.tool_latency_ms for t in traces]), 2) if traces else 0.0,
            "answer": round(statistics.mean([t.answer_latency_ms for t in traces]), 2) if traces else 0.0,
            "summary": round(statistics.mean([t.summary_latency_ms for t in traces]), 2) if traces else 0.0,
        },
        "by_tool": with_rate(by_tool),
        "by_tag": with_rate(by_tag),
        "by_failure_type": by_failure,
        "by_route_source": with_rate(by_route_source),
    }


def save_results(traces: List[TurnTrace], summary: Dict[str, Any], out_json: Path, out_jsonl: Path):
    payload = {
        "summary": summary,
        "traces": [asdict(t) for t in traces],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_jsonl.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM API 版 Day 7 agent 工程化评测")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--api-key", default=DEFAULT_API_KEY)
    p.add_argument("--model", default=None, help="模型 ID；为空时自动选择 /v1/models 的第一个")
    p.add_argument("--out", default=DEFAULT_RESULTS_JSON)
    p.add_argument("--trace-jsonl", default=DEFAULT_TRACE_JSONL)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    probe_client = VLLMChatClient(args.base_url, args.api_key, model="placeholder")
    models = probe_client.list_models()
    if not models:
        raise RuntimeError("/v1/models 返回空列表，vLLM 服务可能未启动")
    model_name = args.model or models[0]

    llm = VLLMChatClient(args.base_url, args.api_key, model=model_name)
    tools = LocalTools()
    agent = APIDatasetAgent(llm, tools)

    traces: List[TurnTrace] = []
    print(f"Using model: {model_name}")
    print(f"Base URL: {args.base_url}")

    for session in DATASET:
        memory = ConversationMemory()
        state = ConversationState()
        print("\n" + "=" * 100)
        print(f"SESSION: {session['session_name']}")
        for idx, turn in enumerate(session["turns"], start=1):
            trace = agent.run_turn(session["session_name"], idx, turn, memory, state)
            trace = evaluate_trace(trace, turn)
            traces.append(trace)
            print("-" * 100)
            print(f"[TURN {idx}] USER: {trace.user}")
            print(f"PLAN: {json.dumps(trace.plan, ensure_ascii=False)}")
            print(f"ANSWER: {trace.answer}")
            print(
                f"PASS: {trace.passed} | failure_type={trace.failure_type} | "
                f"planning={trace.planning_latency_ms}ms | tool={trace.tool_latency_ms}ms | "
                f"answer={trace.answer_latency_ms}ms | summary={trace.summary_latency_ms}ms"
            )
            if args.debug:
                print(f"PLAN_RAW: {trace.plan_raw}")
                if trace.tool_result is not None:
                    print(f"TOOL_RESULT: {json.dumps(trace.tool_result, ensure_ascii=False)}")
                if trace.memory_summary:
                    print(f"SUMMARY_MEMORY: {trace.memory_summary}")

    summary = aggregate_stats(traces)
    out_json = Path(args.out)
    out_jsonl = Path(args.trace_jsonl)
    save_results(traces, summary, out_json, out_jsonl)

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary+traces JSON: {out_json}")
    print(f"Saved trace JSONL: {out_jsonl}")


if __name__ == "__main__":
    main()
