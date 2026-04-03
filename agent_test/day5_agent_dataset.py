import json
import re
import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"

DEBUG = False
SAVE_RESULTS_PATH = "day5_eval_results.json"

# 记忆设置
MAX_RECENT_MESSAGES = 8      # 最近原始消息最多保留 8 条（4 轮）
KEEP_RECENT_MESSAGES = 4     # 发生摘要时，保留最近 4 条原始消息
SUMMARY_MAX_NEW_TOKENS = 180

# ----------------------------
# 1) 模型加载
# ----------------------------
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=None,
        low_cpu_mem_usage=False,
    ).eval().cuda()

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    return tokenizer, model


# ----------------------------
# 2) 推理
# ----------------------------
def generate(tokenizer, model, messages, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ----------------------------
# 3) JSON 解析与兜底
# ----------------------------
def extract_json(text: str):
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

    t = t.replace("多少", "")
    t = t.replace("等于多少", "")
    t = t.replace("等于几", "")
    t = t.replace("等于", "")
    t = t.replace("=?", "")
    t = t.replace("＝", "=")
    t = t.replace("=", "")
    t = t.strip()

    return bool(re.fullmatch(r"[0-9\.\+\-\*/%\(\)\s]+", t))


def extract_math_expression(text: str) -> str:
    expr = text
    expr = expr.replace("多少", "")
    expr = expr.replace("等于多少", "")
    expr = expr.replace("等于几", "")
    expr = expr.replace("等于", "")
    expr = expr.replace("=?", "")
    expr = expr.replace("＝", "=")
    expr = expr.replace("=", "")
    return expr.strip()


def has_weather_intent(text: str) -> bool:
    kws = ["天气", "气温", "温度", "下雨", "风速", "冷不冷", "热不热"]
    return any(k in text for k in kws)


def has_time_intent(text: str) -> bool:
    kws = ["几点", "时间", "现在几点", "当地时间", "现在几时"]
    return any(k in text for k in kws)


def has_history_intent(text: str) -> bool:
    kws = [
        "我问过",
        "我们刚才聊了什么",
        "我们聊到哪了",
        "还记得吗",
        "你刚才说过",
        "前面聊了什么",
        "总结一下我们前面",
        "回顾一下",
    ]
    return any(k in text for k in kws)


def find_supported_city(text: str, city_map: Dict[str, Any]) -> Optional[str]:
    # 优先匹配更长名字，避免“旧金山”被“金山”之类误伤
    for city in sorted(city_map.keys(), key=len, reverse=True):
        if city in text:
            return city
    return None


def safe_parse_plan(plan_text: str, user_input: str, weather_cities: Dict[str, Any], time_cities: Dict[str, Any]):
    # 本地优先兜底：数学、历史、天气、时间
    if looks_like_math_expression(user_input):
        return {
            "tool": "calculator",
            "args": {"expression": extract_math_expression(user_input)}
        }

    if has_history_intent(user_input):
        return {
            "tool": "direct_answer",
            "args": {}
        }

    if has_weather_intent(user_input):
        city = find_supported_city(user_input, weather_cities)
        if city:
            return {
                "tool": "get_weather",
                "args": {"city": city}
            }

    if has_time_intent(user_input):
        city = find_supported_city(user_input, time_cities)
        if city:
            return {
                "tool": "get_time",
                "args": {"city": city}
            }

    try:
        plan = extract_json(plan_text)
        if not isinstance(plan, dict):
            raise ValueError("plan is not dict")
        if "tool" not in plan:
            raise ValueError("missing tool field")
        if "args" not in plan or not isinstance(plan["args"], dict):
            plan["args"] = {}

        tool = plan["tool"]
        if tool not in {"get_weather", "get_time", "calculator", "direct_answer"}:
            raise ValueError(f"unknown tool: {tool}")

        return plan

    except Exception:
        return {
            "tool": "direct_answer",
            "args": {}
        }


# ----------------------------
# 4) 工具
# ----------------------------
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


def get_weather(city: str):
    if not city or not city.strip():
        return {"ok": False, "data": None, "error": "city 不能为空"}

    city = city.strip()
    if city not in CITY_COORDS:
        return {"ok": False, "data": None, "error": f"暂不支持城市：{city}"}

    lat, lon = CITY_COORDS[city]

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
        }
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        payload = r.json()
        current = payload["current_weather"]

        return {
            "ok": True,
            "data": {
                "city": city,
                "temperature_c": current.get("temperature"),
                "windspeed_kmh": current.get("windspeed"),
                "time": current.get("time"),
                "source": "open-meteo",
            },
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "data": None, "error": f"天气接口调用失败：{e}"}


def get_time(city: str):
    if not city or not city.strip():
        return {"ok": False, "data": None, "error": "city 不能为空"}

    city = city.strip()
    tz_name = CITY_TIMEZONES.get(city)
    if not tz_name:
        return {"ok": False, "data": None, "error": f"暂不支持城市时区：{city}"}

    try:
        now = datetime.now(ZoneInfo(tz_name))
        return {
            "ok": True,
            "data": {
                "city": city,
                "timezone": tz_name,
                "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "weekday": now.strftime("%A"),
            },
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "data": None, "error": f"时间工具调用失败：{e}"}


_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

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

_ALLOWED_CONSTS = {
    "pi": math.pi,
    "e": math.e,
}


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


def calculator(expression: str):
    if not expression or not expression.strip():
        return {"ok": False, "data": None, "error": "expression 不能为空"}

    expression = expression.strip()
    try:
        value = safe_eval_expr(expression)
        return {
            "ok": True,
            "data": {
                "expression": expression,
                "result": value,
            },
            "error": None,
        }
    except Exception as e:
        return {"ok": False, "data": None, "error": f"计算失败：{e}"}


def call_local_tool(tool_name: str, args: dict):
    if tool_name == "get_weather":
        return get_weather(args.get("city", ""))
    if tool_name == "get_time":
        return get_time(args.get("city", ""))
    if tool_name == "calculator":
        return calculator(args.get("expression", ""))
    return {"ok": False, "data": None, "error": f"unknown tool: {tool_name}"}


# ----------------------------
# 5) Prompt
# ----------------------------
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
- 请结合当前问题、对话历史、历史摘要、工具返回来回答
- 如果工具返回 ok=true，只根据工具结果回答，不要编造
- 如果工具返回 ok=false，直接告诉用户失败原因
- 如果用户问“我刚才问过什么”“我们聊到哪了”“还记得吗”，请基于对话历史和历史摘要准确回答
- 如果用户输入是纯数学表达式，也可以直接给出自然中文答案
- 用自然、简洁、纯中文回答
"""

SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要助手。

请把“更早的对话内容”压缩成一段中文摘要，要求：
1. 只保留后续回答真正有用的信息
2. 保留：用户问过的主题、已经查过的城市/时间/天气、做过的计算、模型给过的关键答复
3. 保留：未解决的问题或偏好（如果有）
4. 不要抄原文，不要写成逐字对话
5. 输出 80~180 字中文摘要
"""


# ----------------------------
# 6) 摘要记忆
# ----------------------------
@dataclass
class ConversationMemory:
    summary: str = ""
    recent_messages: List[Dict[str, str]] = field(default_factory=list)

    def build_context(self) -> List[Dict[str, str]]:
        msgs = []
        if self.summary.strip():
            msgs.append({
                "role": "system",
                "content": f"以下是更早对话的摘要记忆：\n{self.summary.strip()}",
            })
        msgs.extend(self.recent_messages)
        return msgs

    def add_turn(self, user_text: str, assistant_text: str):
        self.recent_messages.append({"role": "user", "content": user_text})
        self.recent_messages.append({"role": "assistant", "content": assistant_text})

    def maybe_summarize(self, tokenizer, model):
        if len(self.recent_messages) <= MAX_RECENT_MESSAGES:
            return

        old_chunk = self.recent_messages[:-KEEP_RECENT_MESSAGES]
        self.recent_messages = self.recent_messages[-KEEP_RECENT_MESSAGES:]

        transcript_lines = []
        for m in old_chunk:
            role = "用户" if m["role"] == "user" else "助手"
            transcript_lines.append(f"{role}：{m['content']}")
        transcript = "\n".join(transcript_lines)

        existing_summary = self.summary.strip() if self.summary.strip() else "无"

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"已有摘要：\n{existing_summary}\n\n"
                    f"需要压缩的新对话：\n{transcript}\n\n"
                    f"请输出更新后的摘要。"
                ),
            },
        ]

        try:
            new_summary = generate(
                tokenizer,
                model,
                messages,
                max_new_tokens=SUMMARY_MAX_NEW_TOKENS,
            ).strip()
            if new_summary:
                self.summary = new_summary
        except Exception:
            # 摘要失败时保守回退：简单拼接
            fallback = []
            if self.summary.strip():
                fallback.append(self.summary.strip())
            fallback.append("更早对话涉及：" + "；".join(
                [m["content"][:40] for m in old_chunk if m["role"] == "user"][:6]
            ))
            self.summary = " ".join(fallback).strip()


# ----------------------------
# 7) Agent
# ----------------------------
class LocalAgent:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def run_turn(self, user_input: str, memory: ConversationMemory, debug: bool = DEBUG):
        context = memory.build_context()

        planning_messages = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            *context,
            {"role": "user", "content": user_input},
        ]
        plan_text = generate(self.tokenizer, self.model, planning_messages, max_new_tokens=96)

        if debug:
            print("\n=== TOOL PLAN RAW ===")
            print(plan_text)

        plan = safe_parse_plan(
            plan_text,
            user_input,
            CITY_COORDS,
            CITY_TIMEZONES,
        )
        tool_name = plan["tool"]
        args = plan.get("args", {})

        if debug:
            print("\n=== TOOL PLAN PARSED ===")
            print(json.dumps(plan, ensure_ascii=False, indent=2))

        if tool_name == "direct_answer":
            final_messages = [
                {"role": "system", "content": FINAL_SYSTEM_PROMPT},
                *context,
                {"role": "user", "content": user_input},
            ]
            final_text = generate(self.tokenizer, self.model, final_messages, max_new_tokens=160)
            memory.add_turn(user_input, final_text)
            memory.maybe_summarize(self.tokenizer, self.model)
            return {
                "plan": plan,
                "tool_result": None,
                "answer": final_text,
                "memory_summary": memory.summary,
            }

        tool_result = call_local_tool(tool_name, args)

        if debug:
            print("\n=== TOOL RESULT ===")
            print(json.dumps(tool_result, ensure_ascii=False, indent=2))

        final_messages = [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            *context,
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
            {
                "role": "user",
                "content": (
                    f"工具名：{tool_name}\n"
                    f"工具参数：{json.dumps(args, ensure_ascii=False)}\n"
                    f"工具返回：{json.dumps(tool_result, ensure_ascii=False)}\n"
                    f"请基于上述信息回答用户。"
                ),
            },
        ]
        final_text = generate(self.tokenizer, self.model, final_messages, max_new_tokens=160)

        memory.add_turn(user_input, final_text)
        memory.maybe_summarize(self.tokenizer, self.model)

        return {
            "plan": plan,
            "tool_result": tool_result,
            "answer": final_text,
            "memory_summary": memory.summary,
        }


# ----------------------------
# 8) 数据集
# ----------------------------
DATASET = [
    {
        "session_name": "basic_tools_and_memory",
        "turns": [
            {
                "user": "北京天气如何",
                "expected_tool": "get_weather",
                "expected_contains_any": ["北京", "气温", "温度", "℃"],
            },
            {
                "user": "纽约现在几点",
                "expected_tool": "get_time",
                "expected_contains_any": ["纽约", "时间", "点", "星期"],
            },
            {
                "user": "45454+87987987",
                "expected_tool": "calculator",
                "expected_contains_any": ["88033441", "计算结果"],
            },
            {
                "user": "我问过你哪些问题",
                "expected_tool": "direct_answer",
                "expected_contains_all": ["北京", "纽约"],
            },
            {
                "user": "还记得我刚才问过纽约的问题吗",
                "expected_tool": "direct_answer",
                "expected_contains_all": ["纽约"],
                "expected_contains_any": ["时间", "几点"],
            },
        ],
    },
    {
        "session_name": "summary_memory_rollover",
        "turns": [
            {
                "user": "上海天气如何",
                "expected_tool": "get_weather",
                "expected_contains_any": ["上海", "气温", "温度", "℃"],
            },
            {
                "user": "东京现在几点",
                "expected_tool": "get_time",
                "expected_contains_any": ["东京", "时间", "点", "星期"],
            },
            {
                "user": "1+1",
                "expected_tool": "calculator",
                "expected_contains_any": ["2", "计算结果"],
            },
            {
                "user": "深圳天气如何",
                "expected_tool": "get_weather",
                "expected_contains_any": ["深圳", "气温", "温度", "℃"],
            },
            {
                "user": "旧金山现在几点",
                "expected_tool": "get_time",
                "expected_contains_any": ["旧金山", "时间", "点", "星期"],
            },
            {
                "user": "总结一下我们前面聊过哪些城市和任务",
                "expected_tool": "direct_answer",
                "expected_contains_all": ["上海", "东京", "深圳"],
                "expected_contains_any": ["天气", "时间", "计算"],
            },
        ],
    },
]


# ----------------------------
# 9) 评测
# ----------------------------
def check_contains_all(answer: str, items: List[str]) -> bool:
    return all(x in answer for x in items)


def check_contains_any(answer: str, items: List[str]) -> bool:
    return any(x in answer for x in items)


def evaluate_turn(result: Dict[str, Any], turn_spec: Dict[str, Any]) -> Dict[str, Any]:
    answer = result["answer"]
    tool = result["plan"]["tool"]

    tool_ok = True
    contains_all_ok = True
    contains_any_ok = True

    if "expected_tool" in turn_spec:
        tool_ok = (tool == turn_spec["expected_tool"])

    if "expected_contains_all" in turn_spec:
        contains_all_ok = check_contains_all(answer, turn_spec["expected_contains_all"])

    if "expected_contains_any" in turn_spec:
        contains_any_ok = check_contains_any(answer, turn_spec["expected_contains_any"])

    passed = tool_ok and contains_all_ok and contains_any_ok

    return {
        "passed": passed,
        "tool_ok": tool_ok,
        "contains_all_ok": contains_all_ok,
        "contains_any_ok": contains_any_ok,
    }


def run_dataset(tokenizer, model, dataset: List[Dict[str, Any]], debug: bool = DEBUG):
    agent = LocalAgent(tokenizer, model)
    all_results = []
    total_turns = 0
    passed_turns = 0

    for session in dataset:
        session_name = session["session_name"]
        memory = ConversationMemory()
        session_results = []

        print("\n" + "=" * 100)
        print(f"SESSION: {session_name}")

        for idx, turn in enumerate(session["turns"], start=1):
            user_text = turn["user"]
            total_turns += 1

            result = agent.run_turn(user_text, memory, debug=debug)
            eval_info = evaluate_turn(result, turn)

            if eval_info["passed"]:
                passed_turns += 1

            row = {
                "turn_index": idx,
                "user": user_text,
                "plan": result["plan"],
                "tool_result": result["tool_result"],
                "answer": result["answer"],
                "memory_summary": result["memory_summary"],
                "evaluation": eval_info,
            }
            session_results.append(row)

            print("-" * 100)
            print(f"[TURN {idx}] USER: {user_text}")
            print(f"PLAN: {json.dumps(result['plan'], ensure_ascii=False)}")
            print(f"ANSWER: {result['answer']}")
            print(f"PASS: {eval_info['passed']} | tool_ok={eval_info['tool_ok']} | "
                  f"contains_all_ok={eval_info['contains_all_ok']} | "
                  f"contains_any_ok={eval_info['contains_any_ok']}")

            if result["memory_summary"]:
                print(f"SUMMARY_MEMORY: {result['memory_summary']}")

        all_results.append({
            "session_name": session_name,
            "results": session_results,
        })

    final_report = {
        "total_turns": total_turns,
        "passed_turns": passed_turns,
        "pass_rate": round(passed_turns / total_turns, 4) if total_turns else 0.0,
        "sessions": all_results,
    }

    return final_report


# ----------------------------
# 10) 主程序
# ----------------------------
def main():
    print("正在加载模型...")
    tokenizer, model = load_model()
    print("模型已加载，开始跑 Day 5 数据集评测。")

    report = run_dataset(tokenizer, model, DATASET, debug=DEBUG)

    print("\n" + "=" * 100)
    print("FINAL REPORT")
    print(json.dumps(
        {
            "total_turns": report["total_turns"],
            "passed_turns": report["passed_turns"],
            "pass_rate": report["pass_rate"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    with open(SAVE_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {SAVE_RESULTS_PATH}")


if __name__ == "__main__":
    main()