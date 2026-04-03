import json
import re
import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"

DEBUG = False
MAX_HISTORY_MESSAGES = 12  # 保留最近 12 条消息（约 6 轮）


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


def safe_parse_plan(plan_text: str):
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

        if tool == "direct_answer" and "answer" not in plan["args"]:
            plan["args"]["answer"] = "抱歉，我没有完全理解你的问题，你可以换一种说法。"

        return plan

    except Exception:
        return {
            "tool": "direct_answer",
            "args": {
                "answer": "抱歉，我没有完全理解你的问题，你可以换一种说法。"
            },
        }


# ----------------------------
# 4) 工具：天气 / 时间 / 计算器
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
用途：不需要工具时直接回答；也用于回答“我们刚才聊了什么”“我问过什么”这类依赖对话历史的问题
格式：
{"tool":"direct_answer","args":{"answer":"这里填写自然、完整、简洁的中文回答"}}

严格规则：
- 包含“天气 / 温度 / 气温 / 下雨 / 风速”等 → 必须用 get_weather
- 包含“几点 / 时间 / 现在几点 / 当地时间”等 → 必须用 get_time
- 包含明确数学计算，如 + - * / sqrt log 等 → 必须用 calculator
- 常识问答、闲聊、总结历史对话、回顾用户问过什么 → 才能用 direct_answer
- 不确定时优先 direct_answer，但 answer 必须是自然中文，不能空
- 只能输出 JSON 对象
"""

FINAL_SYSTEM_PROMPT = """你是一个简洁、自然、可靠的中文助手。

规则：
- 请结合当前问题、对话历史、工具返回来回答
- 如果工具返回 ok=true，只根据工具结果回答，不要编造
- 如果工具返回 ok=false，直接告诉用户失败原因
- 如果用户问“我刚才问过什么”“我们聊到哪了”，请根据对话历史准确总结
- 用自然、简洁、纯中文回答
"""


# ----------------------------
# 6) 历史管理
# ----------------------------
def trim_history(history, max_messages=MAX_HISTORY_MESSAGES):
    if len(history) <= max_messages:
        return history
    return history[-max_messages:]


# ----------------------------
# 7) 单轮 Agent
# ----------------------------
def run_agent(tokenizer, model, user_input: str, history: list, debug: bool = DEBUG):
    history = trim_history(history)

    planning_messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_input},
    ]
    plan_text = generate(tokenizer, model, planning_messages, max_new_tokens=128)

    if debug:
        print("\n=== TOOL PLAN RAW ===")
        print(plan_text)

    plan = safe_parse_plan(plan_text)
    tool_name = plan["tool"]
    args = plan.get("args", {})

    if debug:
        print("\n=== TOOL PLAN PARSED ===")
        print(json.dumps(plan, ensure_ascii=False, indent=2))

    if tool_name == "direct_answer":
        answer = args.get("answer", "抱歉，我没有完全理解你的问题。")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})
        return answer

    tool_result = call_local_tool(tool_name, args)

    if debug:
        print("\n=== TOOL RESULT ===")
        print(json.dumps(tool_result, ensure_ascii=False, indent=2))

    final_messages = [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        *history,
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

    final_text = generate(tokenizer, model, final_messages, max_new_tokens=128)

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": final_text})

    return final_text


# ----------------------------
# 8) CLI
# ----------------------------
def is_exit_command(text: str) -> bool:
    t = text.strip().lower()
    return t in {"exit", "quit", "q", "exit()", "quit()"}


def interactive_cli():
    print("正在加载模型...")
    tokenizer, model = load_model()
    history = []

    print("Day 4 Agent（带记忆）已启动。")
    print("输入 exit / quit / q / exit() / quit() 退出。")
    print("可试：")
    print("- 北京天气如何")
    print("- 纽约现在几点")
    print("- 45454+87987987")
    print("- 我问过你哪些问题")
    print("- 还记得我刚才问过纽约的问题吗")

    while True:
        try:
            user_query = input("\nUSER> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_query:
            continue

        if is_exit_command(user_query):
            print("已退出。")
            break

        try:
            answer = run_agent(tokenizer, model, user_query, history, debug=DEBUG)
            print(f"ASSISTANT> {answer}")
        except Exception as e:
            print(f"ASSISTANT> 抱歉，这一轮处理失败：{e}")


if __name__ == "__main__":
    interactive_cli()