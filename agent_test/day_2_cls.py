import json
import re
import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"


# ----------------------------
# 1) 加载模型
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

    # 避免 generation_config 触发 warning
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    return tokenizer, model


# ----------------------------
# 2) 推理函数
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
# 3) JSON 解析
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


# ----------------------------
# 4) 工具实现
# ----------------------------
CITY_TIMEZONES = {
    "上海": "Asia/Shanghai",
    "北京": "Asia/Shanghai",
    "深圳": "Asia/Shanghai",
    "广州": "Asia/Shanghai",
    "香港": "Asia/Hong_Kong",
    "东京": "Asia/Tokyo",
    "首尔": "Asia/Seoul",
    "新加坡": "Asia/Singapore",
    "伦敦": "Europe/London",
    "巴黎": "Europe/Paris",
    "柏林": "Europe/Berlin",
    "纽约": "America/New_York",
    "洛杉矶": "America/Los_Angeles",
    "旧金山": "America/Los_Angeles",
}

FAKE_WEATHER = {
    "上海": {"weather": "晴", "temp_c": 26},
    "北京": {"weather": "多云", "temp_c": 22},
    "深圳": {"weather": "小雨", "temp_c": 28},
    "广州": {"weather": "阴", "temp_c": 27},
    "东京": {"weather": "晴", "temp_c": 21},
    "纽约": {"weather": "多云", "temp_c": 18},
}


def get_weather(city: str):
    city = city.strip()
    info = FAKE_WEATHER.get(city, {"weather": "未知", "temp_c": None})
    return {
        "city": city,
        "weather": info["weather"],
        "temp_c": info["temp_c"],
        "source": "mock_weather",
    }


def get_time(city: str):
    city = city.strip()
    tz_name = CITY_TIMEZONES.get(city)
    if not tz_name:
        return {
            "city": city,
            "error": f"unknown city timezone: {city}",
            "supported_cities": sorted(CITY_TIMEZONES.keys()),
        }

    now = datetime.now(ZoneInfo(tz_name))
    return {
        "city": city,
        "timezone": tz_name,
        "time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


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
                raise ValueError(f"不支持的运算符: {op_type.__name__}")
            return _ALLOWED_BIN_OPS[op_type](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY_OPS:
                raise ValueError(f"不支持的一元运算符: {op_type.__name__}")
            return _ALLOWED_UNARY_OPS[op_type](_eval(node.operand))

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("只允许简单函数名调用")
            func_name = node.func.id
            if func_name not in _ALLOWED_FUNCS:
                raise ValueError(f"不支持的函数: {func_name}")
            args = [_eval(arg) for arg in node.args]
            return _ALLOWED_FUNCS[func_name](*args)

        if isinstance(node, ast.Name):
            if node.id in _ALLOWED_CONSTS:
                return _ALLOWED_CONSTS[node.id]
            raise ValueError(f"未知变量: {node.id}")

        raise ValueError(f"不支持的表达式节点: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)


def calculator(expression: str):
    expression = expression.strip()
    try:
        value = safe_eval_expr(expression)
        return {
            "expression": expression,
            "result": value,
        }
    except Exception as e:
        return {
            "expression": expression,
            "error": str(e),
        }


def call_local_tool(tool_name: str, args: dict):
    if tool_name == "get_weather":
        return get_weather(args.get("city", ""))
    if tool_name == "get_time":
        return get_time(args.get("city", ""))
    if tool_name == "calculator":
        return calculator(args.get("expression", ""))
    raise ValueError(f"unknown tool: {tool_name}")


# ----------------------------
# 5) Prompt
# ----------------------------
TOOL_SYSTEM_PROMPT = """你是一个工具路由助手。
你必须在以下 4 种 JSON 中选择 1 种输出，并且只能输出 JSON，不能输出任何额外文字。

可选工具：
1. get_weather
   用途：查询某个城市的天气
   JSON格式：
   {"tool":"get_weather","args":{"city":"上海"}}

2. get_time
   用途：查询某个城市当前时间
   JSON格式：
   {"tool":"get_time","args":{"city":"东京"}}

3. calculator
   用途：计算数学表达式
   JSON格式：
   {"tool":"calculator","args":{"expression":"(23+7)*3"}}

4. direct_answer
   用途：不需要工具时直接回答
   JSON格式：
   {"tool":"direct_answer","args":{"answer":"这里直接回答用户"}}

规则：
- 只输出一个 JSON 对象
- 不要输出 markdown，不要输出解释
- 问天气时优先用 get_weather
- 问时间时优先用 get_time
- 问数学计算时优先用 calculator
- 问常识类、不需要外部工具时用 direct_answer
- 当使用 direct_answer 时，answer 必须是自然、完整、简洁的中文，不要夹杂英文，除非用户明确要求
"""

FINAL_SYSTEM_PROMPT = """你是一个简洁、自然的中文助手。
请用自然、简洁、纯中文回答。
如果给了工具结果，请基于工具结果回答，不要编造额外事实。
如果工具结果里有 error，要直接告诉用户该工具调用失败，并简要说明原因。
"""


# ----------------------------
# 6) 单轮 Agent
# ----------------------------
def run_agent(tokenizer, model, user_query: str, verbose: bool = True):
    planning_messages = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    tool_plan_text = generate(tokenizer, model, planning_messages, max_new_tokens=96)

    if verbose:
        print("\n=== TOOL PLAN ===")
        print(tool_plan_text)

    plan = extract_json(tool_plan_text)
    tool_name = plan["tool"]
    args = plan.get("args", {})

    if tool_name == "direct_answer":
        answer = args.get("answer", "")
        if verbose:
            print("\n=== FINAL ANSWER ===")
            print(answer)
        return answer

    tool_result = call_local_tool(tool_name, args)

    if verbose:
        print("\n=== TOOL RESULT ===")
        print(json.dumps(tool_result, ensure_ascii=False, indent=2))

    final_messages = [
        {"role": "system", "content": FINAL_SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": tool_plan_text},
        {
            "role": "user",
            "content": (
                f"工具名: {tool_name}\n"
                f"工具参数: {json.dumps(args, ensure_ascii=False)}\n"
                f"工具返回: {json.dumps(tool_result, ensure_ascii=False)}\n"
                f"请基于工具返回给出最终中文答复。"
            ),
        },
    ]

    final_text = generate(tokenizer, model, final_messages, max_new_tokens=128)

    if verbose:
        print("\n=== FINAL ANSWER ===")
        print(final_text)

    return final_text


# ----------------------------
# 7) CLI 交互
# ----------------------------
def interactive_cli():
    print("正在加载模型...")
    tokenizer, model = load_model()
    print("模型已就绪。")
    print("输入问题开始对话。输入 exit / quit / q 退出。")
    print("你可以试试：")
    print("- 帮我查一下上海今天天气")
    print("- 东京现在几点了？")
    print("- 帮我算一下 (23 + 7) * 3")
    print("- 什么是 AI agent？")

    while True:
        try:
            user_query = input("\nUSER> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_query:
            continue

        if user_query.lower() in {"exit", "quit", "q"}:
            print("已退出。")
            break

        try:
            print("=" * 80)
            run_agent(tokenizer, model, user_query, verbose=True)
        except Exception as e:
            print("\n[ERROR]")
            print(str(e))


if __name__ == "__main__":
    interactive_cli()