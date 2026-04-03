import json
import re
import ast
import math
import operator
import requests
from datetime import datetime
from zoneinfo import ZoneInfo

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"

DEBUG = False  # ← 打开可看 TOOL PLAN


# ----------------------------
# 1) 模型加载（稳定配置）
# ----------------------------
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        attn_implementation="sdpa",
        device_map=None,
        low_cpu_mem_usage=False,
    ).eval().cuda()

    # 清理 warning
    if hasattr(model, "generation_config") and model.generation_config:
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
# 3) JSON 提取
# ----------------------------
def extract_json(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON解析失败: {text}")
    return json.loads(match.group(0))


# ----------------------------
# 4) 🌍 真实天气工具（Open-Meteo）
# ----------------------------
CITY_COORDS = {
    "上海": (31.23, 121.47),
    "北京": (39.90, 116.40),
    "深圳": (22.54, 114.06),
    "广州": (23.13, 113.27),
    "东京": (35.68, 139.69),
    "纽约": (40.71, -74.00),
}


def get_weather(city: str):
    if not city:
        return {"ok": False, "data": None, "error": "city不能为空"}

    if city not in CITY_COORDS:
        return {"ok": False, "data": None, "error": f"暂不支持城市: {city}"}

    lat, lon = CITY_COORDS[city]

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": True,
        }

        r = requests.get(url, params=params, timeout=5)
        r.raise_for_status()
        data = r.json()

        current = data["current_weather"]

        return {
            "ok": True,
            "data": {
                "city": city,
                "temperature": current["temperature"],
                "windspeed": current["windspeed"],
                "time": current["time"],
            },
            "error": None,
        }

    except Exception as e:
        return {"ok": False, "data": None, "error": str(e)}


# ----------------------------
# 5) 计算器（安全版）
# ----------------------------
def calculator(expression: str):
    if not expression:
        return {"ok": False, "data": None, "error": "expression不能为空"}

    try:
        result = eval(expression, {"__builtins__": {}})
        return {"ok": True, "data": {"result": result}, "error": None}
    except Exception as e:
        return {"ok": False, "data": None, "error": str(e)}


# ----------------------------
# 6) 工具路由
# ----------------------------
def call_tool(tool, args):
    if tool == "get_weather":
        return get_weather(args.get("city"))
    if tool == "calculator":
        return calculator(args.get("expression"))
    return {"ok": False, "data": None, "error": "unknown tool"}


# ----------------------------
# 7) Prompt
# ----------------------------
TOOL_PROMPT = """你是一个工具路由助手。
只能输出 JSON，不要输出任何解释。

工具：
1. get_weather -> {"tool":"get_weather","args":{"city":"上海"}}
2. calculator -> {"tool":"calculator","args":{"expression":"(2+3)*4"}}
3. direct_answer -> {"tool":"direct_answer","args":{"answer":"中文回答"}}

规则：
- 查询天气 → get_weather
- 数学计算 → calculator
- 其他 → direct_answer
"""

FINAL_PROMPT = """你是一个简洁中文助手。
根据工具返回结果回答用户：

规则：
- 如果 ok=true → 正常回答
- 如果 ok=false → 告诉用户失败原因
"""


# ----------------------------
# 8) Agent
# ----------------------------
def run_agent(tokenizer, model, user_input):
    plan_text = generate(tokenizer, model, [
        {"role": "system", "content": TOOL_PROMPT},
        {"role": "user", "content": user_input},
    ])

    if DEBUG:
        print("PLAN:", plan_text)

    plan = extract_json(plan_text)

    if plan["tool"] == "direct_answer":
        return plan["args"]["answer"]

    tool_result = call_tool(plan["tool"], plan["args"])

    final = generate(tokenizer, model, [
        {"role": "system", "content": FINAL_PROMPT},
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": json.dumps(plan, ensure_ascii=False)},
        {"role": "user", "content": json.dumps(tool_result, ensure_ascii=False)},
    ])

    return final


# ----------------------------
# 9) CLI
# ----------------------------
def cli():
    tokenizer, model = load_model()

    print("Day3 Agent 已启动（真实天气API）")
    print("输入 exit 退出\n")

    while True:
        q = input("USER> ").strip()
        if q in ["exit", "quit", "q"]:
            break

        try:
            ans = run_agent(tokenizer, model, q)
            print("ASSISTANT>", ans)
        except Exception as e:
            print("ERROR:", str(e))


if __name__ == "__main__":
    cli()