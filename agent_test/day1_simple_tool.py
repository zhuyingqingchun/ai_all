import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/mnt/PRO6000_disk/models/Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    attn_implementation="sdpa",
    device_map=None,
    low_cpu_mem_usage=False,
).eval().cuda()

def call_local_tool(tool_name: str, args: dict):
    if tool_name == "get_weather":
        city = args.get("city", "")
        return {"city": city, "weather": "晴", "temp_c": 26}
    raise ValueError(f"unknown tool: {tool_name}")

def generate(messages, max_new_tokens=128):
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

def extract_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no json found in: {text}")
    return json.loads(m.group(0))

system_prompt = """你是一个工具调用助手。
当用户询问天气时，你必须只输出一个 JSON 对象，不要输出任何额外文字。
格式如下：
{"tool":"get_weather","args":{"city":"城市名"}}

如果不需要调用工具，就直接正常回答。
"""

# 第一步：让模型决定要不要调工具
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "帮我查一下上海今天天气"},
]

tool_plan_text = generate(messages, max_new_tokens=64)
print("=== TOOL PLAN ===")
print(tool_plan_text)

plan = extract_json(tool_plan_text)
tool_result = call_local_tool(plan["tool"], plan["args"])

# 第二步：把工具结果喂回去，让模型生成自然语言答案
messages2 = [
    {"role": "system", "content": "你是一个简洁的中文助手。"},
    {"role": "user", "content": "帮我查一下上海今天天气"},
    {"role": "assistant", "content": tool_plan_text},
    {"role": "user", "content": f"工具返回结果：{json.dumps(tool_result, ensure_ascii=False)}。请用自然语言回答用户。"},
]

final_text = generate(messages2, max_new_tokens=64)
print("\n=== FINAL ANSWER ===")
print(final_text)