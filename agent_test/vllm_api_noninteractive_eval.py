#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_TIMEOUT = 180


@dataclass
class EvalCase:
    name: str
    description: str
    messages: List[Dict[str, str]]
    expected_contains_any: Optional[List[str]] = None
    expected_contains_all: Optional[List[str]] = None
    max_tokens: int = 256
    temperature: float = 0.0


@dataclass
class EvalResult:
    name: str
    description: str
    latency_ms: float
    response_text: str
    passed: bool
    missing_any: List[str]
    missing_all: List[str]


CASES: List[EvalCase] = [
    EvalCase(
        name="basic_explanation",
        description="基础中文解释能力",
        messages=[
            {"role": "system", "content": "你是一个简洁的中文助手。"},
            {"role": "user", "content": "请用一句话解释什么是 AI agent。"},
        ],
        expected_contains_any=["agent", "智能体", "工具", "任务"],
        max_tokens=128,
    ),
    EvalCase(
        name="json_router_style",
        description="结构化输出能力：只输出 JSON",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严格的工具路由助手。"
                    "只输出一个 JSON 对象，不要输出任何额外文字。"
                    "如果用户输入是数学计算，就输出："
                    '{"tool":"calculator","args":{"expression":"原表达式"}}'
                ),
            },
            {"role": "user", "content": "45454+87987987"},
        ],
        expected_contains_all=["calculator", "45454+87987987"],
        max_tokens=96,
    ),
    EvalCase(
        name="single_request_memory",
        description="单请求多轮上下文记忆能力",
        messages=[
            {"role": "system", "content": "你是一个简洁、准确的中文助手。"},
            {"role": "user", "content": "我刚才查的是纽约时间。"},
            {"role": "assistant", "content": "好的，你刚才查的是纽约时间。"},
            {"role": "user", "content": "我刚才问的是哪个城市？只回答城市名。"},
        ],
        expected_contains_all=["纽约"],
        max_tokens=32,
    ),
    EvalCase(
        name="boundary_invalid_math",
        description="边界输入：非法数学表达式处理倾向",
        messages=[
            {
                "role": "system",
                "content": "你是一个稳健的中文助手。遇到非法表达式时，不要编造结果，要说明格式有误。",
            },
            {"role": "user", "content": "2/**3"},
        ],
        expected_contains_any=["有误", "无法", "错误", "不合法"],
        max_tokens=96,
    ),
    EvalCase(
        name="unsupported_city",
        description="边界输入：不支持城市的处理倾向",
        messages=[
            {
                "role": "system",
                "content": "你是一个中文助手。若用户询问不支持的城市天气，不要编造天气，请直接说明暂不支持。",
            },
            {"role": "user", "content": "帮我查一下火星天气。"},
        ],
        expected_contains_any=["不支持", "无法", "抱歉"],
        max_tokens=96,
    ),
    EvalCase(
        name="long_instruction_following",
        description="长指令遵循与格式控制",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个严谨的中文助手。请严格遵守以下要求："
                    "1. 回答必须是两行；"
                    "2. 第一行以‘结论：’开头；"
                    "3. 第二行以‘原因：’开头；"
                    "4. 不要使用项目符号。"
                ),
            },
            {"role": "user", "content": "为什么说 agent 不是普通聊天机器人？"},
        ],
        expected_contains_all=["结论：", "原因："],
        max_tokens=160,
    ),
]


def get_models(base_url: str, api_key: str, timeout: int) -> List[str]:
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data", [])
    models = [item["id"] for item in data if "id" in item]
    if not models:
        raise RuntimeError(f"未从 {url} 获取到模型列表，返回为: {payload}")
    return models


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> tuple[str, float, Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.perf_counter()
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    r.raise_for_status()
    payload = r.json()
    text = payload["choices"][0]["message"].get("content", "")
    return text, latency_ms, payload


def evaluate_case(response_text: str, case: EvalCase) -> tuple[bool, List[str], List[str]]:
    missing_any: List[str] = []
    missing_all: List[str] = []

    if case.expected_contains_any:
        if not any(x in response_text for x in case.expected_contains_any):
            missing_any = list(case.expected_contains_any)

    if case.expected_contains_all:
        missing_all = [x for x in case.expected_contains_all if x not in response_text]

    passed = (not missing_any) and (not missing_all)
    return passed, missing_any, missing_all


def run_eval(base_url: str, api_key: str, model: str, timeout: int) -> Dict[str, Any]:
    results: List[EvalResult] = []
    raw_payloads: Dict[str, Any] = {}

    for idx, case in enumerate(CASES, start=1):
        text, latency_ms, payload = chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=case.messages,
            max_tokens=case.max_tokens,
            temperature=case.temperature,
            timeout=timeout,
        )
        passed, missing_any, missing_all = evaluate_case(text, case)
        results.append(
            EvalResult(
                name=case.name,
                description=case.description,
                latency_ms=latency_ms,
                response_text=text,
                passed=passed,
                missing_any=missing_any,
                missing_all=missing_all,
            )
        )
        raw_payloads[case.name] = payload

        print("-" * 100)
        print(f"[{idx}] {case.name} | {case.description}")
        print(f"latency_ms: {latency_ms}")
        print(f"passed: {passed}")
        print(f"response: {text}")
        if missing_any:
            print(f"missing_any: {missing_any}")
        if missing_all:
            print(f"missing_all: {missing_all}")

    passed_count = sum(int(r.passed) for r in results)
    avg_latency = round(sum(r.latency_ms for r in results) / len(results), 2) if results else 0.0

    return {
        "base_url": base_url,
        "model": model,
        "total_cases": len(results),
        "passed_cases": passed_count,
        "pass_rate": round(passed_count / len(results), 4) if results else 0.0,
        "avg_latency_ms": avg_latency,
        "results": [asdict(r) for r in results],
        "raw_payloads": raw_payloads,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="针对本地 vLLM OpenAI-compatible 服务做非交互评测。")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="例如 http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY, help="本地 vLLM 服务可填任意非空字符串")
    parser.add_argument("--model", default=None, help="模型名；不填则自动从 /v1/models 取第一个")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP 超时秒数")
    parser.add_argument("--out", default="vllm_api_eval_results.json", help="结果 JSON 保存路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        models = get_models(args.base_url, args.api_key, args.timeout)
    except Exception as e:
        print(f"[ERROR] 获取模型列表失败: {e}", file=sys.stderr)
        return 1

    model = args.model or models[0]

    print("=" * 100)
    print(f"base_url: {args.base_url}")
    print(f"available_models: {models}")
    print(f"selected_model: {model}")

    try:
        report = run_eval(args.base_url, args.api_key, model, args.timeout)
    except Exception as e:
        print(f"[ERROR] 评测执行失败: {e}", file=sys.stderr)
        return 2

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 100)
    print("FINAL REPORT")
    print(json.dumps(
        {
            "base_url": report["base_url"],
            "model": report["model"],
            "total_cases": report["total_cases"],
            "passed_cases": report["passed_cases"],
            "pass_rate": report["pass_rate"],
            "avg_latency_ms": report["avg_latency_ms"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"\n详细结果已保存到: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
