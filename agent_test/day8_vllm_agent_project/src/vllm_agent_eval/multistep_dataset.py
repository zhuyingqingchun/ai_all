from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REGRESSION_MULTISTEP_DATASET: List[Dict[str, Any]] = [
    {
        "session_name": "multistep_regression_cases",
        "turns": [
            {
                "user": "先告诉我东京现在几点，再算一下(23+7)*3。",
                "acceptable_tool_sequences": [["get_time", "calculator"]],
                "expected_contains_all": ["东京", "90"],
                "tags": ["multistep", "regression", "time", "calculator"],
            },
            {
                "user": "先算 23*7，再算 sqrt(81)，最后告诉我两个结果。",
                "acceptable_tool_sequences": [["calculator", "calculator"]],
                "expected_contains_all": ["161", "9"],
                "tags": ["multistep", "regression", "calculator"],
            },
            {
                "user": "先查南京天气，再算一下 12*12，并说明哪一步失败了。",
                "acceptable_tool_sequences": [["get_weather", "calculator"], ["calculator"]],
                "expected_contains_all": ["南京", "144"],
                "expected_contains_any": ["失败", "暂不支持", "无法"],
                "tags": ["multistep", "regression", "boundary", "partial_failure"],
            },
        ],
    }
]

DEFAULT_MULTISTEP_DATASET: List[Dict[str, Any]] = [
    {
        "session_name": "day8_multistep_dataset",
        "turns": [
            {
                "user": "先告诉我东京现在几点，再算一下(23+7)*3。",
                "acceptable_tool_sequences": [["get_time", "calculator"]],
                "expected_contains_all": ["东京", "90"],
                "tags": ["multistep", "time", "calculator"],
            },
            {
                "user": "先查北京天气，再告诉我上海现在几点。",
                "acceptable_tool_sequences": [["get_weather", "get_time"]],
                "expected_contains_all": ["北京", "上海"],
                "tags": ["multistep", "weather", "time"],
            },
            {
                "user": "先算 23*7，再算 sqrt(81)，最后告诉我两个结果。",
                "acceptable_tool_sequences": [["calculator", "calculator"]],
                "expected_contains_all": ["161", "9"],
                "tags": ["multistep", "calculator"],
            },
            {
                "user": "先查广州天气，再算一下 12*12。",
                "acceptable_tool_sequences": [["get_weather", "calculator"]],
                "expected_contains_all": ["广州", "144"],
                "tags": ["multistep", "weather", "calculator"],
            },
            {
                "user": "先算 50+50，再告诉我纽约现在几点。",
                "acceptable_tool_sequences": [["calculator", "get_time"]],
                "expected_contains_all": ["100", "纽约"],
                "tags": ["multistep", "calculator", "time"],
            },
            {
                "user": "先查深圳天气，再算一下 25*4，最后告诉我两个结果。",
                "acceptable_tool_sequences": [["get_weather", "calculator"]],
                "expected_contains_all": ["深圳", "100"],
                "tags": ["multistep", "weather", "calculator"],
            },
            {
                "user": "先告诉我洛杉矶现在几点，再算一下(15+25)*2。",
                "acceptable_tool_sequences": [["get_time", "calculator"]],
                "expected_contains_all": ["洛杉矶", "80"],
                "tags": ["multistep", "time", "calculator"],
            },
            {
                "user": "先查旧金山天气，再算 81除以9，最后用两行总结。",
                "acceptable_tool_sequences": [["get_weather", "calculator"]],
                "expected_contains_all": ["旧金山", "9"],
                "tags": ["multistep", "weather", "calculator"],
            },
            {
                "user": "先算 36除以6，再告诉我东京时间。",
                "acceptable_tool_sequences": [["calculator", "get_time"]],
                "expected_contains_all": ["6", "东京"],
                "tags": ["multistep", "calculator", "time"],
            },
            {
                "user": "先查南京天气，再算一下 12*12，并说明哪一步失败了。",
                "acceptable_tool_sequences": [["get_weather", "calculator"], ["calculator"]],
                "expected_contains_all": ["南京", "144"],
                "expected_contains_any": ["失败", "暂不支持", "无法"],
                "tags": ["multistep", "boundary", "partial_failure"],
            },
            {
                "user": "先算 100-25，再告诉我上海现在几点。",
                "acceptable_tool_sequences": [["calculator", "get_time"]],
                "expected_contains_all": ["75", "上海"],
                "tags": ["multistep", "calculator", "time"],
            },
            {
                "user": "先查纽约天气，再算一下 18*3。",
                "acceptable_tool_sequences": [["get_weather", "calculator"]],
                "expected_contains_all": ["纽约", "54"],
                "tags": ["multistep", "weather", "calculator"],
            },
            {
                "user": "先告诉我深圳现在几点，再算 sqrt(144)。",
                "acceptable_tool_sequences": [["get_time", "calculator"]],
                "expected_contains_all": ["深圳", "12"],
                "tags": ["multistep", "time", "calculator"],
            },
            {
                "user": "先算 7*8，再查广州天气。",
                "acceptable_tool_sequences": [["calculator", "get_weather"]],
                "expected_contains_all": ["56", "广州"],
                "tags": ["multistep", "calculator", "weather"],
            },
            {
                "user": "先查北京天气，再告诉我旧金山现在几点。",
                "acceptable_tool_sequences": [["get_weather", "get_time"]],
                "expected_contains_all": ["北京", "旧金山"],
                "tags": ["multistep", "weather", "time"],
            },
        ],
    }
]


def _load_dataset_file(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_multistep_dataset(path: str | None = None) -> List[Dict[str, Any]]:
    if path is None:
        return DEFAULT_MULTISTEP_DATASET
    return _load_dataset_file(path)


def load_regression_multistep_dataset(path: str | None = None) -> List[Dict[str, Any]]:
    if path is None:
        return REGRESSION_MULTISTEP_DATASET
    return _load_dataset_file(path)


def save_default_multistep_dataset(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_MULTISTEP_DATASET, f, ensure_ascii=False, indent=2)


def save_regression_multistep_dataset(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(REGRESSION_MULTISTEP_DATASET, f, ensure_ascii=False, indent=2)
