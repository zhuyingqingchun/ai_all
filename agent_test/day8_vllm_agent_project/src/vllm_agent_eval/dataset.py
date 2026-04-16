from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_DATASET = [{'session_name': 'basic_tools_and_memory', 'turns': [{'user': '北京天气如何', 'expected_tool': 'get_weather', 'expected_contains_any': ['北京', '温度', '气温', '℃'], 'tags': ['weather', 'basic']}, {'user': '纽约现在几点', 'expected_tool': 'get_time', 'expected_contains_any': ['纽约', '时间', '点', '星期'], 'tags': ['time', 'basic']}, {'user': '45454+87987987', 'expected_tool': 'calculator', 'expected_contains_any': ['88033441', '计算结果'], 'tags': ['calculator', 'basic']}, {'user': '我问过你哪些问题', 'expected_tool': 'direct_answer', 'expected_contains_all': ['北京', '纽约'], 'tags': ['memory', 'history']}, {'user': '还记得我刚才问过纽约的问题吗', 'expected_tool': 'direct_answer', 'expected_contains_all': ['纽约'], 'expected_contains_any': ['时间', '几点'], 'tags': ['memory', 'history', 'reference']}]}, {'session_name': 'synonym_and_ellipsis', 'turns': [{'user': '北京冷不冷', 'acceptable_tools': ['get_weather', 'direct_answer'], 'expected_contains_any': ['北京', '温度', '气温', '℃', '冷'], 'tags': ['weather', 'synonym']}, {'user': '那纽约呢', 'acceptable_tools': ['get_weather', 'get_time', 'direct_answer'], 'expected_contains_all': ['纽约'], 'tags': ['memory', 'ellipsis', 'reference']}, {'user': '23*7', 'expected_tool': 'calculator', 'expected_contains_any': ['161', '计算结果'], 'tags': ['calculator', 'synonym']}, {'user': '前面我们查过哪些城市', 'expected_tool': 'direct_answer', 'expected_contains_all': ['北京', '纽约'], 'tags': ['memory', 'history']}]}, {'session_name': 'tool_errors_and_boundaries', 'turns': [{'user': '南京天气如何', 'acceptable_tools': ['direct_answer', 'get_weather'], 'expected_contains_any': ['南京', '暂不支持', '抱歉'], 'tags': ['weather', 'unsupported_city', 'boundary']}, {'user': '火星天气怎么样', 'expected_tool': 'direct_answer', 'expected_contains_any': ['无法', '不支持', '抱歉'], 'tags': ['weather', 'boundary']}, {'user': '2/**3', 'acceptable_tools': ['direct_answer', 'calculator'], 'expected_contains_any': ['抱歉', '换一种说法', '无法', '格式有误', '计算失败'], 'tags': ['calculator', 'boundary']}]}, {'session_name': 'summary_memory_rollover', 'turns': [{'user': '上海天气如何', 'expected_tool': 'get_weather', 'expected_contains_any': ['上海', '温度', '气温', '℃'], 'tags': ['weather', 'summary']}, {'user': '东京现在几点', 'expected_tool': 'get_time', 'expected_contains_any': ['东京', '时间', '点', '星期'], 'tags': ['time', 'summary']}, {'user': '1+1', 'expected_tool': 'calculator', 'expected_contains_any': ['2', '计算结果'], 'tags': ['calculator', 'summary']}, {'user': '深圳天气如何', 'expected_tool': 'get_weather', 'expected_contains_any': ['深圳', '温度', '气温', '℃'], 'tags': ['weather', 'summary']}, {'user': '旧金山现在几点', 'expected_tool': 'get_time', 'expected_contains_any': ['旧金山', '时间', '点', '星期'], 'tags': ['time', 'summary']}, {'user': '总结一下我们前面聊过哪些城市和任务', 'expected_tool': 'direct_answer', 'expected_contains_all': ['上海', '东京', '深圳'], 'expected_contains_any': ['天气', '时间', '计算'], 'tags': ['memory', 'summary', 'history']}]}]


def load_dataset(path: str | None = None) -> List[Dict[str, Any]]:
    if path is None:
        return DEFAULT_DATASET
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_default_dataset(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_DATASET, f, ensure_ascii=False, indent=2)
