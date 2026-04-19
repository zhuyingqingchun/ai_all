from pathlib import Path

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_API_KEY = "dummy"
DEFAULT_RESULTS_JSON = Path("results/day8_vllm_agent_eval_results.json")
DEFAULT_TRACE_JSONL = Path("results/day8_vllm_agent_eval_traces.jsonl")

MAX_RECENT_MESSAGES = 8
KEEP_RECENT_MESSAGES = 4
SUMMARY_MAX_TOKENS = 180
ANSWER_MAX_TOKENS = 160
PLANNING_MAX_TOKENS = 96

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

STRUCTURED_LOOKUP_TABLE = {
    "servo_max_torque": {
        "key": "servo_max_torque",
        "value": "2.5 N·m",
        "category": "servo_spec",
        "source": "mock_structured_lookup",
    },
    "servo_nominal_voltage": {
        "key": "servo_nominal_voltage",
        "value": "28 V",
        "category": "servo_spec",
        "source": "mock_structured_lookup",
    },
    "qwen2.5-7b_context_length": {
        "key": "qwen2.5-7b_context_length",
        "value": 32768,
        "category": "model_profile",
        "source": "mock_structured_lookup",
    },
    "next80b_fp8_context_length": {
        "key": "next80b_fp8_context_length",
        "value": 65536,
        "category": "model_profile",
        "source": "mock_structured_lookup",
    },
}
