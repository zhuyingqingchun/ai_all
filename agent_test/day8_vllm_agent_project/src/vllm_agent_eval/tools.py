from __future__ import annotations

import ast
import math
import operator
import time
from typing import Any, Dict, Tuple

from .config import CITY_COORDS, CITY_TIMEZONES

_ALLOWED_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_ALLOWED_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10, "exp": math.exp, "abs": abs, "round": round}
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
    return _eval(ast.parse(expr, mode="eval"))


class LocalTools:
    def get_weather(self, city: str) -> Dict[str, Any]:
        if not city or not city.strip():
            return {"ok": False, "data": None, "error": "city 不能为空"}
        city = city.strip()
        if city not in CITY_COORDS:
            return {"ok": False, "data": None, "error": f"暂不支持城市：{city}"}
        temp_seed = sum(ord(ch) for ch in city) % 12
        wind_seed = sum(ord(ch) for ch in city) % 15
        return {"ok": True, "data": {"city": city, "temperature_c": round(16 + temp_seed * 0.8, 1), "windspeed_kmh": round(5 + wind_seed * 1.1, 1), "source": "mock_repeatable_weather"}, "error": None}

    def get_time(self, city: str) -> Dict[str, Any]:
        if not city or not city.strip():
            return {"ok": False, "data": None, "error": "city 不能为空"}
        city = city.strip()
        if city not in CITY_TIMEZONES:
            return {"ok": False, "data": None, "error": f"暂不支持城市时区：{city}"}
        fake_time = {"上海": "2026-04-15 10:30:00", "北京": "2026-04-15 10:30:00", "深圳": "2026-04-15 10:30:00", "广州": "2026-04-15 10:30:00", "东京": "2026-04-15 11:30:00", "纽约": "2026-04-14 22:30:00", "洛杉矶": "2026-04-14 19:30:00", "旧金山": "2026-04-14 19:30:00"}[city]
        weekday = "Wednesday" if city in {"上海", "北京", "深圳", "广州", "东京"} else "Tuesday"
        return {"ok": True, "data": {"city": city, "timezone": CITY_TIMEZONES[city], "time": fake_time, "weekday": weekday, "source": "mock_repeatable_time"}, "error": None}

    def calculator(self, expression: str) -> Dict[str, Any]:
        if not expression or not expression.strip():
            return {"ok": False, "data": None, "error": "expression 不能为空"}
        expression = expression.strip()
        try:
            return {"ok": True, "data": {"expression": expression, "result": safe_eval_expr(expression)}, "error": None}
        except Exception as exc:
            return {"ok": False, "data": None, "error": f"计算失败：{exc}"}

    def call(self, tool_name: str, args: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        start = time.perf_counter()
        if tool_name == "get_weather":
            result = self.get_weather(args.get("city", ""))
        elif tool_name == "get_time":
            result = self.get_time(args.get("city", ""))
        elif tool_name == "calculator":
            result = self.calculator(args.get("expression", ""))
        else:
            result = {"ok": False, "data": None, "error": f"unknown tool: {tool_name}"}
        return result, round((time.perf_counter() - start) * 1000, 2)
