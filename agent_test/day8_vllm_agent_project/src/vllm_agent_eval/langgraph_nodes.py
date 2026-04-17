from __future__ import annotations

from .langgraph_state import LangGraphPOCState
from .routing import safe_parse_plan
from .tools import CITY_COORDS, CITY_TIMEZONES, LocalTools


def planner_node(state: LangGraphPOCState) -> LangGraphPOCState:
    user_input = state["user_input"]
    
    # 简单启发式多步规划
    has_weather = any(k in user_input for k in ["天气", "气温", "温度"])
    has_time = any(k in user_input for k in ["几点", "时间", "现在几点"])
    has_calc = any(k in user_input for k in ["算", "计算", "等于"])
    
    # 如果同时有多个意图，返回多步计划
    if (has_weather + has_time + has_calc) > 1:
        plan_steps = []
        if has_weather:
            plan_steps.append({"tool": "get_weather", "args": {"city": "北京"}})
        if has_time:
            plan_steps.append({"tool": "get_time", "args": {"city": "上海"}})
        if has_calc:
            plan_steps.append({"tool": "calculator", "args": {"expression": "(23+7)*3"}})
        
        plan = {"tool": "multistep", "steps": plan_steps, "route_source": "heuristic_multistep"}
    else:
        plan = safe_parse_plan(
            plan_text='{"tool":"direct_answer","args":{}}',
            user_input=user_input,
            state=None,
        )
    
    trace = list(state.get("debug_trace", []))
    trace.append(f"planner: {plan}")
    return {**state, "plan": plan, "debug_trace": trace}


def _plan_to_steps(plan: dict) -> list[dict]:
    tool = plan.get("tool")
    if tool == "multistep":
        return plan.get("steps", [])
    if tool in {"get_weather", "get_time", "calculator"}:
        return [{"tool": tool, "args": plan.get("args", {})}]
    return []


def executor_node(state: LangGraphPOCState) -> LangGraphPOCState:
    plan = state.get("plan", {})
    steps = _plan_to_steps(plan)
    trace = list(state.get("debug_trace", []))
    tool_results = []
    local_tools = LocalTools()

    for step in steps:
        result, _ = local_tools.call(step["tool"], step.get("args", {}))
        tool_results.append(
            {
                "tool": step["tool"],
                "args": step.get("args", {}),
                "result": result,
            }
        )

    trace.append(f"executor: {tool_results}")
    return {**state, "tool_results": tool_results, "debug_trace": trace}


def synthesizer_node(state: LangGraphPOCState) -> LangGraphPOCState:
    user_input = state.get("user_input", "")
    tool_results = state.get("tool_results", [])
    trace = list(state.get("debug_trace", []))

    if not tool_results:
        answer = f"[LangGraph POC] 直接回答模式：{user_input}"
    else:
        parts = [f"{item['tool']} => {item['result']}" for item in tool_results]
        answer = "[LangGraph POC] " + " | ".join(parts)

    trace.append("synthesizer: final_answer_built")
    return {**state, "final_answer": answer, "debug_trace": trace}
