from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from .client import VLLMChatClient
from .config import ANSWER_MAX_TOKENS
from .multistep_evaluator import build_multistep_plan
from .multistep_prompts import MULTISTEP_FINAL_SYSTEM_PROMPT
from .schemas import ConversationState
from .tools import LocalTools


class LangGraphParityState(TypedDict, total=False):
    user_input: str
    context_messages: List[Dict[str, str]]
    conversation_state: Dict[str, Any]
    plan_raw: str
    plan: Dict[str, Any]
    step_results: List[Dict[str, Any]]
    final_answer: str
    planning_latency_ms: float
    answer_latency_ms: float
    debug_trace: List[str]


class LangGraphParityRuntime:
    def __init__(self, chat_client: VLLMChatClient, tools: LocalTools | None = None, debug: bool = False) -> None:
        self.chat_client = chat_client
        self.tools = tools or LocalTools()
        self.debug = debug

    def _restore_conversation_state(self, payload: Dict[str, Any] | None) -> ConversationState:
        payload = payload or {}
        return ConversationState(
            last_tool=payload.get("last_tool"),
            last_city=payload.get("last_city"),
            last_topic=payload.get("last_topic"),
            seen_cities=list(payload.get("seen_cities", [])),
            asked_topics=list(payload.get("asked_topics", [])),
        )

    def dump_conversation_state(self, state: ConversationState) -> Dict[str, Any]:
        return {
            "last_tool": state.last_tool,
            "last_city": state.last_city,
            "last_topic": state.last_topic,
            "seen_cities": list(state.seen_cities),
            "asked_topics": list(state.asked_topics),
        }

    def planner_node(self, graph_state: LangGraphParityState) -> LangGraphParityState:
        user_input = graph_state.get("user_input", "")
        context_messages = list(graph_state.get("context_messages", []))
        conversation_state = self._restore_conversation_state(graph_state.get("conversation_state"))
        plan, plan_raw, planning_latency_ms = build_multistep_plan(
            self.chat_client,
            context_messages,
            user_input,
            conversation_state,
        )
        debug_trace = list(graph_state.get("debug_trace", []))
        debug_trace.append(
            f"planner route_source={plan.get('route_source', 'unknown')} steps={len(plan.get('steps', []))}"
        )
        return {
            **graph_state,
            "plan": plan,
            "plan_raw": plan_raw,
            "planning_latency_ms": planning_latency_ms,
            "debug_trace": debug_trace,
        }

    def executor_node(self, graph_state: LangGraphParityState) -> LangGraphParityState:
        plan = graph_state.get("plan", {})
        steps = list(plan.get("steps", []))
        conversation_state = self._restore_conversation_state(graph_state.get("conversation_state"))
        debug_trace = list(graph_state.get("debug_trace", []))
        step_results: List[Dict[str, Any]] = []

        for index, step in enumerate(steps, start=1):
            result, latency_ms = self.tools.call(step["tool"], step.get("args", {}))
            step_results.append(
                {
                    "step_index": index,
                    "tool": step["tool"],
                    "args": step.get("args", {}),
                    "purpose": step.get("purpose", ""),
                    "result": result,
                    "latency_ms": latency_ms,
                }
            )
            conversation_state.update({"tool": step["tool"], "args": step.get("args", {})})

        debug_trace.append(f"executor finished steps={len(step_results)}")
        return {
            **graph_state,
            "step_results": step_results,
            "conversation_state": self.dump_conversation_state(conversation_state),
            "debug_trace": debug_trace,
        }

    def synthesizer_node(self, graph_state: LangGraphParityState) -> LangGraphParityState:
        user_input = graph_state.get("user_input", "")
        context_messages = list(graph_state.get("context_messages", []))
        plan = graph_state.get("plan", {})
        step_results = list(graph_state.get("step_results", []))
        final_messages = [
            {"role": "system", "content": MULTISTEP_FINAL_SYSTEM_PROMPT},
            *context_messages,
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": f"多步计划：{plan}"},
            {
                "role": "user",
                "content": f"步骤执行结果：{step_results}\n请基于这些步骤结果给出最终答复。",
            },
        ]
        final_answer, answer_latency_ms = self.chat_client.chat(final_messages, max_tokens=ANSWER_MAX_TOKENS)
        debug_trace = list(graph_state.get("debug_trace", []))
        debug_trace.append("synthesizer built final answer")
        return {
            **graph_state,
            "final_answer": final_answer,
            "answer_latency_ms": answer_latency_ms,
            "debug_trace": debug_trace,
        }

    def build_graph(self):
        builder = StateGraph(LangGraphParityState)
        builder.add_node("planner", self.planner_node)
        builder.add_node("executor", self.executor_node)
        builder.add_node("synthesizer", self.synthesizer_node)
        builder.add_edge(START, "planner")
        builder.add_edge("planner", "executor")
        builder.add_edge("executor", "synthesizer")
        builder.add_edge("synthesizer", END)
        return builder.compile()
