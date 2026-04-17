from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .langgraph_nodes import executor_node, planner_node, synthesizer_node
from .langgraph_state import LangGraphPOCState


def build_langgraph_poc():
    builder = StateGraph(LangGraphPOCState)
    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "executor")
    builder.add_edge("executor", "synthesizer")
    builder.add_edge("synthesizer", END)
    return builder.compile()
