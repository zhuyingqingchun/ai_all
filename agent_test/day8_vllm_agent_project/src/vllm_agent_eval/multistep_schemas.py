from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MultiStepPlan:
    steps: List[Dict[str, Any]] = field(default_factory=list)
    final_instruction: str = ""
    route_source: str = "unknown"


@dataclass
class MultiStepTurnTrace:
    session_name: str
    turn_index: int
    user: str
    tags: List[str]
    expected_tools_sequence: Optional[List[str]] = None
    acceptable_tool_sequences: Optional[List[List[str]]] = None

    plan_raw: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    memory_summary: str = ""

    planning_latency_ms: float = 0.0
    answer_latency_ms: float = 0.0
    summary_latency_ms: float = 0.0
    summary_triggered: bool = False

    passed: bool = False
    failure_type: Optional[str] = None
    failure_detail: Optional[str] = None
