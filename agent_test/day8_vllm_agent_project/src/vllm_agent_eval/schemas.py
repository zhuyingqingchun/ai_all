from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConversationState:
    last_tool: Optional[str] = None
    last_city: Optional[str] = None
    last_topic: Optional[str] = None
    seen_cities: List[str] = field(default_factory=list)
    asked_topics: List[str] = field(default_factory=list)

    def update(self, plan: dict) -> None:
        tool = plan.get("tool")
        args = plan.get("args", {})
        self.last_tool = tool
        if tool == "get_weather":
            self.last_topic = "weather"
            city = args.get("city")
            if city:
                self.last_city = city
                if city not in self.seen_cities:
                    self.seen_cities.append(city)
        elif tool == "get_time":
            self.last_topic = "time"
            city = args.get("city")
            if city:
                self.last_city = city
                if city not in self.seen_cities:
                    self.seen_cities.append(city)
        elif tool == "calculator":
            self.last_topic = "calculator"
        elif tool == "direct_answer":
            self.last_topic = "chat"
        if self.last_topic and self.last_topic not in self.asked_topics:
            self.asked_topics.append(self.last_topic)


@dataclass
class ConversationMemory:
    summary: str = ""
    recent_messages: List[Dict[str, str]] = field(default_factory=list)

    def build_context(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.summary.strip():
            messages.append({"role": "system", "content": f"以下是更早对话的摘要记忆：\n{self.summary.strip()}"})
        messages.extend(self.recent_messages)
        return messages

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.recent_messages.append({"role": "user", "content": user_text})
        self.recent_messages.append({"role": "assistant", "content": assistant_text})


@dataclass
class TurnTrace:
    session_name: str
    turn_index: int
    user: str
    tags: List[str]
    expected_tool: Optional[str] = None
    acceptable_tools: Optional[List[str]] = None

    plan_raw: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    tool_result: Optional[Dict[str, Any]] = None
    memory_summary: str = ""

    planning_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    answer_latency_ms: float = 0.0
    summary_latency_ms: float = 0.0
    summary_triggered: bool = False

    passed: bool = False
    failure_type: Optional[str] = None
    failure_detail: Optional[str] = None
