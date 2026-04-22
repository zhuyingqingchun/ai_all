from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_base import AgentInput, AgentOutput, BaseAgent
from utils.action_parser import ActionParseError, ActionParser
from utils.candidate_grounding import Candidate, CandidateGrounder
from utils.state_manager import GUIStateManager


class Agent(BaseAgent):
    """候选区域选点 / bbox-to-point / SoM 风格的最小工程版 Agent。"""

    def _initialize(self):
        self.parser = ActionParser()
        self.state = GUIStateManager()
        self.grounder = CandidateGrounder()

    def reset(self):
        self.state.reset()

    def _build_system_prompt(self) -> str:
        return (
            "你是一个安卓手机 GUI Agent。你必须只输出一个动作，不能解释，不能输出思考过程。\n"
            "合法动作只有：OPEN, CLICK, TYPE, SCROLL, COMPLETE。\n"
            "\n"
            "输出格式允许以下几种：\n"
            "1. OPEN:['应用名']\n"
            "2. CLICK:[C1]  （优先使用候选区域 ID）\n"
            "3. CLICK:[[x,y]]\n"
            "4. CLICK:[[x1,y1,x2,y2]]  （如果你更擅长输出框，也允许；系统会自动转中心点）\n"
            "5. TYPE:['内容']\n"
            "6. SCROLL:[[x1,y1],[x2,y2]]\n"
            "7. COMPLETE:[]\n"
            "\n"
            "强规则：\n"
            "- 当需要 CLICK 时，优先从候选区域 C1/C2/C3... 中选择一个，输出 CLICK:[Ck]。\n"
            "- 如果候选区域里没有明显目标，再输出 CLICK:[[x,y]] 或 CLICK:[[x1,y1,x2,y2]]。\n"
            "- 不要输出 JSON。\n"
            "- 不要输出 bbox 和 point 混在一起。\n"
            "- 如果要输入 TYPE，尽量输入当前阶段所需的最短有效关键词。\n"
            "- 在搜索入口、搜索结果、详情页，优先 CLICK，不要过早 SCROLL。\n"
            "- 只有任务真正完成时才输出 COMPLETE。"
        )

    def _build_user_text(self, input_data: AgentInput, candidates: List[Candidate]) -> str:
        self.state.bootstrap(input_data.instruction)
        self.state.ingest_external_history(input_data.history_actions)
        state_text = self.state.build_context_text(
            current_image=input_data.current_image,
            step_count=input_data.step_count,
            history_actions=input_data.history_actions,
        )
        if not candidates:
            cand_text = "当前没有候选区域。"
        else:
            lines = []
            for c in candidates:
                lines.append(
                    f"[{c.candidate_id}] {c.label} bbox=[[{c.bbox[0]},{c.bbox[1]},{c.bbox[2]},{c.bbox[3]}]] center=[[{c.center[0]},{c.center[1]}]]"
                )
            cand_text = "候选区域如下：\n" + "\n".join(lines)
        return state_text + "\n\n" + cand_text + "\n\n请输出下一步唯一动作。"

    def generate_messages(self, input_data: AgentInput, candidates: List[Candidate]) -> List[Dict[str, Any]]:
        user_text = self._build_user_text(input_data, candidates)
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": self._encode_image(input_data.current_image)}},
                ],
            },
        ]

    def act(self, input_data: AgentInput) -> AgentOutput:
        self.state.bootstrap(input_data.instruction)

        heuristic = self.state.maybe_first_step_open(input_data.instruction, input_data.step_count)
        if heuristic is not None:
            action, parameters = heuristic
            self.state.record_step(input_data.step_count, action, parameters, raw_output="[heuristic:first_step_open]")
            return AgentOutput(action=action, parameters=parameters, raw_output="[heuristic:first_step_open]")

        page_guess = self.state.infer_page_guess(input_data.step_count)
        candidates = self.grounder.propose_candidates(
            task_app=self.state.state.target_app,
            page_guess=page_guess,
            step_count=input_data.step_count,
            instruction=input_data.instruction,
        )
        messages = self.generate_messages(input_data, candidates)
        response = self._call_api(messages, temperature=0)
        raw_output = getattr(response.choices[0].message, "content", "") or ""
        usage = self.extract_usage_info(response)

        try:
            parsed = self.parser.parse(raw_output)
            action = parsed.action
            parameters = dict(parsed.parameters)
        except ActionParseError:
            action, parameters = self.state.safe_fallback(input_data.step_count, candidates)

        if action == "CLICK" and "candidate_id" in parameters:
            candidate = self._find_candidate(parameters["candidate_id"], candidates)
            if candidate is not None:
                parameters = {"point": candidate.center}
            else:
                action, parameters = self.state.safe_fallback(input_data.step_count, candidates)

        action, parameters = self.state.postprocess(action, parameters, input_data.step_count)

        if action == "SCROLL" and self.state.should_prefer_click(input_data.step_count):
            action, parameters = self.state.safe_fallback(input_data.step_count, candidates)

        if action == "TYPE":
            text = self.state.normalize_type_text(parameters.get("text", ""), input_data.step_count)
            if self.state.should_skip_repeated_type(text):
                action, parameters = self.state.safe_fallback(input_data.step_count, candidates)
            else:
                parameters = {"text": text}

        if action == "COMPLETE" and not self.state.allow_complete(input_data.step_count):
            action, parameters = self.state.safe_fallback(input_data.step_count, candidates)

        self.state.record_step(input_data.step_count, action, parameters, raw_output=raw_output)
        return AgentOutput(action=action, parameters=parameters, raw_output=raw_output, usage=usage)

    @staticmethod
    def _find_candidate(candidate_id: str, candidates: List[Candidate]) -> Optional[Candidate]:
        for c in candidates:
            if c.candidate_id == candidate_id:
                return c
        return None
