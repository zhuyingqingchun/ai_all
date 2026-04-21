from __future__ import annotations

from typing import Any, Dict, List

from agent_base import AgentInput, AgentOutput, BaseAgent
from utils.action_parser import ActionParseError, ActionParser
from utils.state_manager import GUIStateManager


class Agent(BaseAgent):
    """推荐参赛实现：
    1. 首步优先用规则 OPEN 目标 App
    2. 其余步骤用 VLM 直接决策下一步
    3. 用鲁棒解析器把多种模型输出统一成标准动作
    4. 用状态管理器压缩历史并做最后一层安全修正
    """

    def _initialize(self):
        self.parser = ActionParser()
        self.state = GUIStateManager()

    def reset(self):
        self.state.reset()

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        self.state.bootstrap(input_data.instruction)
        context_text = self.state.build_context_text(input_data.current_image, input_data.step_count)
        system_prompt = (
            "你是一个移动端 GUI Agent。请只输出一个 JSON 对象，不要输出 markdown 代码块、不要解释。\n"
            "合法动作只有：CLICK, TYPE, SCROLL, OPEN, COMPLETE。\n"
            "参数格式必须严格符合以下之一：\n"
            '{"action":"CLICK","parameters":{"point":[x,y]}}\n'
            '{"action":"TYPE","parameters":{"text":"内容"}}\n'
            '{"action":"SCROLL","parameters":{"start_point":[x1,y1],"end_point":[x2,y2]}}\n'
            '{"action":"OPEN","parameters":{"app_name":"应用名"}}\n'
            '{"action":"COMPLETE","parameters":{}}\n'
            "坐标必须是 0~1000 的归一化整数。\n"
            "如果已经完成用户任务，则输出 COMPLETE。\n"
            "避免重复给出与最近两步完全相同的动作。"
        )
        user_text = (
            f"{context_text}\n\n"
            "请基于当前截图判断：最合理的下一步是什么？"
        )
        return [
            {"role": "system", "content": system_prompt},
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
            action, parameters = self.state.postprocess(action, parameters)
            self.state.record_step(input_data.step_count, action, parameters, raw_output="[heuristic first-step open]")
            return AgentOutput(action=action, parameters=parameters, raw_output="[heuristic first-step open]")

        messages = self.generate_messages(input_data)
        response = self._call_api(messages, temperature=0)
        usage = self.extract_usage_info(response)
        raw_output = getattr(response.choices[0].message, "content", "") or ""

        try:
            parsed = self.parser.parse(raw_output)
            action, parameters = self.state.postprocess(parsed.action, parsed.parameters)
        except ActionParseError:
            self.state.record_parse_failure()
            # 最后兜底：尝试完成，避免因为格式问题直接输出空动作
            action, parameters = "COMPLETE", {}

        self.state.record_step(input_data.step_count, action, parameters, raw_output=raw_output)
        return AgentOutput(
            action=action,
            parameters=parameters,
            raw_output=raw_output,
            usage=usage,
        )
