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

    def _build_system_prompt(self, instruction: str) -> str:
        return """你是一个手机 GUI Agent。你的任务是：根据【用户指令】和【当前手机截图】，只输出当前这一步最合理的一个动作。

你必须严格遵守以下规则：

1. 你每次只能输出一个动作。
2. 你只能输出以下五种格式之一，不能输出其他任何内容：
CLICK:[[x,y]]
TYPE:['内容']
SCROLL:[[x1,y1],[x2,y2]]
OPEN:['应用名']
COMPLETE:[]

3. 坐标必须是相对坐标：
- 横纵坐标范围都是 0~1000
- [0,0] 是左上角
- [1000,1000] 是右下角

4. 关于 CLICK：
- 如果目标按钮、输入框、商品卡片、搜索框、列表项已经可见，优先 CLICK
- 点击目标元素的中心区域
- 不要点击边缘、空白区域、分隔线、相邻元素
- 如果是根据文字定位，就点击与该文字最对应的可交互区域中心

5. 关于 TYPE：
- 只有当输入框已经被激活或明显应该输入文字时才使用 TYPE
- 只输出要输入的内容，不要额外解释

6. 关于 SCROLL：
- 只有当目标明显不在当前屏幕内时，才允许 SCROLL
- 如果目标已经可见，禁止 SCROLL，优先 CLICK 或 TYPE
- 一次滚动幅度适中，不要过大
- 向下找内容时，通常从屏幕中下部往上滑
- 向上找内容时，通常从屏幕中上部往下滑

7. 关于 OPEN：
- 只有在桌面或应用列表中，需要打开某个应用时才使用 OPEN
- 参数必须是应用名称

8. 关于 COMPLETE（非常重要）：
- 只有在当前截图已经明确显示任务完成时，才能输出 COMPLETE:[]
- 如果还能继续进行下一步操作，就绝不能输出 COMPLETE
- 如果你不确定是否完成，也不能输出 COMPLETE
- 不确定时，优先输出最合理的下一步动作

9. 决策优先级：
- 能点击就先点击
- 输入前先确保输入框已被点击
- 只有目标不在当前屏幕时才滚动
- 绝不提前完成

10. 输出要求：
- 只能输出一个合法动作
- 不能输出解释
- 不能输出思考过程
- 不能输出 JSON
- 不能输出 markdown

用户指令和截图会在后续消息中提供。请根据当前界面直接给出下一步动作。
"""

    def generate_messages(self, input_data: AgentInput) -> List[Dict[str, Any]]:
        self.state.bootstrap(input_data.instruction)
        context_text = self.state.build_context_text(input_data.current_image, input_data.step_count)
        system_prompt = self._build_system_prompt(input_data.instruction)
        user_text = (
            f"任务指令：{input_data.instruction}\n\n"
            f"当前是第 {input_data.step_count} 步。请基于截图输出下一步动作："
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
            # 兜底：解析失败时返回安全动作
            action, parameters = "CLICK", {"point": [500, 500]}

        # 第21轮：前3步强制禁止 COMPLETE
        if action == "COMPLETE" and input_data.step_count <= 3:
            action = "CLICK"
            parameters = {"point": [500, 500]}

        self.state.record_step(input_data.step_count, action, parameters, raw_output=raw_output)
        return AgentOutput(
            action=action,
            parameters=parameters,
            raw_output=raw_output,
            usage=usage,
        )
