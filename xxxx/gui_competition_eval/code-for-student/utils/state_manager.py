from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


@dataclass
class StepTrace:
    step: int
    action: str
    parameters: Dict[str, Any]
    raw_output: str = ""


@dataclass
class AgentState:
    instruction: str = ""
    target_app: str = ""
    steps: List[StepTrace] = field(default_factory=list)
    parse_failures: int = 0


class GUIStateManager:
    """多步状态管理器。

    职责：
    - 保存任务级状态和最近动作摘要
    - 对模型输出做最后一层安全修正
    - 提供轻量上下文，避免直接把所有历史截图重复送入模型
    """

    APP_CANDIDATES = [
        "爱奇艺", "百度地图", "哔哩哔哩", "抖音", "快手", "芒果TV", "美团", "腾讯视频", "喜马拉雅", "QQ", "微信",
        "淘宝", "京东", "高德地图", "饿了么", "小红书", "中兴管家",
    ]

    def __init__(self) -> None:
        self.state = AgentState()

    def reset(self) -> None:
        self.state = AgentState()

    def bootstrap(self, instruction: str) -> None:
        if not self.state.instruction:
            self.state.instruction = instruction
            self.state.target_app = self._extract_target_app(instruction)

    def build_context_text(self, current_image: Image.Image, step_count: int) -> str:
        width, height = current_image.size
        recent = self.state.steps[-4:]
        recent_lines = []
        for item in recent:
            recent_lines.append(f"- step={item.step}, action={item.action}, params={item.parameters}")
        recent_text = "\n".join(recent_lines) if recent_lines else "- 无历史动作"
        target_app_text = self.state.target_app or "未明确识别"
        return (
            f"任务指令: {self.state.instruction}\n"
            f"目标应用候选: {target_app_text}\n"
            f"当前步数: {step_count}\n"
            f"当前截图尺寸: {width}x{height}\n"
            f"最近动作历史:\n{recent_text}\n"
            f"要求: 输出一个最合理的下一步动作，坐标使用 0~1000 归一化。"
        )

    def maybe_first_step_open(self, instruction: str, step_count: int) -> Optional[Tuple[str, Dict[str, Any]]]:
        if step_count != 1:
            return None
        app_name = self._extract_target_app(instruction)
        if app_name:
            return "OPEN", {"app_name": app_name}
        return None

    def postprocess(self, action: str, parameters: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        action = action.upper().strip()
        parameters = dict(parameters or {})

        if action == "CLICK":
            point = parameters.get("point", [500, 500])
            parameters["point"] = self._clamp_point(point)
        elif action == "SCROLL":
            start = self._clamp_point(parameters.get("start_point", [500, 700]))
            end = self._clamp_point(parameters.get("end_point", [500, 300]))
            if start == end:
                end = [start[0], max(0, start[1] - 400)]
            parameters = {"start_point": start, "end_point": end}
        elif action == "TYPE":
            text = str(parameters.get("text", "")).strip()
            parameters = {"text": text}
        elif action == "OPEN":
            app_name = str(parameters.get("app_name", "")).strip()
            if not app_name and self.state.target_app:
                app_name = self.state.target_app
            parameters = {"app_name": app_name}
        elif action == "COMPLETE":
            parameters = {}

        if self._is_repeated(action, parameters):
            if action == "CLICK":
                point = parameters.get("point", [500, 500])
                parameters["point"] = [min(1000, point[0] + 20), min(1000, point[1] + 20)]
            elif action == "SCROLL":
                start = parameters["start_point"]
                end = parameters["end_point"]
                parameters["start_point"] = [start[0], min(1000, start[1] + 60)]
                parameters["end_point"] = [end[0], max(0, end[1] - 60)]
        return action, parameters

    def record_step(self, step: int, action: str, parameters: Dict[str, Any], raw_output: str) -> None:
        self.state.steps.append(StepTrace(step=step, action=action, parameters=dict(parameters), raw_output=raw_output))

    def record_parse_failure(self) -> None:
        self.state.parse_failures += 1

    def _is_repeated(self, action: str, parameters: Dict[str, Any]) -> bool:
        if len(self.state.steps) < 2:
            return False
        recent = self.state.steps[-2:]
        return all(item.action == action and item.parameters == parameters for item in recent)

    @staticmethod
    def _clamp_point(point: Any) -> List[int]:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return [500, 500]
        out = []
        for v in point:
            try:
                iv = int(round(float(v)))
            except Exception:
                iv = 500
            out.append(max(0, min(1000, iv)))
        return out

    def _extract_target_app(self, instruction: str) -> str:
        instruction = instruction or ""
        for app in self.APP_CANDIDATES:
            if app in instruction:
                return app
        return ""
