from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from utils.candidate_grounding import Candidate


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
    page_guess: str = "desktop_or_home"
    current_subgoal: str = ""
    last_typed_text: str = ""
    repeated_action_count: int = 0
    retry_cursor: int = 0
    steps: List[StepTrace] = field(default_factory=list)


class GUIStateManager:
    APP_CANDIDATES = [
        "爱奇艺", "百度地图", "哔哩哔哩", "抖音", "快手", "芒果TV", "美团", "腾讯视频",
        "喜马拉雅", "QQ", "微信", "淘宝", "京东", "高德地图", "饿了么", "小红书", "中兴管家",
    ]
    RETRY_OFFSETS = [(0, 0), (16, 0), (-16, 0), (0, 16), (0, -16), (24, 12), (-24, 12)]

    def __init__(self) -> None:
        self.state = AgentState()

    def reset(self) -> None:
        self.state = AgentState()

    def bootstrap(self, instruction: str) -> None:
        if not self.state.instruction:
            self.state.instruction = instruction
            self.state.target_app = self._extract_target_app(instruction)
            self.state.current_subgoal = self._infer_subgoal(1)

    def ingest_external_history(self, history_actions: List[Dict[str, Any]]) -> None:
        _ = history_actions

    def build_context_text(
        self,
        current_image: Image.Image,
        step_count: int,
        history_actions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        width, height = current_image.size
        self.state.page_guess = self.infer_page_guess(step_count)
        self.state.current_subgoal = self._infer_subgoal(step_count)
        recent = self.state.steps[-2:]
        recent_text = " | ".join(
            f"step={s.step},action={s.action},params={s.parameters}" for s in recent
        ) if recent else "无"
        return (
            f"用户目标：{self.state.instruction}\n"
            f"目标应用：{self.state.target_app or '未知'}\n"
            f"当前页面猜测：{self.state.page_guess}\n"
            f"当前子目标：{self.state.current_subgoal}\n"
            f"当前步数：{step_count}\n"
            f"截图尺寸：{width}x{height}\n"
            f"最近动作：{recent_text}\n"
            f"最近输入：{self.state.last_typed_text or '无'}\n"
            f"重复动作计数：{self.state.repeated_action_count}\n"
        )

    def maybe_first_step_open(self, instruction: str, step_count: int) -> Optional[Tuple[str, Dict[str, Any]]]:
        if step_count != 1:
            return None
        app = self._extract_target_app(instruction)
        if app:
            return "OPEN", {"app_name": app}
        return None

    def infer_page_guess(self, step_count: int) -> str:
        if step_count <= 1:
            return "desktop_or_home"
        if not self.state.steps:
            return self.state.page_guess
        last = self.state.steps[-1]
        if last.action == "OPEN":
            return "app_home"
        if last.action == "TYPE":
            return "input_or_search_result"
        if last.action == "SCROLL":
            return "list_or_feed"
        if last.action == "CLICK":
            if step_count <= 4:
                return "search_entry"
            if self.state.last_typed_text:
                return "detail_or_transition"
            return "search_result"
        return self.state.page_guess

    def _infer_subgoal(self, step_count: int) -> str:
        if step_count <= 1:
            return "打开目标应用"
        if self.state.target_app == "美团":
            if step_count <= 4:
                return "找到搜索入口或输入框"
            if step_count == 5:
                return "输入店铺关键词"
            if 6 <= step_count <= 8:
                return "进入目标店铺并找到菜品搜索入口"
            if step_count == 9:
                return "输入菜品关键词"
            return "完成下单或确认流程"
        if self.state.target_app == "百度地图":
            if step_count <= 4:
                return "找到搜索框并输入目标地点"
            if 5 <= step_count <= 8:
                return "选择搜索结果或导航入口"
            return "完成导航或设置流程"
        return "继续找到任务相关入口"

    def postprocess(self, action: str, parameters: Dict[str, Any], step_count: int) -> Tuple[str, Dict[str, Any]]:
        action = (action or "").upper().strip()
        parameters = dict(parameters or {})
        if action == "CLICK":
            point = self._clamp_point(parameters.get("point", [500, 500]))
            point = self._maybe_reaim_click(point)
            return "CLICK", {"point": point}
        if action == "TYPE":
            return "TYPE", {"text": str(parameters.get("text", "")).strip()}
        if action == "SCROLL":
            start = self._clamp_point(parameters.get("start_point", [500, 760]))
            end = self._clamp_point(parameters.get("end_point", [500, 280]))
            if start == end:
                end = [start[0], max(0, start[1] - 420)]
            return "SCROLL", {"start_point": start, "end_point": end}
        if action == "OPEN":
            return "OPEN", {"app_name": str(parameters.get("app_name", self.state.target_app)).strip()}
        if action == "COMPLETE":
            return "COMPLETE", {}
        return "CLICK", {"point": [500, 500]}

    def safe_fallback(self, step_count: int, candidates: List[Candidate]) -> Tuple[str, Dict[str, Any]]:
        if candidates:
            return "CLICK", {"point": candidates[0].center}
        if self.state.last_typed_text and step_count >= 10:
            return "CLICK", {"point": [840, 910]}
        if step_count <= 2 and self.state.target_app:
            return "OPEN", {"app_name": self.state.target_app}
        last = self._last_click_point()
        if last is not None:
            return "CLICK", {"point": self._next_retry_point(last)}
        return "CLICK", {"point": [500, 500]}

    def should_prefer_click(self, step_count: int) -> bool:
        if step_count <= 2:
            return True
        if self.state.page_guess in {"app_home", "search_entry", "search_result", "detail_or_transition"}:
            return True
        return False

    def allow_complete(self, step_count: int) -> bool:
        if step_count <= 3:
            return False
        if self.state.target_app == "美团" and step_count < 14:
            return False
        return True

    def normalize_type_text(self, text: str, step_count: int) -> str:
        text = (text or "").strip()
        if not text:
            return text
        if self.state.target_app == "美团":
            if step_count <= 5 and "窑村干锅猪蹄" in text:
                return "窑村干锅猪蹄（科技大学店）"
            if step_count >= 9 and "干锅排骨" in text:
                return "干锅排骨"
        return text

    def should_skip_repeated_type(self, text: str) -> bool:
        text = (text or "").strip()
        if not text or not self.state.last_typed_text:
            return False
        return text == self.state.last_typed_text

    def record_step(self, step: int, action: str, parameters: Dict[str, Any], raw_output: str = "") -> None:
        trace = StepTrace(step=step, action=action, parameters=dict(parameters), raw_output=raw_output)
        if self.state.steps and self.state.steps[-1].action == action and self.state.steps[-1].parameters == parameters:
            self.state.repeated_action_count += 1
        else:
            self.state.repeated_action_count = 0
            self.state.retry_cursor = 0
        if action == "TYPE":
            self.state.last_typed_text = str(parameters.get("text", "")).strip()
        self.state.steps.append(trace)
        self.state.page_guess = self.infer_page_guess(step + 1)
        self.state.current_subgoal = self._infer_subgoal(step + 1)

    def _maybe_reaim_click(self, point: List[int]) -> List[int]:
        last = self._last_click_point()
        if last is None or last != point:
            return point
        return self._next_retry_point(point)

    def _next_retry_point(self, point: List[int]) -> List[int]:
        offset = self.RETRY_OFFSETS[min(self.state.retry_cursor, len(self.RETRY_OFFSETS) - 1)]
        self.state.retry_cursor = min(self.state.retry_cursor + 1, len(self.RETRY_OFFSETS) - 1)
        return self._clamp_point([point[0] + offset[0], point[1] + offset[1]])

    def _last_click_point(self) -> Optional[List[int]]:
        for s in reversed(self.state.steps):
            if s.action == "CLICK" and "point" in s.parameters:
                return self._clamp_point(s.parameters["point"])
        return None

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
