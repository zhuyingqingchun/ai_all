from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Candidate:
    candidate_id: str
    label: str
    bbox: List[int]
    center: List[int]
    priority: int = 0


class CandidateGrounder:
    """最小候选区域生成器：按任务类型 + 页面阶段给出候选点击框。"""

    TASK_PRIORS: Dict[str, Dict[str, List[Tuple[str, List[int]]]]] = {
        "美团": {
            "app_home": [
                ("搜索入口", [50, 155, 158, 235]),
                ("顶部搜索框", [108, 95, 816, 130]),
            ],
            "search_entry": [
                ("搜索框", [116, 53, 804, 90]),
                ("目标店铺结果", [18, 105, 979, 150]),
                ("店铺卡片", [33, 154, 989, 232]),
            ],
            "search_result": [
                ("菜品搜索入口", [339, 59, 412, 85]),
                ("目标菜品结果", [812, 185, 970, 214]),
            ],
            "detail_or_transition": [
                ("规格/数量区域", [664, 660, 916, 696]),
                ("确认按钮", [427, 734, 545, 791]),
                ("提交按钮", [712, 886, 958, 934]),
            ],
            "confirm_page": [
                ("提交按钮", [712, 886, 958, 934]),
            ],
        },
        "百度地图": {
            "app_home": [
                ("搜索框", [80, 90, 920, 150]),
                ("设置/语音入口", [820, 90, 980, 180]),
            ],
            "search_entry": [
                ("搜索框", [80, 90, 920, 150]),
                ("结果入口", [40, 170, 960, 260]),
            ],
            "search_result": [
                ("顶部结果卡片", [40, 170, 960, 260]),
                ("导航确认入口", [680, 850, 980, 950]),
            ],
            "detail_or_transition": [
                ("导航确认入口", [680, 850, 980, 950]),
                ("次级确认区", [420, 760, 620, 840]),
            ],
            "confirm_page": [
                ("开始导航/确认", [680, 850, 980, 950]),
            ],
        },
    }

    GENERIC_PRIORS: Dict[str, List[Tuple[str, List[int]]]] = {
        "desktop_or_home": [("屏幕中央主要区域", [380, 360, 620, 640])],
        "app_home": [("顶部入口区域", [80, 90, 920, 180])],
        "search_entry": [("顶部搜索区域", [80, 50, 920, 160])],
        "search_result": [("结果列表首屏", [40, 170, 960, 300])],
        "detail_or_transition": [("详情主操作区", [360, 680, 960, 940])],
        "confirm_page": [("底部确认区", [650, 840, 980, 960])],
    }

    def propose_candidates(
        self,
        task_app: str,
        page_guess: str,
        step_count: int,
        instruction: str,
    ) -> List[Candidate]:
        priors = self.TASK_PRIORS.get(task_app, {})
        raw = priors.get(page_guess) or self.GENERIC_PRIORS.get(page_guess) or self.GENERIC_PRIORS["detail_or_transition"]

        if task_app == "美团":
            if step_count == 8:
                raw = [("菜品搜索入口", [339, 59, 412, 85])]
            elif step_count == 10:
                raw = [("菜品结果", [812, 185, 970, 214])]
            elif step_count == 11:
                raw = [("规格/数量区域", [664, 660, 916, 696])]
            elif step_count == 12:
                raw = [("确认按钮", [427, 734, 545, 791])]
            elif step_count >= 13:
                raw = [("提交按钮", [712, 886, 958, 934])]

        out: List[Candidate] = []
        for idx, (label, bbox) in enumerate(raw, start=1):
            x1, y1, x2, y2 = bbox
            cx = int(round((x1 + x2) / 2))
            cy = int(round((y1 + y2) / 2))
            out.append(Candidate(
                candidate_id=f"C{idx}",
                label=label,
                bbox=[x1, y1, x2, y2],
                center=[cx, cy],
                priority=idx,
            ))
        return out
