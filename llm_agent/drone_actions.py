"""各機 Agent 的動作白名單與驗證（多 Agent，老師 2026-11-16 建議 2、3）。

分工：
    協調 Agent（commander.py）   選策略（mode）、分派角色（role）—— 規格 §10.5 Plan
    各機 Agent（drone_agent.py） 只決定「自己這個角色現在要怎麼做」，從角色專屬白名單選一個動作
    確定性規劃器（planner/）     把動作換成目標點（座標由程式算，Agent 不輸出座標或速度）

信任邊界：
    - 各機 Agent 只能用自己角色的動作；用別的角色的動作 → 驗證失敗，記入動作日誌但不送出，該架沿用上一個合法動作。
    - 每個角色都可以選 hold（原地懸停），這是 Agent 能做的最保守選擇。
    - drone_id、角色由程式填入，Agent 無法冒充別架或改自己的角色。
"""
import json
from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

DRONE_ACTION_SCHEMA_VERSION = "0.3.0"

ROLE_ACTIONS: Dict[str, Tuple[str, ...]] = {
    # 捕捉策略（mode = capture）
    "herder": ("push", "circle_behind", "hold"),  # 從後方施壓 / 繞到後方 / 懸停
    "capturer": ("ambush", "engage", "hold"),  # 在捕捉區伏擊 / 迎上去抓拍 / 懸停
    "blocker": ("flank", "hold"),  # 守側翼（左右由規劃器依目標偏離方向決定）
    # 包圍與追蹤（mode = encircle / track）
    "chaser": ("follow_slot", "hold"),
    "blocker_left": ("follow_slot", "hold"),
    "blocker_right": ("follow_slot", "hold"),
    "overwatch": ("overwatch", "hold"),
    # 其他
    "searcher": ("orbit", "hold"),
    "idle": ("hold",),
}
ALL_ACTIONS = tuple(sorted({a for v in ROLE_ACTIONS.values() for a in v}))

ACTION_ZH = {
    "push": "從後方施壓", "circle_behind": "繞到目標後方", "ambush": "在捕捉區伏擊", "engage": "迎上去抓拍",
    "flank": "守住側翼", "follow_slot": "維持隊形位置", "overwatch": "高空監看", "orbit": "繞圈搜尋", "hold": "原地懸停",
}


class DroneAction(BaseModel):
    """各機 Agent 的輸出（LLM 或規則式）。"""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    action: Literal[ALL_ACTIONS]
    reason: str = Field(min_length=1, max_length=60)


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    action: Optional[DroneAction] = None
    error: Optional[str] = None


def validate_action(role: str, action: DroneAction) -> Optional[str]:
    """角色專屬白名單。回傳 None 代表通過。"""
    allowed = ROLE_ACTIONS.get(role)
    if allowed is None:
        return f"未知的角色 {role!r}"
    if action.action not in allowed:
        return f"角色 {role} 不能用 {action.action}，只能用 {'/'.join(allowed)}"
    return None


def parse_drone_action(text: str, role: str) -> ActionResult:
    """從 LLM 輸出取出唯一一個 JSON 物件，做格式 + 角色白名單驗證。"""
    if not isinstance(text, str) or not text.strip():
        return ActionResult(False, error="輸出為空")
    start = text.find("{")
    if start < 0:
        return ActionResult(False, error="找不到 JSON 物件")
    try:
        obj, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as e:
        return ActionResult(False, error=f"JSON 格式錯誤：{e.msg}")
    if "{" in text[end:]:
        return ActionResult(False, error="輸出包含多個 JSON 物件")
    if not isinstance(obj, dict):
        return ActionResult(False, error="輸出必須是 JSON 物件")
    try:
        act = DroneAction.model_validate(obj)
    except ValidationError as e:
        return ActionResult(False, error="不符合動作格式：" + "；".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()))
    err = validate_action(role, act)
    return ActionResult(False, error=err) if err else ActionResult(True, action=act)
