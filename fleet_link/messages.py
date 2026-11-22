"""A ↔ C 多機訊息（規格 §10.2 FleetPose、§10.6 GoalPoint、§10.8 執行狀態），以及 A 的 agent_server 傳輸層外框。

信任邊界：
    - FleetPose 的 data 只接受規格欄位（extra="forbid"），A 的補充資訊（mode、failsafe_reason、yield_alt_m…）
      放在 WebSocket 訊息外層，C 讀進來但不依賴它控制飛行。
    - GoalPointAck、YieldEvent、ActionResult 是 A 的回覆 / 推播，欄位可能增加，所以寬鬆解析；
      它們只用來記錄與驗證（動作日誌、面板），不會回頭影響 C 的控制決策。
"""
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.2.0"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NED(_Strict):
    n: float
    e: float
    d: float


class Attitude(_Strict):
    roll: float
    pitch: float
    yaw: float


class FleetPose(_Strict):
    """規格 §10.2：一架無人機在 world_ned 的位姿（A → C）。"""

    schema_version: Literal["0.2.0"]
    drone_id: str = Field(min_length=1, max_length=16)
    timestamp_ns: int
    position_world_ned_m: NED
    attitude_rad: Attitude
    velocity_world_ned_mps: Optional[NED] = None
    status: Literal["landed", "flying", "failsafe"]


class GoalPointAck(BaseModel):
    """A 對每一則 GoalPoint 的回覆（A 的實作，規格 §10.6）。"""

    model_config = ConfigDict(extra="allow")

    drone_id: Optional[str] = None
    plan_id: Optional[str] = None
    timestamp_ns: Optional[int] = None
    accepted: bool = False
    adjusted: bool = False
    reason: Optional[str] = None
    applied_position_world_ned_m: Optional[Dict[str, float]] = None
    latency_ms: Optional[float] = None
    flight_command: Optional[Dict[str, Any]] = None  # §10.8(a)：實際呼叫的 MAVSDK 函式；被拒絕時為 None


class YieldEvent(BaseModel):
    """§10.8(c)：A 的升降讓路事件。每個 start 之後一定有一個 end。"""

    model_config = ConfigDict(extra="allow")

    event: Literal["start", "end"]
    drone_id: str
    yield_to: Optional[str] = None
    timestamp_ns: Optional[int] = None  # PX4 時間（同 FleetPose）
    action_id: Optional[str] = None
    from_alt_m: Optional[float] = None
    to_alt_m: Optional[float] = None
    direction: Optional[str] = None
    reason: Optional[str] = None  # end：back_to_layer / landed


class ActionResult(BaseModel):
    """§10.8(b)：A 的動作結束結果。"""

    model_config = ConfigDict(extra="allow")

    action_id: Optional[str] = None
    drone_id: Optional[str] = None
    action: Optional[str] = None
    agent_id: Optional[str] = None
    role: Optional[str] = None
    result: Literal["succeeded", "failed", "overridden", "cancelled", "rejected"]
    why: Optional[str] = None
    metrics: Dict[str, Any] = {}
