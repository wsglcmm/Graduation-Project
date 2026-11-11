"""A ↔ C 多機訊息（規格 §10.2 FleetPose、§10.6 GoalPoint），以及 A 的 agent_server 傳輸層外框。

信任邊界：
    - FleetPose 的 data 只接受規格欄位（extra="forbid"），A 的補充資訊（mode、failsafe_reason、yield_alt_m…）
      放在 WebSocket 訊息外層，C 讀進來但不依賴它控制飛行。
    - GoalPointAck 是 A 的回覆，欄位可能增加，所以寬鬆解析。
"""
from typing import Dict, Literal, Optional

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
