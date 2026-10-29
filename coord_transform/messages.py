"""規格 §3（Detection，B → C）與 §4（Pose，A → C）的訊息驗證。

設計取捨：
    - extra="forbid"：欄位名打錯（例如 "postion_ned_m"）要在整合時就報錯，而不是被默默忽略。
      要新增欄位，請先改 docs/interface_spec.md 並開 PR。
    - 時間戳與 ID 用 StrictInt：奈秒時間戳約 1.7e18，存成 float 會失去約 256 ns 的精度，
      而且 "123" 這種字串不應被默默轉型。
    - 浮點欄位允許整數（JSON 的 30 和 30.0 都合法），但拒絕 NaN / Infinity。
"""
import math
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

from . import rotations

SCHEMA_VERSION = "0.1.0"  # 與 docs/interface_spec.md 版本一致；之後可移到 shared/ 作為單一來源
CLASSES = ("car", "person", "truck", "bus", "motorcycle", "bicycle")
QUAT_EULER_TOL_RAD = math.radians(1.0)  # 四元數與歐拉角不一致的容忍度


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


# ---------- §4 Pose ----------
class NED(_Msg):
    n: float
    e: float
    d: float


class Attitude(_Msg):
    roll: float
    pitch: float
    yaw: float


class Gimbal(_Msg):
    pitch: float
    yaw: float


class PoseMessage(_Msg):
    timestamp_ns: StrictInt
    position_ned_m: NED
    attitude_rad: Attitude
    quaternion_wxyz: Optional[List[float]] = None
    altitude_agl_m: Optional[float] = None  # 目前未使用：平坦地面假設下 = -d（規格 §2.4）
    velocity_ned_mps: Optional[NED] = None
    gimbal_rad: Optional[Gimbal] = None

    @field_validator("quaternion_wxyz")
    @classmethod
    def _quat_unit(cls, q):
        if q is None:
            return q
        if len(q) != 4:
            raise ValueError("quaternion_wxyz 必須有 4 個數值 [w, x, y, z]")
        n = math.sqrt(sum(c * c for c in q))
        if abs(n - 1.0) > 1e-3:
            raise ValueError(f"quaternion_wxyz 不是單位四元數（長度 {n:.4f}）")
        return q

    @model_validator(mode="after")
    def _quat_matches_euler(self):
        """兩者都給時必須一致。不一致通常代表座標系慣例不同（例如 ENU/FLU 沒轉成 NED/FRD）。"""
        if self.quaternion_wxyz is not None:
            a = self.attitude_rad
            diff = rotations.angle_between(self.quaternion_wxyz, rotations.euler_to_quat(a.roll, a.pitch, a.yaw))
            if diff > QUAT_EULER_TOL_RAD:
                raise ValueError(
                    f"quaternion_wxyz 與 attitude_rad 相差 {math.degrees(diff):.2f}°，"
                    "請確認兩者都是「機體 FRD 相對 NED」（規格 §4）"
                )
        return self

    def quaternion(self):
        """優先使用四元數；沒給就由歐拉角換算。"""
        if self.quaternion_wxyz is not None:
            return rotations.normalize(self.quaternion_wxyz)
        a = self.attitude_rad
        return rotations.euler_to_quat(a.roll, a.pitch, a.yaw)


# ---------- §3 Detection ----------
class BBox(_Msg):
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self):
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bbox 必須是左上 (x1, y1)、右下 (x2, y2)")
        return self


class Detection(_Msg):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, populate_by_name=True)

    track_id: StrictInt = Field(ge=0)
    cls: Literal[CLASSES] = Field(alias="class")  # class 是 Python 保留字
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox


class DetectionFrame(_Msg):
    schema_version: str
    frame_id: StrictInt = Field(ge=0)
    timestamp_ns: StrictInt
    image_width: StrictInt = Field(ge=1)
    image_height: StrictInt = Field(ge=1)
    detections: List[Detection]
