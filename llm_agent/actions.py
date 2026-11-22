"""LLM Agent 的動作白名單與驗證（規格 docs/interface_spec.md §7）。

信任邊界（刻意的設計）：
    - LLM 只輸出 {"action": ..., "params": {...}}。
    - schema_version 與 request_id 由程式在驗證通過後才加上（build_request），LLM 無法偽造。
    - 這裡只檢查「格式與基本合理性」（型別、必填、> 0、不得有多餘欄位）。
      速度與高度上限、地理圍欄、電量保護等安全限制由 A 的飛控層最終負責，這裡不重複實作也不放寬。
    - 一律 strict：字串 "15" 不會被悄悄轉成數字，布林值不會被當成整數，NaN / Infinity 一律拒絕。
"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

SCHEMA_VERSION = "0.1.0"

PosFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class NedPoint2D(_Strict):
    """水平面上的 NED 點（m）。⚠️ 規格 §7 只寫 center_ned_m，未定義欄位；此處暫定 {n, e}，需併入規格。"""

    n: FiniteFloat
    e: FiniteFloat


class FollowParams(_Strict):
    track_id: int = Field(ge=0)
    distance_m: PosFloat
    altitude_m: PosFloat


class SearchParams(_Strict):
    pattern: Literal["orbit", "lawnmower"]
    center_ned_m: NedPoint2D
    radius_m: PosFloat


class SetAltitudeParams(_Strict):
    altitude_m: PosFloat


class NoParams(_Strict):
    pass


class FollowAction(_Strict):
    action: Literal["follow"]
    params: FollowParams


class HoverAction(_Strict):
    action: Literal["hover"]
    params: NoParams = NoParams()


class SearchAction(_Strict):
    action: Literal["search"]
    params: SearchParams


class SetAltitudeAction(_Strict):
    action: Literal["set_altitude"]
    params: SetAltitudeParams


class ReturnHomeAction(_Strict):
    action: Literal["return_home"]
    params: NoParams = NoParams()


LlmAction = Annotated[
    Union[FollowAction, HoverAction, SearchAction, SetAltitudeAction, ReturnHomeAction],
    Field(discriminator="action"),
]
ACTION_ADAPTER = TypeAdapter(LlmAction)
ACTION_NAMES = ("follow", "hover", "search", "set_altitude", "return_home")


def build_request(action, request_id: str) -> dict:
    """驗證通過的動作 → 規格 §7 的完整請求（由程式補上 schema_version 與 request_id，不來自 LLM）。"""
    if not request_id:
        raise ValueError("request_id 不可為空")
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "action": action.action,
        "params": action.params.model_dump(),
    }
