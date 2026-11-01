"""多機 Agent 計畫 Plan 的 schema 與驗證（規格 §10.5）。

信任邊界與單機版（actions.py）相同：
    - LLM 只輸出 {mode, target_id, assignments, formation, reason}，不輸出任何座標或速度。
    - 格式檢查用 pydantic strict（型別、範圍、不得有多餘欄位）；
      語意檢查（每架恰好一次、目標存在）需要「目前世界狀態」，所以另外寫在 validate_plan。
    - plan_id、時間戳、延遲由程式加上（commander.py），LLM 無法偽造。
"""
import json
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

PLAN_SCHEMA_VERSION = "0.2.0"
MODES = ("search", "track", "encircle", "hold", "return_home")
ROLES = ("chaser", "blocker_left", "blocker_right", "overwatch", "searcher", "idle")
MODES_NEED_TARGET = ("track", "encircle")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Assignment(_Strict):
    drone_id: str = Field(min_length=1, max_length=16)
    role: Literal[ROLES]


class Formation(_Strict):
    radius_m: float = Field(default=15.0, ge=5.0, le=60.0, allow_inf_nan=False)


class Plan(_Strict):
    mode: Literal[MODES]
    target_id: Optional[StrictInt] = Field(default=None, ge=0)
    assignments: List[Assignment] = Field(min_length=1, max_length=8)
    formation: Formation = Formation()
    reason: str = Field(min_length=1, max_length=80)


@dataclass(frozen=True)
class PlanResult:
    ok: bool
    plan: Optional[Plan] = None
    error: Optional[str] = None


def validate_plan(plan: Plan, drone_ids: Sequence[str], target_ids: Sequence[int]) -> Optional[str]:
    """語意檢查。回傳 None 代表通過，否則回傳錯誤訊息（可餵回 LLM 重試）。"""
    got = [a.drone_id for a in plan.assignments]
    if sorted(got) != sorted(drone_ids):
        return f"assignments 必須讓每架無人機恰好出現一次：應為 {sorted(drone_ids)}，收到 {got}"
    if plan.mode in MODES_NEED_TARGET:
        if plan.target_id is None:
            return f"mode={plan.mode} 必須指定 target_id"
        if plan.target_id not in target_ids:
            return f"target_id={plan.target_id} 不存在，目前的目標：{list(target_ids)}"
    return None


def _errors(e: ValidationError) -> str:
    return "；".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors())


def parse_plan(text: str, drone_ids: Sequence[str], target_ids: Sequence[int]) -> PlanResult:
    """從 LLM 輸出取出唯一一個 JSON 物件，做格式 + 語意驗證。容忍前後說明文字與 ``` 圍欄。"""
    if not isinstance(text, str) or not text.strip():
        return PlanResult(False, error="輸出為空")
    start = text.find("{")
    if start < 0:
        return PlanResult(False, error="找不到 JSON 物件")
    try:
        obj, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as e:
        return PlanResult(False, error=f"JSON 格式錯誤：{e.msg}")
    if "{" in text[end:]:
        return PlanResult(False, error="輸出包含多個 JSON 物件，一次只能輸出一個計畫")
    if not isinstance(obj, dict):
        return PlanResult(False, error="輸出必須是 JSON 物件")
    try:
        plan = Plan.model_validate(obj)
    except ValidationError as e:
        return PlanResult(False, error=f"不符合 Plan 格式：{_errors(e)}")
    err = validate_plan(plan, drone_ids, target_ids)
    return PlanResult(False, error=err) if err else PlanResult(True, plan=plan)
