"""各機 Agent：每架無人機一個，依協調 Agent 分派的角色，從角色白名單選「現在要怎麼做」。

規則式（預設）：重現原本寫在規劃器裡的決策（例如抓拍機的迎上去遲滯），所以多 Agent 版的規則式表現和單一指揮官相同。
LLM（選配）：每架各自呼叫 LLM（背景執行，不卡模擬）；角色改變時立刻問，之後每 period_s 秒問一次。
    LLM 還沒回來或失敗時，用規則式動作，不會沒有動作可執行。
    3 架 + 協調 Agent 同時用 LLM 時呼叫量會變 4 倍，免費額度下 period_s 建議 ≥ 6 s。

每一次決策都產生 DroneAgentEvent（面板的即時指令翻譯與動作日誌都用它）。規則式只在動作改變時產生事件。
"""
import itertools
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from planner.capture import CaptureConfig

from .drone_actions import ACTION_ZH, ROLE_ACTIONS, DroneAction, parse_drone_action


@dataclass
class DroneAgentEvent:
    timestamp_ns: int
    drone_id: str
    agent_id: str
    role: str
    mode: str
    trigger: str  # role_assigned / periodic / changed（規則式偵測到情況改變）
    planner: str  # llm / rule
    latency_ms: float
    valid: bool
    action: Optional[str]
    reason: Optional[str]
    error: Optional[str] = None
    model: Optional[str] = None
    decision_id: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class DroneContext:
    """各機 Agent 看得到的資訊（自己的位姿、角色、目標、捕捉區、隊友）。座標 world_ned，m。"""

    drone: object  # fusion.DroneState
    role: str
    mode: str
    target: Optional[dict]  # FusedTarget；None = 沒有可用目標
    zone: Optional[object] = None  # planner.capture.CaptureZone
    teammates: Sequence = ()


def rule_action(ctx: DroneContext, prev: Optional[str], cfg: CaptureConfig = CaptureConfig()) -> DroneAction:
    """規則式動作（和原本規劃器內建的決策相同）。"""
    role, t = ctx.role, ctx.target
    live = t is not None and t.get("status") != "lost"
    if role == "idle":
        return DroneAction(action="hold", reason="待命")
    if role == "searcher":
        return DroneAction(action="orbit", reason="還沒有目標，繞圈搜尋")
    if role == "overwatch":
        return DroneAction(action="overwatch", reason="在目標上方高處監看")
    if role in ("chaser", "blocker_left", "blocker_right"):
        return DroneAction(action="follow_slot", reason="維持隊形位置")
    if role == "blocker":
        return DroneAction(action="flank", reason="守住目標側邊逃跑的方向")
    if not live:
        return DroneAction(action="hold", reason="目標不在視野內，先懸停")
    p = t["position_world_ned_m"]
    if role == "capturer":
        if ctx.zone is None:
            return DroneAction(action="ambush", reason="前往捕捉區伏擊")
        dz = math.hypot(p["n"] - ctx.zone.n, p["e"] - ctx.zone.e)
        if dz <= cfg.engage_radius_m:
            return DroneAction(action="engage", reason=f"目標離捕捉區 {dz:.0f} m，迎上去抓拍")
        if dz > cfg.disengage_radius_m or prev != "engage":
            return DroneAction(action="ambush", reason=f"目標離捕捉區 {dz:.0f} m，在捕捉區等")
        return DroneAction(action="engage", reason=f"目標被嚇退到 {dz:.0f} m，繼續追（未超過 {cfg.disengage_radius_m:.0f} m）")
    if role == "herder":
        if ctx.zone is None:
            return DroneAction(action="push", reason="從目標後方施壓")
        un, ue = p["n"] - ctx.zone.n, p["e"] - ctx.zone.e  # 捕捉區 → 目標
        rn, re = ctx.drone.n - p["n"], ctx.drone.e - p["e"]  # 目標 → 自己
        if un * rn + ue * re < 0:
            return DroneAction(action="circle_behind", reason="自己在目標和捕捉區之間，先繞到後方，避免把目標往反方向趕")
        return DroneAction(action="push", reason="在目標後方，把它往捕捉區趕")
    return DroneAction(action="hold", reason="沒有對應的規則，懸停")


DRONE_SYSTEM_PROMPT = """你是無人機 {drone_id} 的 Agent，目前角色是「{role}」。協調 Agent 已經決定整體策略（{mode}）與每架的角色，你只決定「自己現在要怎麼做」。

你能用的動作（只能從這裡選）：
{actions}

規則：
1. 只輸出一個 JSON 物件：{{"action": "<動作>", "reason": "<一句話理由，60 字以內>"}}，不要輸出座標、速度或其他欄位。
2. 不能用別的角色的動作；不確定時選 hold。
3. 防撞由程式負責，你不用處理。
"""

ROLE_HINTS = {
    "capturer": "抓拍機：目標還遠時在捕捉區伏擊（ambush）；目標接近捕捉區（約 18 m 內）時迎上去（engage），讓目標進入相機畫面中央；"
                "迎上去後目標被嚇退但還沒超過約 30 m，繼續追，不要立刻退回（否則會來回擺盪）。",
    "herder": "驅趕機：要在目標「遠離捕捉區」的那一側施壓（push），目標會往遠離你的方向跑，也就是往捕捉區；"
              "如果你在目標和捕捉區之間，先繞到後方（circle_behind），不要直接衝過去。",
    "blocker": "封鎖機：守住目標往側邊逃的方向（flank）。",
}


class DroneAgent:
    def __init__(self, drone_id: str, llm: Optional[object] = None, period_s: float = 6.0,
                 capture_cfg: CaptureConfig = CaptureConfig(), clock: Callable[[], float] = time.monotonic):
        self.drone_id = drone_id
        self.agent_id = f"agent_{drone_id}"
        self.llm = llm
        self.period_s = period_s
        self.capture_cfg = capture_cfg
        self.clock = clock
        self.current: Optional[DroneAction] = None
        self.role: Optional[str] = None
        self.mode: Optional[str] = None
        self.events: List[DroneAgentEvent] = []
        self._pool = ThreadPoolExecutor(max_workers=1) if llm is not None else None
        self._pending: Optional[tuple] = None  # (future, trigger, start, role, mode)
        self._last_ask_ns: Optional[int] = None
        self._ids = (f"{drone_id}-d{i:04d}" for i in itertools.count(1))

    @property
    def planner_name(self) -> str:
        return "llm" if self.llm is not None else "rule"

    def tick(self, t_ns: int, ctx: DroneContext) -> List[DroneAgentEvent]:
        new: List[DroneAgentEvent] = []
        role_changed = (ctx.role, ctx.mode) != (self.role, self.mode)
        if role_changed:
            self.role, self.mode = ctx.role, ctx.mode
            self.current = None  # 舊角色的動作不能沿用到新角色
            if self._pending is not None:
                self._pending[0].cancel()
                self._pending = None
        if self.llm is None:
            act = rule_action(ctx, self.current.action if self.current else None, self.capture_cfg)
            if role_changed or self.current is None or act.action != self.current.action:
                new.append(self._accept(t_ns, act, "role_assigned" if role_changed else "changed", "rule", 0.0))
            else:
                self.current = act  # 動作沒變，理由可能更新（例如距離）
            return new
        # LLM：收結果
        if self._pending is not None and self._pending[0].done():
            new.extend(self._collect(t_ns, ctx))
        if self._pending is None and (role_changed or self._last_ask_ns is None
                                      or (t_ns - self._last_ask_ns) / 1e9 >= self.period_s):
            self._last_ask_ns = t_ns
            msgs = [{"role": "user", "content": self.state_text(ctx)}]
            self._pending = (self._pool.submit(self.llm.complete, self.system_prompt(ctx), msgs),
                             "role_assigned" if role_changed else "periodic", self.clock(), ctx.role, ctx.mode)
        if self.current is None:  # LLM 還沒回來：先用規則式，不能沒有動作
            act = rule_action(ctx, None, self.capture_cfg)
            ev = self._accept(t_ns, act, "role_assigned", "rule", 0.0)
            ev.error = "等待 LLM，先用規則式"
            new.append(ev)
        return new

    def _collect(self, t_ns: int, ctx: DroneContext) -> List[DroneAgentEvent]:
        fut, trig, start, role, mode = self._pending
        self._pending = None
        latency = (self.clock() - start) * 1000
        try:
            res = parse_drone_action(fut.result(), role)
            model = getattr(self.llm, "last_model", None)
        except Exception as e:  # noqa: BLE001  網路、金鑰、逾時
            res, model = None, None
            err = f"{type(e).__name__}: {e}"[:200]
        else:
            err = res.error
        if res is not None and res.ok:
            return [self._accept(t_ns, res.action, trig, "llm", latency, model)]
        ev = DroneAgentEvent(t_ns, self.drone_id, self.agent_id, role, mode, trig, "llm", latency, False, None, None,
                             error=err, model=model, decision_id=next(self._ids))
        self.events.append(ev)
        return [ev]  # 不合法：沿用目前的動作（self.current 不變）

    def _accept(self, t_ns, act: DroneAction, trig, planner, latency, model=None) -> DroneAgentEvent:
        self.current = act
        ev = DroneAgentEvent(t_ns, self.drone_id, self.agent_id, self.role, self.mode, trig, planner, latency, True,
                             act.action, act.reason, model=model, decision_id=next(self._ids))
        self.events.append(ev)
        return ev

    def system_prompt(self, ctx: DroneContext) -> str:
        acts = "\n".join(f"- {a}：{ACTION_ZH[a]}" for a in ROLE_ACTIONS.get(ctx.role, ("hold",)))
        hint = ROLE_HINTS.get(ctx.role, "")
        return DRONE_SYSTEM_PROMPT.format(drone_id=self.drone_id, role=ctx.role, mode=ctx.mode, actions=acts) + hint

    def state_text(self, ctx: DroneContext) -> str:
        d = ctx.drone
        s = {"me": {"drone_id": d.drone_id, "n": round(d.n, 1), "e": round(d.e, 1), "alt_m": round(d.alt_m, 1)},
             "role": ctx.role, "mode": ctx.mode, "current_action": self.current.action if self.current else None,
             "teammates": [{"drone_id": m.drone_id, "n": round(m.n, 1), "e": round(m.e, 1)} for m in ctx.teammates]}
        t = ctx.target
        if t is not None:
            p, v = t["position_world_ned_m"], t["velocity_world_ned_mps"]
            s["target"] = {"status": t["status"], "n": round(p["n"], 1), "e": round(p["e"], 1),
                           "vn": round(v["n"], 1), "ve": round(v["e"], 1),
                           "distance_to_me_m": round(math.hypot(p["n"] - d.n, p["e"] - d.e), 1)}
            if ctx.zone is not None:
                s["capture_zone"] = {"n": round(ctx.zone.n, 1), "e": round(ctx.zone.e, 1), "radius_m": ctx.zone.radius_m,
                                     "target_to_zone_m": round(math.hypot(p["n"] - ctx.zone.n, p["e"] - ctx.zone.e), 1),
                                     "me_to_zone_m": round(math.hypot(d.n - ctx.zone.n, d.e - ctx.zone.e), 1)}
        return "目前狀態：\n" + json.dumps(s, ensure_ascii=False) + "\n請輸出你的動作 JSON。"

    def wait_idle(self, timeout_s: float = 30.0) -> None:
        if self._pending is not None:
            try:
                self._pending[0].result(timeout=timeout_s)
            except Exception:  # noqa: BLE001  例外在下一次 tick 記成事件
                pass

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
