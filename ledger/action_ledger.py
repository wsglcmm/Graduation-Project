"""動作日誌與驗證（action ledger）：證明「Agent 的動作確實對應到無人機的操作指令，而且有執行到」。

老師 2026-11-16 的建議 4：動作要記錄下來、可以被驗證，且可對應到實際的無人機操作指令。

一筆「動作」= 一架無人機在一段時間內的任務（mode、role、target 不變）。Agent 每 2 秒重新規劃，
但角色沒變時不開新動作，只把 plan_id 接在後面，否則每筆動作都還沒驗證就被下一筆取代。
多 Agent 模式（per_drone=True）：一筆動作 = 一架的 Agent 選的一個動作（例如 uav3 的抓拍 Agent 選 engage），
協調 Agent 的計畫只負責結束角色改變的動作，新動作由各機 Agent 的決策開啟。

每筆動作的驗證鏈（全部可匯出查核）：
    1. 白名單驗證       Agent 輸出是否通過 schema（不合法 → rejected，不送出）
    2. 對應飛控指令     送了幾個 GoalPoint、最後的目標點；C 去衝突改了幾次
    3. A 的回覆         accepted / adjusted / rejected 次數與原因；A 實際呼叫的 MAVSDK 函式（flight_command）
    4. 實際是否達成     無人機位置與目前的目標點比對，高度差 ≤ 1 m，水平誤差 ≤ 容許值：
                          目標點不動（懸停、伏擊點、返航）：1.5 m，和 A 的 move_to 相同（規格 §10.8）
                          目標點在動（驅趕、封鎖、搜尋、迎上去抓拍）：1.5 m + 目標點速度 × 1.0 s，
                          而且要連續維持 2 s（只擦過一下不算）
                        目標點在動時，無人機一定落後一點（位置控制的延遲），只用 1.5 m 會把「有照指令飛、
                        只是跟在後面」誤判成失敗（實測：抓拍機已經抓拍成功，動作卻判未達成）；
                        但容許值放太寬，驗證就永遠通過、沒有意義，所以加上「連續 2 s」並把落後時間定在 1 s。
                        到達後繼續統計追蹤誤差（平均、p95），對應 A 的 follow / encircle 結算方式
                        搜尋（繞圈）例外：搜尋圈上的目標點是「掃描用的參考點」，角速度固定、半徑還會擴大，
                        移動得比無人機快，本來就追不到（11/23 A 的真實 SITL：通過率 50% 裡 2/3 的 failed 都是它）。
                        所以搜尋改判「有沒有飛到搜尋圈上」：誤差 = |到圓心距離 − 半徑|，容許值同目標點在動
                        （速度 = 半徑擴大的速度），期限也照到搜尋圈的距離算

另外記錄「指令一致率」：每一步，無人機是不是正朝著目前的目標點飛（速度方向和「到目標點的方向」
夾角 < 45°，或已經在容許值內）。兩層分開看才不會混在一起：
    指令一致率高、到達失敗   → 無人機有照指令飛，是目標點本身追不上（規劃的問題，例如目標點移動得和無人機一樣快）
    指令一致率低             → 無人機沒有照指令飛（被 A 修改、讓路、安全層介入，或飛控本身的問題）

結果使用和 A 的 ActionResult 相同的詞彙與意思（規格 §10.8、§10.11；11/23 依 A 的定義對齊），
兩邊的「動作驗證通過率」才可以比較：
    succeeded   在期限內到達目標點
    failed      超過期限還沒到達
    rejected    白名單驗證失敗（C 不送出），或 A 拒絕了所有目標點
    cancelled   還沒驗證完就被新的動作 / 計畫取代（superseded = True，面板顯示「已改派」）、
                被操作員指令中斷（hold / return_home）、計畫在套用前就被取代，或 Demo 結束時仍在進行
    overridden  A 的安全層介入接手（C 自己不產生，保留給 A 的 ActionResult）
cancelled / overridden 都不列入通過率：還沒有機會驗證完。

期限 = max(min_deadline_s, 初始距離 / 速度上限 × deadline_factor + deadline_margin_s)。
座標：world_ned，m；時間：Demo 引擎的時鐘（s）。
"""
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

RESULTS = ("succeeded", "failed", "rejected", "overridden", "cancelled")
RESULT_ZH = {"succeeded": "通過", "failed": "未達成", "rejected": "被拒絕", "overridden": "A 安全層接手", "cancelled": "已取消",
             None: "進行中"}
OPERATOR_STOP_MODES = ("hold", "return_home")


@dataclass(frozen=True)
class VerifyConfig:
    reach_h_m: float = 1.5  # 同 A 的 move_to（規格 §10.8）
    reach_v_m: float = 1.0
    lag_s: float = 1.0  # 目標點在動時，容許的落後時間：容許值 = reach_h_m + 目標點速度 × lag_s
    hold_s: float = 2.0  # 目標點在動時，要連續在容許值內多久才算跟上
    moving_goal_mps: float = 0.3  # 目標點速度超過這個值才算「在動」
    jump_m: float = 3.0  # 一步內跳超過這個距離（讓路橫移、換角色）視為「換了目標點」，不算進速度
    consistent_cos: float = 0.7071  # 指令一致：速度方向與目標方向夾角 < 45°
    min_move_mps: float = 0.5  # 速度低於這個值時無法判斷方向（剛起步或懸停），不列入一致率
    min_deadline_s: float = 8.0
    deadline_factor: float = 1.5
    deadline_margin_s: float = 5.0
    max_speed_mps: float = 8.0
    reject_after_s: float = 1.0  # A 連續拒絕超過這麼久、而且一次都沒接受過 → rejected


@dataclass
class ActionRecord:
    action_id: str
    drone_id: Optional[str]
    role: Optional[str]
    mode: Optional[str]
    target_id: Optional[int]
    plan_ids: List[str]
    planner: str  # llm / rule
    trigger: str
    operator_command: Optional[str]
    reason: Optional[str]
    latency_ms: float
    model: Optional[str]
    valid: bool
    validation_error: Optional[str]
    t_start_s: float
    agent_id: Optional[str] = None  # 多 Agent：哪一架的 Agent 做的決策（agent_uav1…）；單一指揮官為 coordinator
    action: Optional[str] = None  # 多 Agent：各機 Agent 選的動作（engage、push…）；單一指揮官為 None
    decision_id: Optional[str] = None
    deadline_s: Optional[float] = None
    t_end_s: Optional[float] = None
    # 2. 對應飛控指令
    goals_sent: int = 0
    last_goal_ned: Optional[List[float]] = None
    goal_speed_mps: float = 0.0  # 目標點移動速度（平滑後），決定到達的容許值
    search_ring: Optional[List[float]] = None  # 搜尋時：[圓心 n, 圓心 e, 半徑]，到達改判「飛到搜尋圈上」
    c_adjustments: int = 0
    # 3. A 的回覆
    acks: int = 0
    accepted: int = 0
    adjusted: int = 0
    rejected: int = 0
    last_ack_reason: Optional[str] = None
    flight_command: Optional[dict] = None
    # 4. 實際是否達成
    reached_at_s: Optional[float] = None
    time_to_reach_s: Optional[float] = None
    final_err_h_m: Optional[float] = None
    final_err_v_m: Optional[float] = None
    track_err_mean_m: Optional[float] = None
    track_err_p95_m: Optional[float] = None
    consistent_steps: int = 0  # 指令一致的步數
    judged_steps: int = 0  # 可以判斷的步數
    result: Optional[str] = None
    why: Optional[str] = None
    superseded: bool = False  # cancelled 的原因是被新的動作 / 計畫取代（Agent 改派）
    _errs: List[float] = field(default_factory=list, repr=False)
    _first_reject_s: Optional[float] = field(default=None, repr=False)
    _last_goal_t: Optional[float] = field(default=None, repr=False)
    _ring_t: Optional[float] = field(default=None, repr=False)
    _within_since: Optional[float] = field(default=None, repr=False)

    @property
    def command_consistency(self) -> Optional[float]:
        """指令一致率：無人機朝目標點飛（或已在容許值內）的時間比例。"""
        return round(self.consistent_steps / self.judged_steps, 3) if self.judged_steps else None

    @property
    def status_zh(self) -> str:
        """面板用的狀態：通過 / 被 A 修改 / 被拒絕 / 未達成 / 已改派 / 已取消 / 進行中。"""
        if self.result == "succeeded" and self.adjusted:
            return "被 A 修改"
        if self.result == "cancelled" and self.superseded:
            return "已改派"
        return RESULT_ZH[self.result]

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["status_zh"] = self.status_zh
        d["command_consistency"] = self.command_consistency
        return d


class ActionLedger:
    def __init__(self, cfg: VerifyConfig = VerifyConfig(), per_drone: bool = False):
        self.cfg = cfg
        self.per_drone = per_drone
        self._plan_ctx: Dict[str, dict] = {}  # 多 Agent：drone_id → 目前協調 Agent 分派的角色與計畫資訊
        self.records: List[ActionRecord] = []
        self.active: Dict[str, ActionRecord] = {}  # drone_id → 進行中的動作
        self._by_plan: Dict[tuple, ActionRecord] = {}  # (plan_id, drone_id) → 動作（遲到的 ack 用）
        self._n = 0
        self.newly_resolved: List[ActionRecord] = []  # 最近一次 update 新產生結果的動作（面板顯示用）
        self._prev_pos: Dict[str, tuple] = {}  # drone_id → (n, e, t_s)，估速度方向用

    # ---------- 1. Agent 輸出 ----------
    def on_event(self, ev, t_s: float) -> None:
        """AgentEvent（llm_agent.commander）。不合法 → 記一筆 rejected；合法且已套用 → 開新動作或接續。"""
        if not ev.valid:
            self._add(ActionRecord(self._next_id("x"), None, None, None, None, [], ev.planner, ev.trigger, ev.operator_command,
                                   None, ev.latency_ms, ev.model, False, ev.error, t_s, t_end_s=t_s, result="rejected",
                                   why=f"白名單驗證失敗，未送出：{ev.error}"))
            return
        plan = ev.plan
        if ev.plan_id is None:  # 合法，但在套用前就被操作員的新指令取代
            self._add(ActionRecord(self._next_id("x"), None, None, plan["mode"], plan.get("target_id"), [], ev.planner,
                                   ev.trigger, ev.operator_command, plan["reason"], ev.latency_ms, ev.model, True, None,
                                   t_s, t_end_s=t_s, result="cancelled", why="套用前就被操作員的新指令取代"))
            return
        stop = plan["mode"] in OPERATOR_STOP_MODES and ev.trigger == "operator"
        if self.per_drone:
            self._on_plan_per_drone(ev, plan, stop, t_s)
            return
        for a in plan["assignments"]:
            did = a["drone_id"]
            key = (plan["mode"], a["role"], plan.get("target_id"))
            cur = self.active.get(did)
            if cur is not None and (cur.mode, cur.role, cur.target_id) == key:
                cur.plan_ids.append(ev.plan_id)  # 角色沒變：同一筆動作繼續
                self._by_plan[(ev.plan_id, did)] = cur
                continue
            if cur is not None:
                self._close(cur, t_s, "cancelled", "操作員指令中斷" if stop else f"Agent 改派為 {plan['mode']} / {a['role']}",
                            superseded=not stop)
            rec = ActionRecord(self._next_id(did), did, a["role"], plan["mode"], plan.get("target_id"), [ev.plan_id],
                               ev.planner, ev.trigger, ev.operator_command, plan["reason"], ev.latency_ms, ev.model, True,
                               None, t_s, agent_id="coordinator")
            self._add(rec)
            self.active[did] = rec
            self._by_plan[(ev.plan_id, did)] = rec

    def _on_plan_per_drone(self, ev, plan: dict, stop: bool, t_s: float) -> None:
        """多 Agent：協調 Agent 的計畫只結束「角色改變」的動作；新動作等該架的 Agent 決定。"""
        for a in plan["assignments"]:
            did = a["drone_id"]
            ctx = {"plan_id": ev.plan_id, "mode": plan["mode"], "role": a["role"], "target_id": plan.get("target_id"),
                   "planner": ev.planner, "trigger": ev.trigger, "operator_command": ev.operator_command,
                   "reason": plan["reason"], "latency_ms": ev.latency_ms, "model": ev.model, "stop": stop}
            self._plan_ctx[did] = ctx
            cur = self.active.get(did)
            if cur is not None and (cur.mode, cur.role, cur.target_id) == (ctx["mode"], ctx["role"], ctx["target_id"]):
                cur.plan_ids.append(ev.plan_id)
                self._by_plan[(ev.plan_id, did)] = cur
            elif cur is not None:
                self._close(cur, t_s, "cancelled",
                            "操作員指令中斷" if stop else f"協調 Agent 改派為 {plan['mode']} / {a['role']}", superseded=not stop)

    def on_drone_event(self, dev, t_s: float) -> None:
        """各機 Agent 的決策（llm_agent.drone_agent.DroneAgentEvent）。不合法 → rejected；動作改變 → 開新動作。"""
        ctx = self._plan_ctx.get(dev.drone_id, {})
        if not dev.valid:
            self._add(ActionRecord(self._next_id(dev.drone_id), dev.drone_id, dev.role, dev.mode, ctx.get("target_id"), [],
                                   dev.planner, dev.trigger, None, None, dev.latency_ms, dev.model, False, dev.error, t_s,
                                   agent_id=dev.agent_id, decision_id=dev.decision_id, t_end_s=t_s, result="rejected",
                                   why=f"角色白名單驗證失敗，未送出（沿用上一個動作）：{dev.error}"))
            return
        cur = self.active.get(dev.drone_id)
        key = (dev.mode, dev.role, ctx.get("target_id"), dev.action)
        if cur is not None and (cur.mode, cur.role, cur.target_id, cur.action) == key:
            return
        if cur is not None:
            self._close(cur, t_s, "cancelled", f"{dev.agent_id} 改為 {dev.action}", superseded=True)
        rec = ActionRecord(self._next_id(dev.drone_id), dev.drone_id, dev.role, dev.mode, ctx.get("target_id"),
                           [ctx["plan_id"]] if ctx.get("plan_id") else [], dev.planner, dev.trigger,
                           ctx.get("operator_command"), dev.reason, dev.latency_ms, dev.model, True, None, t_s,
                           agent_id=dev.agent_id, action=dev.action, decision_id=dev.decision_id)
        self._add(rec)
        self.active[dev.drone_id] = rec
        if ctx.get("plan_id"):
            self._by_plan[(ctx["plan_id"], dev.drone_id)] = rec

    # ---------- 2. 對應飛控指令 ----------
    def tag_goals(self, goals: Sequence[dict]) -> None:
        """送出前在 GoalPoint 加上 action_id、agent_id、role（選填欄位，11/23 A 要求）。

        A 的紀錄就能用 action_id 對回 C 的動作日誌：「哪一架的哪個 Agent、什麼角色的動作 → A 實際呼叫的 MAVSDK 函式」，
        口試時有兩份獨立的紀錄可以互相印證。該架目前沒有進行中的動作（例如剛被改派、Agent 還沒決定）就不加。"""
        for g in goals:
            rec = self.active.get(g["drone_id"])
            if rec is None:
                continue
            g["action_id"] = rec.action_id
            if rec.agent_id is not None:
                g["agent_id"] = rec.agent_id
            if rec.role is not None:
                g["role"] = rec.role

    def on_goals(self, goals: Sequence[dict], adjustments: Iterable = (), drones: Sequence = (), t_s: Optional[float] = None,
                 search: Optional[dict] = None) -> None:
        """這一步送出的 GoalPoint（規格 §10.6）與 C 去衝突的修改紀錄（planner.deconflict.Adjustment）。
        search：planner.FormationPlanner.last_search（搜尋圈與正在搜尋的無人機），None = 沒有在搜尋。"""
        pos = {d.drone_id: d for d in drones}
        for g in goals:
            rec = self._by_plan.get((g.get("plan_id"), g["drone_id"]))
            if rec is None or rec.t_end_s is not None:
                continue
            p = g["position_world_ned_m"]
            if search is not None and g["drone_id"] in search["drones"]:
                self._on_search_goal(rec, g, search, pos.get(g["drone_id"]), t_s)
                continue
            rec.search_ring = None
            if rec.last_goal_ned is not None and t_s is not None and rec._last_goal_t is not None and t_s > rec._last_goal_t:
                step = math.hypot(p["n"] - rec.last_goal_ned[0], p["e"] - rec.last_goal_ned[1])
                if step <= self.cfg.jump_m:  # 跳動（讓路橫移、伏擊 → 迎上去）不是「目標點在動」，不算進速度
                    v = min(step / (t_s - rec._last_goal_t), self.cfg.max_speed_mps)
                    rec.goal_speed_mps = round(0.7 * rec.goal_speed_mps + 0.3 * v, 3)  # 平滑
            rec._last_goal_t = t_s
            rec.goals_sent += 1
            rec.last_goal_ned = [round(p["n"], 2), round(p["e"], 2), round(p["d"], 2)]
            if rec.deadline_s is None and g["drone_id"] in pos:
                d = pos[g["drone_id"]]
                dist = math.hypot(p["n"] - d.n, p["e"] - d.e)
                c = self.cfg
                rec.deadline_s = rec.t_start_s + max(c.min_deadline_s, dist / c.max_speed_mps * c.deadline_factor + c.deadline_margin_s)
        for adj in adjustments:
            rec = self.active.get(adj.drone_id)
            if rec is not None:
                rec.c_adjustments += 1

    def _on_search_goal(self, rec: ActionRecord, g: dict, search: dict, d, t_s: Optional[float]) -> None:
        """搜尋：記下搜尋圈；目標點速度改用「半徑擴大的速度」（圓心換了就重新開始算）。"""
        p = g["position_world_ned_m"]
        (cn, ce), r = search["center"], search["radius_m"]
        old = rec.search_ring
        if old is not None and t_s is not None and rec._ring_t is not None and t_s > rec._ring_t:
            if math.hypot(cn - old[0], ce - old[1]) <= self.cfg.jump_m:
                v = min(abs(r - old[2]) / (t_s - rec._ring_t), self.cfg.max_speed_mps)
                rec.goal_speed_mps = round(0.7 * rec.goal_speed_mps + 0.3 * v, 3)
        elif old is None:
            rec.goal_speed_mps = 0.0
        rec.search_ring = [round(cn, 2), round(ce, 2), round(r, 2)]
        rec._ring_t = t_s
        rec._last_goal_t = None  # 之後換回一般目標點時，速度重新估
        rec.goals_sent += 1
        rec.last_goal_ned = [round(p["n"], 2), round(p["e"], 2), round(p["d"], 2)]
        if rec.deadline_s is None and d is not None:
            c = self.cfg
            dist = abs(math.hypot(d.n - cn, d.e - ce) - r)
            rec.deadline_s = rec.t_start_s + max(c.min_deadline_s, dist / c.max_speed_mps * c.deadline_factor + c.deadline_margin_s)

    def horizontal_error_m(self, rec: ActionRecord, d) -> float:
        """到達判定用的水平誤差（m）：一般是到目標點的距離；搜尋時是離搜尋圈多遠。"""
        if rec.search_ring is not None:
            cn, ce, r = rec.search_ring
            return abs(math.hypot(d.n - cn, d.e - ce) - r)
        gn, ge, _ = rec.last_goal_ned
        return math.hypot(d.n - gn, d.e - ge)

    # ---------- 3. A 的回覆 ----------
    def on_acks(self, acks: Iterable, t_s: float) -> None:
        """fleet_link.GoalPointAck（含選填的 flight_command，規格 §10.8）。"""
        for a in acks:
            rec = self._by_plan.get((a.plan_id, a.drone_id)) or self.active.get(a.drone_id)
            if rec is None:
                continue
            rec.acks += 1
            if a.flight_command is not None:
                rec.flight_command = a.flight_command
            if not a.accepted:
                rec.rejected += 1
                rec.last_ack_reason = a.reason
                if rec.accepted == 0 and rec._first_reject_s is None:
                    rec._first_reject_s = t_s
                continue
            rec.accepted += 1
            if a.adjusted:
                rec.adjusted += 1
                rec.last_ack_reason = a.reason

    # ---------- 4. 實際是否達成 ----------
    def on_poses(self, drones: Sequence, t_s: float) -> None:
        """每步呼叫：用實際位姿（world_ned）比對該架目前的目標點。"""
        self.newly_resolved = []
        c = self.cfg
        for d in drones:
            rec = self.active.get(d.drone_id)
            if rec is None or rec.last_goal_ned is None:
                continue
            gn, ge, gd = rec.last_goal_ned
            eh, ev = self.horizontal_error_m(rec, d), abs(-d.alt_m - gd)
            rec.final_err_h_m, rec.final_err_v_m = round(eh, 2), round(ev, 2)
            self._judge_consistency(rec, d, gn, ge, math.hypot(d.n - gn, d.e - ge), t_s)
            if rec.reached_at_s is not None:
                rec._errs.append(eh)
            if rec.result is not None:
                continue
            tol = self.tolerance_m(rec)
            moving = tol > c.reach_h_m
            inside = eh <= tol and ev <= c.reach_v_m
            if inside and rec._within_since is None:
                rec._within_since = t_s
            elif not inside:
                rec._within_since = None
            if inside and (not moving or t_s - rec._within_since >= c.hold_s - 1e-9):
                rec.reached_at_s, rec.time_to_reach_s = t_s, round(t_s - rec.t_start_s, 2)
                if rec.search_ring is not None:
                    kind = f"飛到搜尋圈上（連續 {c.hold_s:g} s）" if moving else "飛到搜尋圈上"
                else:
                    kind = f"連續 {c.hold_s:g} s 跟上移動中的目標點" if moving else "到達目標點"
                self._resolve(rec, "succeeded", f"{kind} {tol:.1f} m 內（{rec.time_to_reach_s:.1f} s）")
            elif rec._first_reject_s is not None and rec.accepted == 0 and t_s - rec._first_reject_s >= c.reject_after_s:
                self._resolve(rec, "rejected", f"A 拒絕：{rec.last_ack_reason}")
            elif rec.deadline_s is not None and t_s > rec.deadline_s:
                where = "搜尋圈" if rec.search_ring is not None else "目標點"
                self._resolve(rec, "failed", f"逾時：期限內沒有到達{where}（還差 {eh:.1f} m）")

    def _judge_consistency(self, rec: ActionRecord, d, gn: float, ge: float, eh: float, t_s: float) -> None:
        prev = self._prev_pos.get(d.drone_id)
        self._prev_pos[d.drone_id] = (d.n, d.e, t_s)
        if eh <= self.tolerance_m(rec):
            rec.judged_steps += 1
            rec.consistent_steps += 1
            return
        if prev is None or t_s <= prev[2]:
            return
        vn, ve = (d.n - prev[0]) / (t_s - prev[2]), (d.e - prev[1]) / (t_s - prev[2])
        sp = math.hypot(vn, ve)
        if sp < self.cfg.min_move_mps:
            return
        rec.judged_steps += 1
        if (vn * (gn - d.n) + ve * (ge - d.e)) / (sp * eh) >= self.cfg.consistent_cos:
            rec.consistent_steps += 1

    def tolerance_m(self, rec: ActionRecord) -> float:
        """到達判定的水平容許值（m）：目標點不動時 1.5 m，在動時加上速度 × 落後時間。"""
        c = self.cfg
        if rec.goal_speed_mps < c.moving_goal_mps:
            return c.reach_h_m
        return c.reach_h_m + min(rec.goal_speed_mps, c.max_speed_mps) * c.lag_s

    def finish(self, t_s: float) -> None:
        """Demo 結束：還沒有結果的動作收尾（沒到期限的不算失敗，記 cancelled）。"""
        for rec in list(self.active.values()):
            self._close(rec, t_s, "cancelled", "Demo 結束時仍在進行")

    # ---------- 匯出與統計 ----------
    def summary(self) -> dict:
        counts = {r: sum(1 for x in self.records if x.result == r) for r in RESULTS}
        counts["pending"] = sum(1 for x in self.records if x.result is None)
        judged = counts["succeeded"] + counts["failed"] + counts["rejected"]
        reach = [x.time_to_reach_s for x in self.records if x.time_to_reach_s is not None]
        cs = sum(x.consistent_steps for x in self.records)
        js = sum(x.judged_steps for x in self.records)
        return {
            "actions": len(self.records),
            "by_result": counts,
            # 動作驗證通過率：有被判定的動作中，通過的比例（cancelled / overridden 不列入，因為還沒機會驗證完）
            "action_pass_rate": counts["succeeded"] / judged if judged else None,
            "adjusted_by_a": sum(1 for x in self.records if x.adjusted),
            "adjusted_by_c": sum(1 for x in self.records if x.c_adjustments),
            "time_to_reach_mean_s": sum(reach) / len(reach) if reach else None,
            # 指令一致率：所有動作合計，無人機朝目標點飛（或已到達）的時間比例
            "command_consistency": cs / js if js else None,
        }

    def write(self, out_dir: Path) -> None:
        """actions.jsonl（完整）與 actions.csv（報告用，Excel 可直接開）。"""
        out_dir = Path(out_dir)
        rows = [r.to_dict() for r in self.records]
        with open(out_dir / "actions.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        cols = ["action_id", "t_start_s", "t_end_s", "drone_id", "agent_id", "role", "action", "mode", "target_id", "planner", "trigger",
                "operator_command", "valid", "goals_sent", "c_adjustments", "acks", "accepted", "adjusted", "rejected",
                "time_to_reach_s", "track_err_mean_m", "track_err_p95_m", "command_consistency", "result", "superseded", "status_zh",
                "why", "reason"]
        with open(out_dir / "actions.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    # ---------- 內部 ----------
    def _next_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n:04d}"

    def _add(self, rec: ActionRecord) -> None:
        self.records.append(rec)

    def _resolve(self, rec: ActionRecord, result: str, why: str) -> None:
        rec.result, rec.why = result, why
        self.newly_resolved.append(rec)

    def _close(self, rec: ActionRecord, t_s: float, result: str, why: str, superseded: bool = False) -> None:
        rec.t_end_s = t_s
        if rec._errs:
            e = sorted(rec._errs)
            rec.track_err_mean_m = round(sum(e) / len(e), 2)
            rec.track_err_p95_m = round(e[min(len(e) - 1, int(0.95 * len(e)))], 2)
        if rec.result is None:
            rec.result, rec.why, rec.superseded = result, why, superseded
        if self.active.get(rec.drone_id) is rec:
            del self.active[rec.drone_id]
