"""確定性隊形規劃器：Plan 的「角色」→ 每架的目標點（world_ned），約 10 Hz（規格 §10.6）。

LLM 只說「uav2 當左翼攔截」，「左翼攔截要站在哪個座標」由這裡算，理由：
    - 每 0.1 秒都要更新（目標一直在動），LLM 太慢也太貴
    - 幾何計算必須可預測、可測試，不能每次問都不一樣

角色位置（以目標行進方向為前方，半徑 r）：
    chaser          正後方（180°）
    blocker_left    左前方（-60°）
    blocker_right   右前方（+60°）    → 三架平均分佈在 120° 間隔上，形成包圍
    overwatch       目標正上方，高度 + overwatch_extra_alt_m
    searcher        以最後已知位置為中心繞圈，多架時相位錯開
    idle            原地不動
    herder / capturer / blocker   捕捉策略（mode = capture）的驅趕 / 抓拍 / 封鎖，幾何見 planner/capture.py
encircle / track 時，目標位置用「現在位置 + 速度 × lead_s」，讓攔截點提前，不會永遠追在後面。
高度依 drone_id 分層（A 的安全層最後把關，這裡先避免垂直衝突）。
送出前再做水平去衝突（planner/deconflict.py，防撞第二層）：任兩架目標點水平 ≥ min_goal_separation_m，
優先權低的角色讓開；返航不做（各機回自己的出生點，由 A 的返航流程負責）。
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from llm_agent.fleet_schema import CAPTURE_ONLY_ROLES, PLAN_SCHEMA_VERSION, Plan

from .capture import CaptureConfig, CaptureGeometry, CaptureZone
from .deconflict import Adjustment, deconflict_goals, yield_crossings

ROLE_ANGLES = {"chaser": math.pi, "blocker_left": -math.pi / 3, "blocker_right": math.pi / 3}


@dataclass(frozen=True)
class FormationConfig:
    altitude_layers_m: Dict[str, float] = field(default_factory=lambda: {"uav1": 25.0, "uav2": 30.0, "uav3": 35.0})
    default_altitude_m: float = 30.0
    lead_s: float = 1.0  # 攔截提前量
    track_extra_radius_factor: float = 1.6  # track 模式隊形放大，先保持距離跟上
    overwatch_extra_alt_m: float = 10.0
    search_angular_speed_rps: float = 0.25
    search_growth_mps: float = 2.0  # 搜尋圈半徑每秒擴大多少（找不到就越找越大）
    search_max_radius_m: float = 60.0
    initial_search_center: tuple = (0.0, 0.0)  # 還沒看過目標時的搜尋中心（場地中心或操作員給的大致位置）
    max_speed_mps: float = 8.0  # 建議值；A 的速度上限優先
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    min_goal_separation_m: float = 6.0  # 防撞第二層；0 = 關閉（只用來做前後對照）
    yield_margin_m: float = 2.0  # 預測交錯距離 < 6 + 2 m 就讓路（取捨見 planner/deconflict.py）
    yield_horizon_s: float = 6.0  # 預測時間上限
    yield_lookahead_m: float = 20.0  # 前瞻距離：預測時間 = clamp(20 m / 速度上限, 2, 6) s


class FormationPlanner:
    def __init__(self, cfg: FormationConfig = FormationConfig(), home: Optional[Dict[str, tuple]] = None):
        self.cfg = cfg
        self.home = home or {}
        self._heading: Optional[float] = None
        self._search_center: Optional[tuple] = None
        self._search_since_ns: Optional[int] = None
        self.capture = CaptureGeometry(cfg.capture)
        self.last_adjustments: List[Adjustment] = []  # 最近一次去衝突改了哪些目標點（之後進動作日誌）
        self.adjustment_count = 0  # 累計被去衝突修改的目標點數

    @property
    def capture_zone(self) -> Optional[CaptureZone]:
        """目前 capture 的捕捉區（沒有在 capture 時為 None），畫面與指標用。"""
        return self.capture.zone

    def altitude(self, drone_id: str) -> float:
        return self.cfg.altitude_layers_m.get(drone_id, self.cfg.default_altitude_m)

    def _target_heading(self, tgt: dict, drones: Sequence) -> float:
        v = tgt["velocity_world_ned_mps"]
        p = tgt["position_world_ned_m"]
        if math.hypot(v["n"], v["e"]) > 0.5:
            self._heading = math.atan2(v["e"], v["n"])
        elif self._heading is None:  # 目標靜止且沒有歷史：以「機群 → 目標」方向當前方
            cn, ce = sum(d.n for d in drones) / len(drones), sum(d.e for d in drones) / len(drones)
            self._heading = math.atan2(p["e"] - ce, p["n"] - cn)
        return self._heading

    def goal_points(self, plan: Plan, plan_id: str, drones: Sequence, targets: Sequence[dict], t_ns: int) -> List[dict]:
        tgt = next((t for t in targets if t["target_id"] == plan.target_id), None)
        if tgt is not None and tgt["status"] != "lost":
            p = tgt["position_world_ned_m"]
            self._search_center = (p["n"], p["e"])
        searching = plan.mode == "search" or tgt is None or tgt["status"] == "lost"
        if searching and self._search_since_ns is None:
            self._search_since_ns = t_ns
        elif not searching:
            self._search_since_ns = None
        if plan.mode == "capture" and tgt is not None and tgt["status"] != "lost":
            self.capture.ensure_zone(tgt, drones, self._target_heading(tgt, drones))
        elif plan.mode != "capture":
            self.capture.reset()
        roles = {a.drone_id: a.role for a in plan.assignments}
        out = []
        for k, d in enumerate(drones):
            role = roles.get(d.drone_id, "idle")
            n, e, alt = self._goal_for(plan, role, d, tgt, drones, t_ns, k, len(drones))
            yaw = None
            if tgt is not None:
                tp = tgt["position_world_ned_m"]
                if math.hypot(tp["n"] - d.n, tp["e"] - d.e) > 1.0:
                    yaw = math.atan2(tp["e"] - d.e, tp["n"] - d.n)
            gp = {
                "schema_version": PLAN_SCHEMA_VERSION,
                "plan_id": plan_id,
                "drone_id": d.drone_id,
                "timestamp_ns": int(t_ns),
                "position_world_ned_m": {"n": n, "e": e, "d": -alt},
                "max_speed_mps": self.cfg.max_speed_mps,
            }
            if yaw is not None:
                gp["yaw_rad"] = yaw
            out.append(gp)
        if plan.mode == "return_home":
            self.last_adjustments = []
            return out
        out, adj = deconflict_goals(out, roles, self.cfg.min_goal_separation_m)
        # 預測時間用「前瞻距離 / 速度上限」換算：速度快時看太遠會有很多假警報、一直讓路（實測 8 m/s 時視野內比例掉 9%）
        horizon = max(2.0, min(self.cfg.yield_horizon_s, self.cfg.yield_lookahead_m / self.cfg.max_speed_mps))
        out, adj2 = yield_crossings(out, drones, roles, self.cfg.min_goal_separation_m, horizon,
                                    vmax=self.cfg.max_speed_mps, margin_m=self.cfg.yield_margin_m)
        self.last_adjustments = adj + adj2
        self.adjustment_count += len(self.last_adjustments)
        return out

    def _goal_for(self, plan, role, d, tgt, drones, t_ns, k, count):
        alt = self.altitude(d.drone_id)
        if plan.mode == "hold" or role == "idle":
            return d.n, d.e, alt
        if plan.mode == "return_home":
            hn, he = self.home.get(d.drone_id, (d.n, d.e))
            return hn, he, alt
        r = plan.formation.radius_m
        if plan.mode == "search" or role == "searcher" or tgt is None or tgt["status"] == "lost":
            cn, ce = self._search_center or self.cfg.initial_search_center
            elapsed = (t_ns - self._search_since_ns) / 1e9 if self._search_since_ns is not None else 0.0
            rr = min(self.cfg.search_max_radius_m, r + self.cfg.search_growth_mps * elapsed)
            ang = self.cfg.search_angular_speed_rps * t_ns / 1e9 + 2 * math.pi * k / count
            return cn + rr * math.cos(ang), ce + rr * math.sin(ang), alt
        p, v = tgt["position_world_ned_m"], tgt["velocity_world_ned_mps"]
        pn, pe = p["n"] + v["n"] * self.cfg.lead_s, p["e"] + v["e"] * self.cfg.lead_s
        if role == "overwatch":
            return pn, pe, alt + self.cfg.overwatch_extra_alt_m
        if plan.mode == "capture" and role in CAPTURE_ONLY_ROLES and self.capture.zone is not None:
            n, e = self.capture.goal(role, tgt, d)
            return n, e, alt
        heading = self._target_heading(tgt, drones)
        radius = r * (self.cfg.track_extra_radius_factor if plan.mode == "track" else 1.0)
        a = heading + ROLE_ANGLES.get(role, math.pi)
        return pn + radius * math.cos(a), pe + radius * math.sin(a), alt
