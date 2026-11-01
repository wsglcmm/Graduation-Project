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
encircle / track 時，目標位置用「現在位置 + 速度 × lead_s」，讓攔截點提前，不會永遠追在後面。
高度依 drone_id 分層（A 的安全層最後把關，這裡先避免垂直衝突）。
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from llm_agent.fleet_schema import PLAN_SCHEMA_VERSION, Plan

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


class FormationPlanner:
    def __init__(self, cfg: FormationConfig = FormationConfig(), home: Optional[Dict[str, tuple]] = None):
        self.cfg = cfg
        self.home = home or {}
        self._heading: Optional[float] = None
        self._search_center: Optional[tuple] = None
        self._search_since_ns: Optional[int] = None

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
        heading = self._target_heading(tgt, drones)
        radius = r * (self.cfg.track_extra_radius_factor if plan.mode == "track" else 1.0)
        a = heading + ROLE_ANGLES.get(role, math.pi)
        return pn + radius * math.cos(a), pe + radius * math.sin(a), alt
