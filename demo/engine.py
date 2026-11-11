"""Demo 的核心迴圈（不含畫面），方便測試與無畫面執行。

每一步（預設 0.1 s，10 Hz）：
    1. 兔子移動（之後改成：B 的影片偵測 + 地面標定）
    2. 每架依視野判斷看不看得到，產生帶誤差的觀測       fusion.visibility
    3. 多機觀測融合 → 卡爾曼濾波 → FusedTarget          fusion.fuse / fusion.kalman
    4. 指揮官決定要不要問 Agent（背景執行，不阻塞）      llm_agent.commander
    5. 依目前計畫算各機目標點                           planner.formation
    6. 無人機替身追目標點（之後換成 A 的 3 架 SITL）     sim2d.fleet_sim
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from coord_transform.config import CameraConfig
from coord_transform.uncertainty import ErrorModel
from fusion import TargetTracker, ViewModel, fuse
from llm_agent.commander import Commander
from planner import FormationConfig, FormationPlanner
from sim2d import FleetSim, RabbitConfig, RabbitPath

DEFAULT_START = {"uav1": (-30.0, -20.0, 25.0), "uav2": (-30.0, 0.0, 30.0), "uav3": (-30.0, 20.0, 35.0)}
S = 1_000_000_000


TRACK_STATUSES = ("tracking", "predicted", "lost")


def _rmse(sq: List[float]) -> Optional[float]:
    return math.sqrt(float(np.mean(sq))) if sq else None


@dataclass
class Metrics:
    """報告用的成功指標（規格「可量化的成功指標」）。

    估計誤差依目標狀態分開統計，因為三種狀態的意義不同：
        tracking   至少一架看得到，誤差反映座標轉換 + 融合 + 濾波的品質
        predicted  暫時看不到，用速度外推（< 3 s），誤差會隨時間變大
        lost       超過 3 s 沒看到，位置只是最後估計，不能算成定位誤差
    estimate_rmse_m 只算 tracking + predicted；lost 另外列出，避免懸停、返航時拉高數字。
    """

    steps: int = 0
    visible_steps: int = 0
    sq_err: Dict[str, List[float]] = field(default_factory=lambda: {s: [] for s in TRACK_STATUSES})
    status_steps: Dict[str, int] = field(default_factory=lambda: {**{s: 0 for s in TRACK_STATUSES}, "none": 0})
    encircled_at_s: Optional[float] = None
    encircled_steps: int = 0
    encircle_onsets_s: List[float] = field(default_factory=list)  # 每次從「沒包圍」變成「包圍」的時間
    _was_encircled: bool = False

    def summary(self, commander: Commander, sim: FleetSim) -> dict:
        evs = commander.events
        lat = [e.latency_ms for e in evs if e.planner == "llm"]
        llm_evs = [e for e in evs if e.planner == "llm"]
        return {
            "duration_s": self.steps * 0.1,
            "visible_fraction": self.visible_steps / max(1, self.steps),
            "estimate_rmse_m": _rmse(self.sq_err["tracking"] + self.sq_err["predicted"]),
            "estimate_rmse_by_status_m": {s: _rmse(v) for s, v in self.sq_err.items()},
            "target_status_fraction": {s: n / max(1, self.steps) for s, n in self.status_steps.items()},
            "time_to_encircle_s": self.encircled_at_s,
            "encircled_fraction": self.encircled_steps / max(1, self.steps),
            "encircle_onsets_s": list(self.encircle_onsets_s),
            "plans": len(evs),
            "llm_requests": len(llm_evs),
            "llm_valid_rate": (sum(e.valid for e in llm_evs) / len(llm_evs)) if llm_evs else None,
            "llm_latency_ms_mean": float(np.mean(lat)) if lat else None,
            "llm_latency_ms_p95": float(np.percentile(lat, 95)) if lat else None,
            "min_drone_distance_m": sim.min_pair_distance_m,
        }


def is_encircled(drones, tn: float, te: float, radius: float) -> bool:
    """三架都在 0.5r–1.5r 環帶內，且相鄰方位角間隔都 ≥ 60°（大致包住）。"""
    if len(drones) < 3:
        return False
    angs = []
    for d in drones:
        dist = math.hypot(d.n - tn, d.e - te)
        if not 0.5 * radius <= dist <= 1.5 * radius:
            return False
        angs.append(math.atan2(d.e - te, d.n - tn))
    angs.sort()
    gaps = [b - a for a, b in zip(angs, angs[1:])] + [2 * math.pi - (angs[-1] - angs[0])]
    return min(gaps) >= math.radians(60)


class DemoEngine:
    def __init__(self, cam: CameraConfig, err: ErrorModel, commander: Commander, rabbit: RabbitConfig = RabbitConfig(),
                 start=None, seed: int = 0, dt: float = 0.1, fleet=None):
        """fleet：機隊後端，介面同 sim2d.FleetSim。None = 用 FleetSim 替身；接 A 的 SITL 時傳 fleet_link.SitlFleet。"""
        start = start or (fleet.start_positions() if fleet is not None else DEFAULT_START)
        self.dt, self.t_ns = dt, 0
        self.rng = np.random.default_rng(seed)
        self.view = ViewModel(cam, err)
        self.rabbit = RabbitPath(rabbit)
        self.tracker = TargetTracker(1, "car")
        self.commander = commander
        cap = getattr(fleet, "max_speed_mps", None)  # A 的速度上限（SitlFleet 提供）；目標點不要求超過它
        cfg = FormationConfig(max_speed_mps=cap) if cap else FormationConfig()
        self.planner = FormationPlanner(home={k: (v[0], v[1]) for k, v in start.items()}, cfg=cfg)
        self.sim = fleet if fleet is not None else FleetSim(start)
        self.metrics = Metrics()
        self.target_state: Optional[dict] = None
        self.goals: List[dict] = []
        self.truth = self.rabbit.n, self.rabbit.e
        self.observations = []

    def step(self) -> list:
        """前進一步，回傳這一步新的 AgentEvent。"""
        self.t_ns += int(self.dt * S)
        self.truth = self.rabbit.step(self.dt)
        drones = self.sim.states()
        self.observations = self.view.observe(drones, 1, *self.truth, self.rng)
        fused = fuse(self.observations)
        self.target_state = self.tracker.step(self.t_ns, fused[0] if fused else None)
        targets = [self.target_state] if self.target_state else []
        events = self.commander.tick(self.t_ns, drones, targets)
        if self.commander.current is not None:
            self.goals = self.planner.goal_points(self.commander.current, self.commander.current_id, drones, targets, self.t_ns)
            self.sim.apply_goals(self.goals, self.t_ns)
        self.sim.step(self.dt, self.t_ns)
        self._update_metrics()
        return events

    def _update_metrics(self) -> None:
        m = self.metrics
        m.steps += 1
        if self.observations:
            m.visible_steps += 1
        if self.target_state:
            p = self.target_state["position_world_ned_m"]
            st = self.target_state["status"]
            m.sq_err[st].append((p["n"] - self.truth[0]) ** 2 + (p["e"] - self.truth[1]) ** 2)
            m.status_steps[st] += 1
        else:
            m.status_steps["none"] += 1
        plan = self.commander.current
        r = plan.formation.radius_m if plan else 15.0
        now = is_encircled(self.sim.states(), *self.truth, r)
        if now:
            m.encircled_steps += 1
            if m.encircled_at_s is None:
                m.encircled_at_s = self.t_ns / S
            if not m._was_encircled:
                m.encircle_onsets_s.append(self.t_ns / S)
        m._was_encircled = now

    def summary(self) -> dict:
        return self.metrics.summary(self.commander, self.sim)
