"""DemoEngine 的機隊後端：把 A 的 3 架 SITL 包成和 sim2d.FleetSim 一樣的介面。

    FleetSim（替身）           SitlFleet（A 的 SITL）
    states()                 → 最新的 FleetPose（world_ned）
    apply_goals(goals, t)    → 送 GoalPoints 給 A（限制在 send_hz 以內）
    step(dt, t)              → 不推進物理（SITL 自己照真實時間跑），只更新最小機間距離
    min_pair_distance_m      → 3D 最小距離（§10.6 安全要求 ≥ 5 m）

所以 Demo 引擎、融合、Agent、隊形規劃都不用改，切換 --fleet sim / sitl 即可。
"""
import math
import time
from typing import Dict, List, Sequence

from fusion import DroneState

from .client import FleetLink, LinkError


class SitlFleet:
    def __init__(self, link: FleetLink, send_hz: float = 10.0, max_speed_mps: float = 3.0):
        """max_speed_mps：A 安全層的速度上限（2026-11-02 為 3 m/s）。隊形規劃的目標點不會要求超過它，
        否則每一則 GoalPoint 都會被 A 標成 adjusted，ack 統計就看不出真正被修改的情況。"""
        self.link = link
        self.max_speed_mps = max_speed_mps
        self.min_interval_s = 1.0 / send_hz
        self._last_send = 0.0
        self.min_pair_distance_m = math.inf

    def states(self) -> List[DroneState]:
        return self.link.states()

    def apply_goals(self, goal_points: Sequence[dict], t_ns: int) -> None:
        now = time.monotonic()
        if now - self._last_send >= self.min_interval_s:
            self._last_send = now
            self.link.send_goals(list(goal_points))

    def step(self, dt: float, t_ns: int) -> None:
        d = self.link.min_pair_distance_m()
        if math.isfinite(d):
            self.min_pair_distance_m = min(self.min_pair_distance_m, d)

    def start_positions(self) -> Dict[str, tuple]:
        """目前各機位置（n, e, alt），給隊形規劃當 home。"""
        return {s.drone_id: (s.n, s.e, s.alt_m) for s in self.states()}

    def ensure_airborne(self, drone_ids: Sequence[str], timeout_s: float = 90.0) -> None:
        """還在地上的話送 takeoff，等全部到高度層（A 實測約 24 s）。"""
        self.link.wait_for_poses(drone_ids)
        landed = [did for did, r in self.link.poses().items() if did in drone_ids and r.pose.status == "landed"]
        if landed:
            print(f"起飛中：{', '.join(sorted(landed))}（約 25 秒）…", flush=True)
            self.link.command("takeoff", timeout_s=timeout_s)
            # takeoff 的回覆可能比下一則 FleetPose 早到（20 Hz），等每架都有回覆之後的新位姿再判斷
            self.link.wait_for_fresh_poses(drone_ids, since_s=time.monotonic())
        bad = [did for did, r in self.link.poses().items() if did in drone_ids and r.pose.status != "flying"]
        if bad:
            raise LinkError(f"這些無人機沒有進入飛行狀態：{bad}")
