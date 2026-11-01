"""多機運動學替身：接收各機目標點（§10.6），輸出各機位姿（§10.2）。

不是飛控模擬（沒有姿態動力學、風、EKF），只模擬 A 的安全層「看得見的行為」：
    - 速度、加速度上限
    - 超過 goal_timeout_s 沒收到新目標點 → 原地懸停（§10.6 的 1 秒逾時）
    - 最小機間水平距離：太近時加入互斥速度（真正生效的安全層在 A，這裡只為了讓 Demo 行為合理）
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from fusion import DroneState


@dataclass(frozen=True)
class FleetSimConfig:
    max_speed_mps: float = 8.0
    max_accel_mps2: float = 4.0
    max_climb_mps: float = 3.0
    max_yaw_rate_rps: float = math.radians(90)
    position_gain: float = 0.8  # 1/s：離目標點 1 m → 期望速度 0.8 m/s
    goal_timeout_s: float = 1.0
    min_separation_m: float = 5.0
    separation_gain: float = 2.0


@dataclass
class _Drone:
    drone_id: str
    n: float
    e: float
    alt: float
    yaw: float = 0.0
    vn: float = 0.0
    ve: float = 0.0
    goal: Optional[dict] = None
    goal_ns: Optional[int] = None


class FleetSim:
    def __init__(self, start: Dict[str, tuple], cfg: FleetSimConfig = FleetSimConfig(), start_alt_m: float = 30.0):
        """start：drone_id → (n, e) 或 (n, e, alt)。"""
        self.cfg = cfg
        self.drones: List[_Drone] = []
        for did, pos in start.items():
            alt = pos[2] if len(pos) > 2 else start_alt_m
            self.drones.append(_Drone(did, float(pos[0]), float(pos[1]), float(alt)))
        self.min_pair_distance_m = math.inf

    def apply_goals(self, goal_points: Sequence[dict], t_ns: int) -> None:
        by_id = {d.drone_id: d for d in self.drones}
        for g in goal_points:
            d = by_id.get(g["drone_id"])
            if d is not None:
                d.goal, d.goal_ns = g, t_ns

    def step(self, dt: float, t_ns: int) -> None:
        c = self.cfg
        for d in self.drones:
            fresh = d.goal is not None and (t_ns - d.goal_ns) / 1e9 <= c.goal_timeout_s
            if fresh:
                gp = d.goal["position_world_ned_m"]
                gn, ge, galt = gp["n"], gp["e"], -gp["d"]
                vmax = min(c.max_speed_mps, d.goal.get("max_speed_mps", c.max_speed_mps))
            else:  # 逾時：懸停
                gn, ge, galt, vmax = d.n, d.e, d.alt, c.max_speed_mps
            want_n, want_e = c.position_gain * (gn - d.n), c.position_gain * (ge - d.e)
            # 機間互斥（只看水平距離且高度差 < 3 m 的才算衝突）
            for o in self.drones:
                if o is d:
                    continue
                dn, de = d.n - o.n, d.e - o.e
                dist = math.hypot(dn, de)
                if dist < c.min_separation_m and abs(d.alt - o.alt) < 3.0 and dist > 1e-6:
                    push = c.separation_gain * (c.min_separation_m - dist)
                    want_n += push * dn / dist
                    want_e += push * de / dist
            sp = math.hypot(want_n, want_e)
            if sp > vmax:
                want_n, want_e = want_n * vmax / sp, want_e * vmax / sp
            # 加速度限制
            an, ae = (want_n - d.vn) / dt, (want_e - d.ve) / dt
            a = math.hypot(an, ae)
            if a > c.max_accel_mps2:
                an, ae = an * c.max_accel_mps2 / a, ae * c.max_accel_mps2 / a
            d.vn += an * dt
            d.ve += ae * dt
            d.n += d.vn * dt
            d.e += d.ve * dt
            climb = max(-c.max_climb_mps, min(c.max_climb_mps, galt - d.alt))
            d.alt += climb * dt
            # 航向：有指定 yaw 就轉過去，否則朝移動方向
            want_yaw = d.goal.get("yaw_rad") if fresh and d.goal.get("yaw_rad") is not None else (
                math.atan2(d.ve, d.vn) if math.hypot(d.vn, d.ve) > 0.5 else d.yaw)
            err = (want_yaw - d.yaw + math.pi) % (2 * math.pi) - math.pi
            d.yaw = (d.yaw + max(-c.max_yaw_rate_rps * dt, min(c.max_yaw_rate_rps * dt, err)) + math.pi) % (2 * math.pi) - math.pi
        for i, a in enumerate(self.drones):
            for b in self.drones[i + 1 :]:
                self.min_pair_distance_m = min(self.min_pair_distance_m, math.dist((a.n, a.e, a.alt), (b.n, b.e, b.alt)))

    def states(self) -> List[DroneState]:
        return [DroneState(d.drone_id, d.n, d.e, d.alt, d.yaw) for d in self.drones]

    def fleet_pose(self, t_ns: int) -> List[dict]:
        """規格 §10.2 FleetPose。"""
        return [
            {
                "schema_version": "0.2.0",
                "drone_id": d.drone_id,
                "timestamp_ns": int(t_ns),
                "position_world_ned_m": {"n": d.n, "e": d.e, "d": -d.alt},
                "attitude_rad": {"roll": 0.0, "pitch": 0.0, "yaw": d.yaw},
                "velocity_world_ned_mps": {"n": d.vn, "e": d.ve, "d": 0.0},
                "status": "flying",
            }
            for d in self.drones
        ]
