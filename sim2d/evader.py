"""會逃的模擬目標（策略場景用）：目標會對無人機做出反應，驅趕策略才有意義。

這是「我們設計的簡化逃跑模型」，報告要說明它不代表真實車輛或行人的行為。模型：

    威脅向量 = Σ（每架在 flee_radius_m 內的無人機）w_i × 單位向量（無人機 → 目標）
        w_i = 接近程度 × 動態係數
        接近程度 = 1 − 水平距離 / flee_radius_m（越近越大，0–1）
        動態係數 = hover_weight + (1 − hover_weight) × clip(接近速度 / approach_ref_mps, 0, 1)
                   → 懸停的無人機威脅較小、正在逼近的較大（會動的東西比較嚇人）
    威脅程度 = min(1, Σ w_i)
    牆：離場地邊界 wall_margin_m 內時加一個指向中心的向量（目標不會往牆衝，所以靠近邊界時會被「卡住」，
        這也是驅趕可以利用的地方）

    有威脅：朝（威脅向量 + 牆向量）方向轉（有轉彎率上限），速度 = cruise + (flee_max − cruise) × 威脅程度
    沒有威脅：和 RabbitPath 一樣平滑遊走（Ornstein–Uhlenbeck 轉彎率），速度 = cruise

只看水平距離（無人機在 25–35 m 的高度層，高度差對「被追」的感受影響不大，這是簡化）。
參數選擇：flee_max_mps 要低於無人機速度上限（A 的 SITL 是 3 m/s），否則任何驅趕都追不上，報告要說明。

介面同 RabbitPath：屬性 n, e, cfg.arena_radius_m；step(dt, drones) 回傳 (n, e)，單位 m、world_ned。
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class EvaderConfig:
    cruise_mps: float = 1.0  # 沒有威脅時的速度
    flee_max_mps: float = 2.2  # 威脅程度 = 1 時的速度（< 無人機上限 3 m/s）
    max_accel_mps2: float = 1.5
    max_turn_rate_rps: float = math.radians(60)
    flee_radius_m: float = 25.0  # 無人機水平距離在這以內才會被注意到
    hover_weight: float = 0.3  # 懸停中的無人機威脅打幾折
    approach_ref_mps: float = 2.0  # 接近速度達到這個值時，動態係數 = 1
    arena_radius_m: float = 60.0  # 圓形場地（A 的地理圍欄是 ±80 m，留餘裕）
    wall_margin_m: float = 15.0
    wall_gain: float = 1.5
    start_ne: Tuple[float, float] = (40.0, 0.0)
    start_heading_rad: float = 0.0
    seed: int = 0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Evader:
    def __init__(self, cfg: EvaderConfig = EvaderConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n, self.e = cfg.start_ne
        self.heading = cfg.start_heading_rad
        self.speed = cfg.cruise_mps
        self.threat = 0.0  # 最近一步的威脅程度（0–1），畫面與分析用
        self._turn = 0.0
        self._last_drone_ne: Dict[str, Tuple[float, float]] = {}

    @property
    def velocity_ne(self) -> Tuple[float, float]:
        """目前速度（world_ned 的 n, e 分量，m/s）。"""
        return self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)

    def threat_vector(self, drones: Sequence, dt: float) -> Tuple[float, float, float]:
        """（威脅向量 n, e, 威脅程度）。drones 為 fusion.DroneState（world_ned, m）；會更新無人機速度估計。"""
        c = self.cfg
        tn = te = 0.0
        total = 0.0
        seen = {}
        for d in drones:
            dn, de = self.n - d.n, self.e - d.e
            dist = math.hypot(dn, de)
            prev = self._last_drone_ne.get(d.drone_id)
            seen[d.drone_id] = (d.n, d.e)
            if dist >= c.flee_radius_m or dist < 1e-6:
                continue
            ux, uy = dn / dist, de / dist
            closing = 0.0
            if prev is not None and dt > 0:  # 無人機速度用前後兩步差分估計（目標「看到」無人機在動）
                vn, ve = (d.n - prev[0]) / dt, (d.e - prev[1]) / dt
                closing = vn * ux + ve * uy
            dyn = c.hover_weight + (1 - c.hover_weight) * _clip(closing / c.approach_ref_mps, 0.0, 1.0)
            w = (1 - dist / c.flee_radius_m) * dyn
            tn += w * ux
            te += w * uy
            total += w
        self._last_drone_ne = seen
        return tn, te, min(1.0, total)

    def wall_vector(self) -> Tuple[float, float]:
        """靠近邊界時指向中心的向量（大小 0–1）。"""
        c = self.cfg
        r = math.hypot(self.n, self.e)
        inner = c.arena_radius_m - c.wall_margin_m
        if r <= inner or r < 1e-6:
            return 0.0, 0.0
        k = _clip((r - inner) / c.wall_margin_m, 0.0, 1.0)
        return -k * self.n / r, -k * self.e / r

    def step(self, dt: float, drones: Optional[Sequence] = None) -> Tuple[float, float]:
        c = self.cfg
        tn, te, threat = self.threat_vector(drones or [], dt)
        wn, we = self.wall_vector()
        self.threat = threat
        if threat > 0:
            want = math.atan2(te + c.wall_gain * we, tn + c.wall_gain * wn)
            self._turn = _clip(_wrap(want - self.heading) / dt, -c.max_turn_rate_rps, c.max_turn_rate_rps)
            want_speed = c.cruise_mps + (c.flee_max_mps - c.cruise_mps) * threat
        else:
            self._turn += (-0.5 * self._turn) * dt + 0.4 * c.max_turn_rate_rps * math.sqrt(dt) * self.rng.normal()
            if wn or we:  # 遊走時靠近邊界：轉回中心
                err = _wrap(math.atan2(we, wn) - self.heading)
                self._turn = math.copysign(c.max_turn_rate_rps, err)
            self._turn = _clip(self._turn, -c.max_turn_rate_rps, c.max_turn_rate_rps)
            want_speed = c.cruise_mps
        self.heading = _wrap(self.heading + self._turn * dt)
        self.speed += _clip(want_speed - self.speed, -c.max_accel_mps2 * dt, c.max_accel_mps2 * dt)
        self.n += self.speed * math.cos(self.heading) * dt
        self.e += self.speed * math.sin(self.heading) * dt
        r = math.hypot(self.n, self.e)
        if r > c.arena_radius_m:  # 硬邊界：不會跑出場地
            self.n, self.e = self.n * c.arena_radius_m / r, self.e * c.arena_radius_m / r
        return self.n, self.e
