"""兔子（地面目標）路線：平滑但不可預測的遊走，速度與轉彎率有上限，不會跑出場地。

鳥瞰影片接上之前先用這個；之後目標位置改由「B 的影片偵測 + 地面標定」提供，介面一樣是 (n, e)。
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RabbitConfig:
    speed_min_mps: float = 2.0
    speed_max_mps: float = 5.0
    max_turn_rate_rps: float = math.radians(25)
    arena_radius_m: float = 80.0  # 超出就轉回中心
    start_ne: tuple = (40.0, 0.0)
    start_heading_rad: float = 0.0
    seed: int = 0


class RabbitPath:
    def __init__(self, cfg: RabbitConfig = RabbitConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.n, self.e = cfg.start_ne
        self.heading = cfg.start_heading_rad
        self.speed = (cfg.speed_min_mps + cfg.speed_max_mps) / 2
        self._turn = 0.0

    def step(self, dt: float) -> tuple:
        c = self.cfg
        # 轉彎率做隨機遊走（Ornstein–Uhlenbeck），讓轉彎平滑而不是每步亂跳
        self._turn += (-0.5 * self._turn) * dt + 0.6 * c.max_turn_rate_rps * math.sqrt(dt) * self.rng.normal()
        if math.hypot(self.n, self.e) > c.arena_radius_m:  # 出界：轉向中心
            to_center = math.atan2(-self.e, -self.n)
            err = (to_center - self.heading + math.pi) % (2 * math.pi) - math.pi
            self._turn = math.copysign(c.max_turn_rate_rps, err)
        self._turn = max(-c.max_turn_rate_rps, min(c.max_turn_rate_rps, self._turn))
        self.heading += self._turn * dt
        self.speed += 0.3 * self.rng.normal() * math.sqrt(dt)
        self.speed = max(c.speed_min_mps, min(c.speed_max_mps, self.speed))
        self.n += self.speed * math.cos(self.heading) * dt
        self.e += self.speed * math.sin(self.heading) * dt
        return self.n, self.e
