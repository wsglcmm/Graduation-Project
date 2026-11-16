"""會逃的模擬目標：方向、速度上限、懸停 vs 逼近、場地邊界。"""
import math

import pytest

from fusion import DroneState
from sim2d import Evader, EvaderConfig

DT = 0.1


def run(ev, drones_fn, seconds):
    for k in range(int(seconds / DT)):
        ev.step(DT, drones_fn(k))


def test_no_drones_wanders_at_cruise_inside_arena():
    ev = Evader(EvaderConfig(seed=3))
    for _ in range(3000):  # 5 分鐘
        ev.step(DT, [])
        assert math.hypot(ev.n, ev.e) <= ev.cfg.arena_radius_m + 1e-9
    assert ev.speed == pytest.approx(ev.cfg.cruise_mps)
    assert ev.threat == 0.0


def test_flees_away_from_approaching_drone_up_to_max_speed():
    cfg = EvaderConfig(start_ne=(0.0, 0.0), start_heading_rad=math.pi / 2)  # 一開始朝東
    ev = Evader(cfg)
    # 無人機一直跟在目標南邊 12 m（跟著目標往北移動 = 逼近中）
    run(ev, lambda k: [DroneState("uav1", ev.n - 12.0, ev.e, 30.0)], 6.0)
    vn, ve = ev.velocity_ne
    assert vn > 0.9 * ev.speed  # 轉向北逃
    assert cfg.cruise_mps < ev.speed <= cfg.flee_max_mps + 1e-9


def test_hovering_drone_is_less_scary_than_approaching_one():
    pos = DroneState("uav1", -10.0, 0.0, 30.0)
    hover = Evader(EvaderConfig(start_ne=(0.0, 0.0)))
    hover.threat_vector([pos], DT)
    _, _, w_hover = hover.threat_vector([pos], DT)
    chase = Evader(EvaderConfig(start_ne=(0.0, 0.0)))
    chase.threat_vector([DroneState("uav1", -10.3, 0.0, 30.0)], DT)  # 0.1 s 內前進 0.3 m = 3 m/s 逼近
    _, _, w_chase = chase.threat_vector([pos], DT)
    assert w_hover == pytest.approx((1 - 10 / 25) * 0.3)
    assert w_chase == pytest.approx(1 - 10 / 25)


def test_ignores_drones_outside_flee_radius():
    ev = Evader(EvaderConfig(start_ne=(0.0, 0.0)))
    tn, te, w = ev.threat_vector([DroneState("uav1", 30.0, 0.0, 30.0)], DT)
    assert (tn, te, w) == (0.0, 0.0, 0.0)


def test_pinned_against_wall_stays_inside():
    cfg = EvaderConfig(start_ne=(50.0, 0.0), start_heading_rad=0.0)  # 朝北、離北邊界 10 m
    ev = Evader(cfg)
    run(ev, lambda k: [DroneState("uav1", ev.n - 10.0, ev.e, 30.0)], 20.0)  # 從南邊一直往北趕
    assert math.hypot(ev.n, ev.e) <= cfg.arena_radius_m + 1e-9
    assert ev.n > 45.0  # 被趕到北邊界附近，沒有穿過無人機往回跑
