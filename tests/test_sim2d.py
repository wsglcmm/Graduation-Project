"""無人機替身與兔子路線的測試：速度上限、逾時懸停、機間互斥、場地邊界。"""
import math

import pytest

from sim2d import FleetSim, FleetSimConfig, RabbitConfig, RabbitPath

S = 1_000_000_000


def goal(did, n, e, alt=30.0, **kw):
    return {"drone_id": did, "position_world_ned_m": {"n": n, "e": e, "d": -alt}, **kw}


def run(sim, goals, seconds, refresh=True, dt=0.1):
    t = 0
    for _ in range(int(seconds / dt)):
        t += int(dt * S)
        if refresh:
            sim.apply_goals(goals, t)
        sim.step(dt, t)
    return t


def test_reaches_goal_and_respects_speed_limit():
    sim = FleetSim({"uav1": (0.0, 0.0, 30.0)})
    speeds = []
    t = 0
    for _ in range(300):
        t += S // 10
        sim.apply_goals([goal("uav1", 100.0, 0.0)], t)
        sim.step(0.1, t)
        d = sim.drones[0]
        speeds.append(math.hypot(d.vn, d.ve))
    assert max(speeds) <= FleetSimConfig().max_speed_mps + 1e-9
    assert sim.states()[0].n == pytest.approx(100.0, abs=0.5)


def test_goal_max_speed_is_respected():
    sim = FleetSim({"uav1": (0.0, 0.0, 30.0)})
    run(sim, [goal("uav1", 100.0, 0.0, max_speed_mps=3.0)], 10)
    d = sim.drones[0]
    assert math.hypot(d.vn, d.ve) <= 3.0 + 1e-9


def test_hover_after_goal_timeout():
    sim = FleetSim({"uav1": (0.0, 0.0, 30.0)})
    t = 0
    for k in range(200):  # 只在第一步給目標點，之後不再更新
        t += S // 10
        if k == 0:
            sim.apply_goals([goal("uav1", 100.0, 0.0)], t)
        sim.step(0.1, t)
    n_after = sim.states()[0].n
    assert n_after < 20.0  # 1 秒逾時後就停下來，不會飛到 100 m
    d = sim.drones[0]
    assert math.hypot(d.vn, d.ve) < 0.1


def test_separation_keeps_same_altitude_drones_apart():
    sim = FleetSim({"uav1": (0.0, -10.0, 30.0), "uav2": (0.0, 10.0, 30.0)})
    run(sim, [goal("uav1", 0.0, 0.0), goal("uav2", 0.0, 0.0)], 20)  # 兩架被叫到同一點、同高度
    a, b = sim.states()
    assert math.hypot(a.n - b.n, a.e - b.e) > 0.6 * FleetSimConfig().min_separation_m


def test_fleet_pose_matches_spec_fields():
    sim = FleetSim({"uav1": (1.0, 2.0, 25.0)})
    (fp,) = sim.fleet_pose(5 * S)
    assert fp["schema_version"] == "0.2.0" and fp["drone_id"] == "uav1" and fp["timestamp_ns"] == 5 * S
    assert fp["position_world_ned_m"] == {"n": 1.0, "e": 2.0, "d": -25.0}


def test_rabbit_stays_near_arena_and_respects_speed():
    r = RabbitPath(RabbitConfig(seed=3, arena_radius_m=50.0))
    prev = (r.n, r.e)
    for _ in range(3000):
        pos = r.step(0.1)
        assert math.dist(pos, prev) / 0.1 <= r.cfg.speed_max_mps + 1e-9
        prev = pos
        assert math.hypot(*pos) < 50.0 + 30.0


def test_rabbit_is_deterministic_for_seed():
    a, b = RabbitPath(RabbitConfig(seed=7)), RabbitPath(RabbitConfig(seed=7))
    assert [a.step(0.1) for _ in range(50)] == [b.step(0.1) for _ in range(50)]
