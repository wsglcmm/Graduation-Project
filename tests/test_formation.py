"""隊形規劃器：角色 → 各機目標點的幾何測試。"""
import math

import pytest

from fleet_fixtures import DRONES, ENC, IDS, S, target  # noqa: F401
from llm_agent.fleet_schema import Plan
from planner import FormationPlanner


# ---------- 隊形規劃 ----------
def plan(mode="encircle", **kw):
    return Plan.model_validate({**ENC, "mode": mode, **kw})


def test_encircle_slots_are_120_degrees_apart_at_radius():
    fp = FormationPlanner()
    gps = fp.goal_points(plan(), "p1", DRONES, [target(vn=4.0)], 0)
    lead = (4.0 * fp.cfg.lead_s, 0.0)  # 目標往北 4 m/s，攔截中心提前 4 m
    pos = {g["drone_id"]: g["position_world_ned_m"] for g in gps}
    for g in pos.values():
        assert math.hypot(g["n"] - lead[0], g["e"] - lead[1]) == pytest.approx(15.0)
    assert pos["uav1"]["n"] == pytest.approx(lead[0] - 15.0) and pos["uav1"]["e"] == pytest.approx(0.0)  # chaser 在正後方
    assert pos["uav2"]["e"] < 0 < pos["uav3"]["e"]  # 左翼在西、右翼在東
    assert pos["uav2"]["n"] > lead[0] and pos["uav3"]["n"] > lead[0]  # 兩翼在前方


def test_altitude_layers_and_goal_point_fields():
    gps = FormationPlanner().goal_points(plan(), "p7", DRONES, [target()], 5 * S)
    assert [g["position_world_ned_m"]["d"] for g in gps] == [-25.0, -30.0, -35.0]
    g = gps[0]
    assert set(g) >= {"schema_version", "plan_id", "drone_id", "timestamp_ns", "position_world_ned_m", "max_speed_mps"}
    assert g["plan_id"] == "p7" and g["timestamp_ns"] == 5 * S and g["schema_version"] == "0.2.0"


def test_hold_keeps_position_and_track_uses_wider_radius():
    fp = FormationPlanner()
    hold = plan("hold", target_id=None, assignments=[{"drone_id": i, "role": "idle"} for i in IDS])
    gps = fp.goal_points(hold, "p", DRONES, [target()], 0)
    assert [(g["position_world_ned_m"]["n"], g["position_world_ned_m"]["e"]) for g in gps] == [(d.n, d.e) for d in DRONES]
    gp_track = fp.goal_points(plan("track"), "p", DRONES, [target(vn=0.0)], 0)[0]["position_world_ned_m"]
    assert math.hypot(gp_track["n"], gp_track["e"]) == pytest.approx(15.0 * fp.cfg.track_extra_radius_factor)


def test_search_orbits_last_known_position_with_phase_offsets():
    fp = FormationPlanner()
    fp.goal_points(plan(), "p", DRONES, [target(n=50, e=10, vn=0)], 0)  # 記住最後位置
    srch = plan("search", target_id=None, assignments=[{"drone_id": i, "role": "searcher"} for i in IDS], formation={"radius_m": 20.0})
    gps = fp.goal_points(srch, "p", DRONES, [target(n=50, e=10, status="lost")], 0)
    pts = [(g["position_world_ned_m"]["n"], g["position_world_ned_m"]["e"]) for g in gps]
    for n, e in pts:
        assert math.hypot(n - 50, e - 10) == pytest.approx(20.0)
    assert len({(round(n, 3), round(e, 3)) for n, e in pts}) == 3  # 三架相位錯開


def test_return_home_uses_home_positions():
    fp = FormationPlanner(home={"uav1": (0.0, 0.0), "uav2": (0.0, 5.0), "uav3": (0.0, 10.0)})
    rth = plan("return_home", target_id=None, assignments=[{"drone_id": i, "role": "idle"} for i in IDS])
    rth = rth.model_copy(update={"assignments": [a.model_copy(update={"role": "chaser"}) for a in rth.assignments]})
    gps = fp.goal_points(rth, "p", DRONES, [], 0)
    assert [(g["position_world_ned_m"]["n"], g["position_world_ned_m"]["e"]) for g in gps] == [(0.0, 0.0), (0.0, 5.0), (0.0, 10.0)]
