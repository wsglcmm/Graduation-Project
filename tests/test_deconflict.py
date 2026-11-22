"""防撞第二層：目標點水平去衝突 + 飛行途中交錯讓路（已知答案）。"""
import math

import pytest

from fusion import DroneState
from planner.deconflict import closest_approach, deconflict_goals, min_horizontal_separation, yield_crossings


def gp(did, n, e, d=-30.0):
    return {"drone_id": did, "plan_id": "p1", "position_world_ned_m": {"n": n, "e": e, "d": d}}


def ne(goals, did):
    g = next(x for x in goals if x["drone_id"] == did)["position_world_ned_m"]
    return g["n"], g["e"]


def test_far_apart_goals_unchanged():
    goals = [gp("uav1", 0, 0), gp("uav2", 0, 10)]
    out, adj = deconflict_goals(goals, {})
    assert out == goals and adj == []


def test_lower_priority_moves_capturer_stays():
    goals = [gp("uav1", 0, 0), gp("uav2", 0, 2)]
    out, adj = deconflict_goals(goals, {"uav1": "capturer", "uav2": "herder"})
    assert ne(out, "uav1") == (0, 0)  # 抓拍機不動（才不會被推離目標正上方）
    assert ne(out, "uav2") == pytest.approx((0, 6))
    assert len(adj) == 1 and adj[0].drone_id == "uav2" and adj[0].because_of == "uav1"
    assert adj[0].moved_m == pytest.approx(4.0)
    assert out[1]["position_world_ned_m"]["d"] == -30.0  # 高度不動


def test_same_priority_split_and_coincident_points():
    out, _ = deconflict_goals([gp("uav1", 0, 0), gp("uav2", 0, 0)], {"uav1": "blocker", "uav2": "blocker"})
    assert ne(out, "uav1") == pytest.approx((0, -3)) and ne(out, "uav2") == pytest.approx((0, 3))


def test_three_way_cluster_converges():
    goals = [gp("uav1", 0, 0), gp("uav2", 1, 1), gp("uav3", -1, 2)]
    out, adj = deconflict_goals(goals, {"uav1": "capturer", "uav2": "herder", "uav3": "blocker"})
    assert min_horizontal_separation(out) >= 6.0 - 1e-6
    assert ne(out, "uav1") == (0, 0) and {a.drone_id for a in adj} == {"uav2", "uav3"}


def test_disabled_with_zero_separation():
    goals = [gp("uav1", 0, 0), gp("uav2", 0, 1)]
    assert deconflict_goals(goals, {}, min_sep_m=0)[0] == goals


def test_closest_approach_head_on_and_parallel():
    d, t = closest_approach((0, 0), (0, 3), (0, 30), (0, -3), horizon_s=10)  # 正面對飛，5 s 後相撞
    assert d == pytest.approx(0) and t == pytest.approx(5)
    d, t = closest_approach((0, 0), (3, 0), (0, 10), (3, 0), horizon_s=10)  # 平行同速
    assert d == pytest.approx(10) and t == 0
    d, _ = closest_approach((0, 0), (0, 3), (0, 30), (0, -3), horizon_s=2)  # 預測時間內還沒碰到
    assert d == pytest.approx(18)


def test_crossing_paths_lower_priority_yields_sideways():
    """uav1（抓拍）往東飛，uav2（驅趕）從北往南穿過它的路線：goal 都很遠、彼此分開，但途中會交錯。"""
    drones = [DroneState("uav1", 0, -20, 25), DroneState("uav2", 20, 0, 30)]
    goals = [gp("uav1", 0, 40), gp("uav2", -40, 0)]
    assert min_horizontal_separation(goals) > 6  # 只看目標點看不出問題
    out, adj = yield_crossings(goals, drones, {"uav1": "capturer", "uav2": "herder"}, vmax=3.0)
    assert ne(out, "uav1") == (0, 40)
    n, e = ne(out, "uav2")
    assert math.hypot(n - 20, e - 0) == pytest.approx(8.0)  # 原地橫移 6 + 2 m
    assert n > 20  # 往自己所在那側（北邊）讓開，不是穿過對方路線
    assert adj[0].because_of == "uav1"


def test_no_yield_when_paths_clear():
    drones = [DroneState("uav1", 0, 0, 25), DroneState("uav2", 30, 0, 30)]
    goals = [gp("uav1", 0, 30), gp("uav2", 30, 30)]  # 平行往東，相距 30 m
    out, adj = yield_crossings(goals, drones, {})
    assert adj == [] and out == goals


def test_planner_deconflicts_but_not_return_home():
    from llm_agent.fleet_schema import Plan
    from planner import FormationPlanner

    drones = [DroneState("uav1", 0, 0, 25), DroneState("uav2", 0, 1, 30)]
    fp = FormationPlanner(home={"uav1": (0.0, 0.0), "uav2": (0.0, 1.0)})
    hold = Plan(mode="hold", assignments=[{"drone_id": "uav1", "role": "idle"}, {"drone_id": "uav2", "role": "idle"}], reason="t")
    out = fp.goal_points(hold, "p1", drones, [], 0)
    assert min_horizontal_separation(out) >= 6.0 - 1e-6 and fp.adjustment_count >= 1
    rth = Plan(mode="return_home", assignments=hold.assignments, reason="t")
    out = fp.goal_points(rth, "p2", drones, [], 1)
    assert fp.last_adjustments == [] and ne(out, "uav2") == (0.0, 1.0)  # 返航由 A 負責，不改出生點
