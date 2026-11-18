"""捕捉策略：schema 白名單、規則式分角色、捕捉區與各角色目標點的幾何。"""
import json
import math

import pytest

from fusion import DroneState
from llm_agent.commander import Commander
from llm_agent.fleet_schema import parse_plan
from llm_agent.rule_planner import capture_assignment, rule_plan
from planner import CaptureConfig, CaptureGeometry, FormationPlanner, propose_zone

IDS = ["uav1", "uav2", "uav3"]
DRONES = [DroneState("uav1", -20, 0, 25), DroneState("uav2", 5, -15, 30), DroneState("uav3", 5, 15, 35)]


def target(n=0.0, e=0.0, vn=0.0, ve=0.0, status="tracking", tid=1):
    return {"target_id": tid, "class": "car", "status": status, "position_world_ned_m": {"n": n, "e": e, "d": 0.0},
            "velocity_world_ned_mps": {"n": vn, "e": ve, "d": 0.0}, "uncertainty_m": 1.0, "observed_by": ["uav1"], "last_seen_ns": 0}


def plan_json(mode, roles, tid=1):
    return json.dumps({"mode": mode, "target_id": tid, "assignments": [{"drone_id": d, "role": r} for d, r in zip(IDS, roles)],
                       "reason": "測試"})


# ---------- schema：角色專屬白名單 ----------
def test_capture_plan_valid():
    assert parse_plan(plan_json("capture", ["herder", "blocker", "capturer"]), IDS, [1]).ok


@pytest.mark.parametrize("roles, msg", [
    (["herder", "chaser", "capturer"], "只能用"),
    (["herder", "herder", "blocker"], "恰好一架 capturer"),
    (["capturer", "blocker", "blocker"], "至少一架 herder"),
])
def test_capture_plan_role_rules(roles, msg):
    res = parse_plan(plan_json("capture", roles), IDS, [1])
    assert not res.ok and msg in res.error


def test_capture_roles_rejected_outside_capture():
    res = parse_plan(plan_json("encircle", ["herder", "blocker_left", "blocker_right"]), IDS, [1])
    assert not res.ok and "只能在 mode=capture" in res.error


def test_capture_needs_existing_target():
    assert "必須指定 target_id" in parse_plan(plan_json("capture", ["herder", "blocker", "capturer"], tid=None), IDS, [1]).error


# ---------- 規則式 ----------
@pytest.mark.parametrize("cmd", ["把那台車趕進捕捉區抓拍", "捕捉那台車", "跟上去抓拍"])
def test_rule_operator_capture_keywords(cmd):
    p = rule_plan(DRONES, [target()], cmd)
    assert p.mode == "capture" and parse_plan(p.model_dump_json(), IDS, [1]).ok


def test_rule_capture_assignment_by_geometry():
    """機群在目標南邊 → 捕捉區在北邊：最北的當抓拍機、最南的當驅趕機。"""
    roles = {a["drone_id"]: a["role"] for a in capture_assignment(DRONES, 0.0, 0.0)}
    assert roles["uav1"] == "herder"
    assert roles["uav3"] in ("capturer", "blocker") and roles["uav2"] in ("capturer", "blocker")


def test_rule_default_strategy_capture_and_lost_target_searches():
    assert rule_plan(DRONES, [target()], strategy="capture").mode == "capture"
    assert rule_plan(DRONES, [target(status="lost")], strategy="capture").mode == "search"
    assert rule_plan(DRONES, [target()]).mode == "encircle"  # 預設策略不變


def test_commander_capture_directive_and_strategy():
    c = Commander(IDS, llm=None, strategy="capture")
    c.tick(0, DRONES, [target()])
    assert c.current.mode == "capture"
    c.command("三架一起包圍")
    c.tick(int(1e8), DRONES, [target()])
    assert c.current.mode == "encircle"  # 操作員指令優先於預設策略


# ---------- 幾何 ----------
def test_zone_ahead_of_target_away_from_fleet():
    z = propose_zone(0.0, 0.0, DRONES, CaptureConfig())
    assert z.n == pytest.approx(35.0, abs=1e-6) and z.e == pytest.approx(0.0, abs=1e-6)


def test_zone_stays_inside_arena_near_wall():
    cfg = CaptureConfig()
    drones = [DroneState("uav1", 25.0, 0.0, 25)]
    z = propose_zone(45.0, 0.0, drones, cfg)  # 往北已經靠牆 → 改轉向
    assert math.hypot(z.n, z.e) <= cfg.arena_radius_m - cfg.zone_margin_m + 1e-6
    assert math.hypot(z.n - 45.0, z.e) > 15.0


def make_geo(tgt, drones=DRONES):
    g = CaptureGeometry(CaptureConfig(lead_s=0.0))
    g.ensure_zone(tgt, drones, 0.0)
    return g


def test_herder_behind_target_relative_to_zone():
    t = target()
    g = make_geo(t)  # 捕捉區在北 (35, 0)
    n, e = g.goal("herder", t, DroneState("uav1", -20, 0, 25))
    assert (n, e) == pytest.approx((-12.0, 0.0))


def test_herder_on_wrong_side_goes_around():
    t = target()
    g = make_geo(t)
    n, e = g.goal("herder", t, DroneState("uav1", 10, 3, 25))  # 在目標和捕捉區之間
    assert abs(e) == pytest.approx(18.0) and n == pytest.approx(0.0, abs=1e-6)  # 先到側面，不直接穿過目標


def test_capturer_waits_then_engages():
    g = make_geo(target())
    assert g.goal("capturer", target(), DRONES[2]) == pytest.approx((35.0, 0.0))
    near = target(n=20.0)  # 離捕捉區 15 m < engage 18 m
    assert g.goal("capturer", near, DRONES[2]) == pytest.approx((20.0, 0.0))


def test_capturer_engage_has_hysteresis():
    """迎上去之後，目標被嚇退到 18–30 m 之間仍繼續追；超過 30 m 才回伏擊點（避免來回擺盪）。"""
    g = make_geo(target())  # 捕捉區 (35, 0)
    assert g.goal("capturer", target(n=20.0), DRONES[2]) == pytest.approx((20.0, 0.0))  # 15 m：迎上去
    assert g.goal("capturer", target(n=10.0), DRONES[2]) == pytest.approx((10.0, 0.0))  # 25 m：繼續追
    assert g.goal("capturer", target(n=0.0), DRONES[2]) == pytest.approx((35.0, 0.0))  # 35 m：放棄、回伏擊點
    assert g.goal("capturer", target(n=10.0), DRONES[2]) == pytest.approx((35.0, 0.0))  # 25 m 但未重新進入 18 m：繼續等


def test_blocker_follows_lateral_drift_with_hysteresis():
    g = make_geo(target())
    east = g.goal("blocker", target(ve=1.0), DRONES[1])
    assert east[1] > 0
    still = g.goal("blocker", target(ve=0.1), DRONES[1])  # 側向速度太小：維持原邊
    assert still[1] > 0
    west = g.goal("blocker", target(ve=-1.0), DRONES[1])
    assert west[1] < 0


def test_formation_planner_keeps_zone_and_resets_when_leaving_capture():
    fp = FormationPlanner()
    cap = rule_plan(DRONES, [target()], "抓拍那台車")
    fp.goal_points(cap, "p1", DRONES, [target()], 0)
    z1 = fp.capture_zone
    fp.goal_points(cap, "p1", DRONES, [target(n=5.0)], int(1e8))
    assert fp.capture_zone == z1  # 同一次 capture 捕捉區固定
    fp.goal_points(rule_plan(DRONES, [target()], "包圍"), "p2", DRONES, [target()], int(2e8))
    assert fp.capture_zone is None


def test_rule_keeps_capture_roles_during_same_capture():
    first = rule_plan(DRONES, [target()], "抓拍那台車")
    moved = [DroneState("uav1", 30, 0, 25), DroneState("uav2", -30, 0, 30), DroneState("uav3", 0, 30, 35)]
    again = rule_plan(moved, [target(n=3.0)], "抓拍那台車", previous=first)
    assert again.assignments == first.assignments
    other = rule_plan(moved, [target(n=3.0, tid=2)], "抓拍那台車", previous=first)  # 換目標才重新分派
    assert [x.model_dump() for x in other.assignments] == capture_assignment(moved, 3.0, 0.0)



def test_capture_geometry_obeys_drone_agent_actions():
    """多 Agent：各機 Agent 選的動作優先於內建規則。"""
    g = make_geo(target())  # 捕捉區 (35, 0)
    assert g.goal("capturer", target(n=0.0), DRONES[2], "engage") == pytest.approx((0.0, 0.0))  # 35 m 外也照 Agent 的決定迎上去
    assert g.goal("capturer", target(n=20.0), DRONES[2], "ambush") == pytest.approx((35.0, 0.0))
    assert g.goal("herder", target(), DroneState("uav1", -20, 0, 25), "push") == pytest.approx((-12.0, 0.0))
    n, e = g.goal("herder", target(), DroneState("uav1", -20, 0, 25), "circle_behind")
    assert abs(e) == pytest.approx(18.0) and n == pytest.approx(0.0, abs=1e-6)


def test_zone_direction_options():
    drones = [DroneState("uav1", -20, 0, 25)]  # 機群在南邊
    def zone(mode, vel=None, tn=0.0, te=20.0):
        return propose_zone(tn, te, drones, CaptureConfig(zone_direction=mode), vel=vel)
    z = zone("fleet")
    assert z.n > 0  # 機群 → 目標的延長線（往北偏東）
    z = zone("velocity", vel=(0.0, -2.0))
    assert z.e < 20.0 - 30  # 目標往西跑 → 捕捉區在西邊
    z = zone("center")
    assert abs(z.n) < 1e-6 and z.e < 20.0  # 目標 → 場地中心
    assert zone("auto", vel=(0.1, 0.0)) == zone("center")  # 幾乎不動：朝中心


def test_zone_replan_only_when_enabled_and_target_turns_away():
    from planner.capture import CaptureConfig, CaptureGeometry

    drones = [type("D", (), {"n": -30.0, "e": 0.0})()]

    def tgt(n, e, vn, ve):
        return {"target_id": 1, "position_world_ned_m": {"n": n, "e": e}, "velocity_world_ned_mps": {"n": vn, "e": ve}}

    fixed = CaptureGeometry(CaptureConfig(zone_direction="velocity", arena_radius_m=200.0))
    z0 = fixed.ensure_zone(tgt(0, 0, 5, 0), drones, 0.0, 0.0)
    assert (round(z0.n), round(z0.e)) == (35, 0)
    assert fixed.ensure_zone(tgt(0, 0, 0, 5), drones, 0.0, 10.0) == z0  # 預設：固定不動（策略場景）

    g = CaptureGeometry(CaptureConfig(zone_direction="velocity", arena_radius_m=200.0, replan_zone_s=3.0))
    g.ensure_zone(tgt(0, 0, 5, 0), drones, 0.0, 0.0)
    assert g.ensure_zone(tgt(0, 0, 0, 5), drones, 0.0, 1.0) == z0  # 還沒到 3 s
    assert g.ensure_zone(tgt(0, 0, 0.5, 0.5), drones, 0.0, 5.0) == z0  # 太慢（停車抖動）不換
    assert g.ensure_zone(tgt(0, 0, 4, 1), drones, 0.0, 5.0) == z0  # 仍大致朝捕捉區前進
    z1 = g.ensure_zone(tgt(0, 0, 0, 5), drones, 0.0, 5.0)  # 轉 90°：重新預測到東方
    assert (round(z1.n), round(z1.e)) == (0, 35) and g.zone_replans == 1
