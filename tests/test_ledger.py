"""動作日誌與驗證：動作的開始 / 接續 / 結束、到達判定、A 的回覆、匯出。"""
import csv
import json

import pytest

from fleet_link.messages import GoalPointAck
from fusion import DroneState
from ledger import ActionLedger, VerifyConfig
from llm_agent.commander import AgentEvent, Commander

IDS = ["uav1", "uav2"]


def event(plan_id="p1", mode="encircle", roles=("chaser", "blocker_left"), valid=True, trigger="periodic", error=None, target_id=1):
    plan = None
    if valid:
        plan = {"mode": mode, "target_id": target_id, "reason": "測試",
                "assignments": [{"drone_id": d, "role": r} for d, r in zip(IDS, roles)], "formation": {"radius_m": 15.0}}
    return AgentEvent(timestamp_ns=0, plan_id=plan_id, trigger=trigger, planner="rule", latency_ms=1.0, valid=valid,
                      plan=plan, error=error)


def goal(did, n, e, plan_id="p1", d=-25.0):
    return {"drone_id": did, "plan_id": plan_id, "position_world_ned_m": {"n": n, "e": e, "d": d}}


def drone(did, n, e, alt=25.0):
    return DroneState(did, n, e, alt)


def test_invalid_plan_is_rejected_and_not_sent():
    lg = ActionLedger()
    lg.on_event(event(valid=False, plan_id=None, error="assignments 必須讓每架無人機恰好出現一次"), 1.0)
    (r,) = lg.records
    assert r.result == "rejected" and r.drone_id is None and not r.valid and "未送出" in r.why
    assert lg.active == {}


def test_unapplied_plan_is_cancelled():
    lg = ActionLedger()
    lg.on_event(event(plan_id=None), 1.0)
    assert lg.records[0].result == "cancelled" and "套用前" in lg.records[0].why


def test_same_role_continues_one_action_role_change_overrides():
    lg = ActionLedger()
    lg.on_event(event("p1"), 0.0)
    lg.on_event(event("p2"), 2.0)  # 角色沒變：同一筆
    assert len(lg.records) == 2 and lg.active["uav1"].plan_ids == ["p1", "p2"]
    lg.on_event(event("p3", roles=("blocker_left", "chaser")), 4.0)  # 兩架互換
    assert [r.result for r in lg.records[:2]] == ["overridden", "overridden"]
    assert lg.active["uav1"].role == "blocker_left" and len(lg.records) == 4


def test_operator_hold_cancels_running_actions():
    lg = ActionLedger()
    lg.on_event(event("p1"), 0.0)
    lg.on_event(event("p2", mode="hold", roles=("idle", "idle"), trigger="operator", target_id=None), 3.0)
    assert [r.result for r in lg.records[:2]] == ["cancelled", "cancelled"]


def test_static_goal_reached_like_a_move_to():
    lg = ActionLedger(VerifyConfig(max_speed_mps=5.0))
    lg.on_event(event(), 0.0)
    lg.on_goals([goal("uav1", 20.0, 0.0)], drones=[drone("uav1", 0.0, 0.0)], t_s=0.0)
    lg.on_poses([drone("uav1", 18.0, 0.0)], 4.0)
    assert lg.active["uav1"].result is None  # 2 m：還沒到
    lg.on_goals([goal("uav1", 20.0, 0.0)], t_s=4.1)
    lg.on_poses([drone("uav1", 19.0, 0.6, alt=25.5)], 4.1)  # 水平 1.17 m、高度差 0.5 m
    r = lg.active["uav1"]
    assert r.result == "succeeded" and r.time_to_reach_s == pytest.approx(4.1) and "到達目標點" in r.why
    assert r.deadline_s == pytest.approx(max(8.0, 20 / 5 * 1.5 + 5))
    assert [x.action_id for x in lg.newly_resolved] == [r.action_id]


def test_moving_goal_needs_sustained_tracking():
    lg = ActionLedger(VerifyConfig(max_speed_mps=5.0))
    lg.on_event(event(), 0.0)
    t, gn = 0.0, 0.0
    for k in range(40):  # 目標點以 2 m/s 往北移動，無人機落後 3 m
        t, gn = k * 0.1, 10.0 + 2.0 * k * 0.1
        lg.on_goals([goal("uav1", gn, 0.0)], drones=[drone("uav1", gn - 3.0, 0.0)], t_s=t)
        lg.on_poses([drone("uav1", gn - 3.0, 0.0)], t)
        if k == 15:
            assert lg.active["uav1"].result is None  # 已在容許值內，但還沒滿 2 s
    r = lg.active["uav1"]
    assert lg.tolerance_m(r) == pytest.approx(1.5 + 2.0 * 1.0, abs=0.3)
    assert r.result == "succeeded" and "連續 2 s" in r.why


def test_goal_jump_is_not_counted_as_speed():
    lg = ActionLedger()
    lg.on_event(event(), 0.0)
    lg.on_goals([goal("uav1", 0.0, 0.0)], t_s=0.0)
    lg.on_goals([goal("uav1", 8.0, 0.0)], t_s=0.1)  # 讓路橫移 8 m：不是 80 m/s 的移動目標
    assert lg.active["uav1"].goal_speed_mps == 0.0 and lg.tolerance_m(lg.active["uav1"]) == 1.5


def test_deadline_failure():
    lg = ActionLedger(VerifyConfig(max_speed_mps=5.0))
    lg.on_event(event(), 0.0)
    lg.on_goals([goal("uav1", 10.0, 0.0)], drones=[drone("uav1", 0.0, 0.0)], t_s=0.0)
    lg.on_poses([drone("uav1", 0.0, 0.0)], 9.0)  # 期限 = max(8, 10/5×1.5+5) = 8 s
    r = lg.active["uav1"]
    assert r.result == "failed" and "還差 10.0 m" in r.why


def ack(did="uav1", plan_id="p1", accepted=True, adjusted=False, reason=None, fc=None):
    return GoalPointAck(drone_id=did, plan_id=plan_id, accepted=accepted, adjusted=adjusted, reason=reason, flight_command=fc)


def test_acks_flight_command_and_adjusted_status():
    lg = ActionLedger()
    lg.on_event(event(), 0.0)
    lg.on_goals([goal("uav1", 0.0, 0.0)], t_s=0.0)
    fc = {"mavsdk": "offboard.set_position_ned", "north_m": 8.0}
    lg.on_acks([ack(adjusted=True, reason="推開到 6 m", fc=fc), ack()], 0.1)
    lg.on_poses([drone("uav1", 0.5, 0.0)], 0.2)
    r = lg.active["uav1"]
    assert (r.acks, r.accepted, r.adjusted) == (2, 2, 1) and r.flight_command == fc
    assert r.result == "succeeded" and r.status_zh == "被 A 修改"


def test_a_rejecting_everything_marks_rejected():
    lg = ActionLedger()
    lg.on_event(event(), 0.0)
    lg.on_goals([goal("uav1", 30.0, 0.0)], drones=[drone("uav1", 0.0, 0.0)], t_s=0.0)
    lg.on_acks([ack(accepted=False, reason="該架還在地上或起降中")], 0.1)
    lg.on_poses([drone("uav1", 0.0, 0.0)], 0.5)
    assert lg.active["uav1"].result is None
    lg.on_acks([ack(accepted=False, reason="該架還在地上或起降中")], 1.0)
    lg.on_poses([drone("uav1", 0.0, 0.0)], 1.2)
    r = lg.active["uav1"]
    assert r.result == "rejected" and "地上" in r.why


def test_summary_pass_rate_and_export(tmp_path):
    lg = ActionLedger(VerifyConfig(max_speed_mps=5.0))
    lg.on_event(event(valid=False, plan_id=None, error="壞掉"), 0.0)
    lg.on_event(event(), 0.0)
    lg.on_goals([goal("uav1", 0.0, 0.0), goal("uav2", 50.0, 0.0)], drones=[drone("uav1", 0, 0), drone("uav2", 0, 0)], t_s=0.0)
    lg.on_poses([drone("uav1", 0.0, 0.0), drone("uav2", 0.0, 0.0)], 0.1)
    lg.on_poses([drone("uav1", 0.0, 0.0), drone("uav2", 0.0, 0.0)], 30.0)
    lg.finish(30.0)
    s = lg.summary()
    assert s["by_result"]["succeeded"] == 1 and s["by_result"]["failed"] == 1 and s["by_result"]["rejected"] == 1
    assert s["action_pass_rate"] == pytest.approx(1 / 3)
    lg.write(tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "actions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3 and rows[1]["status_zh"] == "通過"
    with open(tmp_path / "actions.csv", encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 3


def test_command_consistency_separates_following_from_ignoring():
    """朝目標點飛 → 一致；往反方向飛 → 不一致。"""
    lg = ActionLedger(VerifyConfig(max_speed_mps=5.0))
    lg.on_event(event(), 0.0)
    for k in range(10):  # uav1 朝北邊的目標點飛；uav2 背對目標點往南飛
        t = k * 0.1
        lg.on_goals([goal("uav1", 50.0, 0.0), goal("uav2", 50.0, 0.0)], drones=[drone("uav1", 0, 0), drone("uav2", 0, 0)], t_s=t)
        lg.on_poses([drone("uav1", 3.0 * t, 0.0), drone("uav2", -3.0 * t, 0.0)], t)
    a, b = lg.active["uav1"], lg.active["uav2"]
    assert a.command_consistency == 1.0 and b.command_consistency == 0.0
    assert lg.summary()["command_consistency"] == pytest.approx(0.5)


# ---------- 多 Agent：每架 Agent 的每個動作一筆 ----------
from llm_agent.team import Team  # noqa: E402
from planner import CaptureZone  # noqa: E402

S = 1_000_000_000
IDS3 = ["uav1", "uav2", "uav3"]
DRONES = [DroneState("uav1", -20, 0, 25), DroneState("uav2", 5, -15, 30), DroneState("uav3", 5, 15, 35)]
ZONE = CaptureZone(35.0, 0.0, 10.0)


def target(n=0.0, e=0.0, status="tracking"):
    return {"target_id": 1, "class": "car", "status": status, "position_world_ned_m": {"n": n, "e": e, "d": 0.0},
            "velocity_world_ned_mps": {"n": 0.0, "e": 0.0, "d": 0.0}, "uncertainty_m": 1.0, "observed_by": ["uav1"],
            "last_seen_ns": 0}


def test_ledger_per_drone_actions():
    team = Team(Commander(IDS3, llm=None, strategy="capture"))
    lg = ActionLedger(per_drone=True)
    evs = team.tick(0, DRONES, [target()], zone=ZONE)
    for ev in evs:
        lg.on_event(ev, 0.0)
    for dev in team.new_drone_events:
        lg.on_drone_event(dev, 0.0)
    assert {r.agent_id for r in lg.records} == {"agent_uav1", "agent_uav2", "agent_uav3"}
    cap = next(r for r in lg.records if r.role == "capturer")
    assert cap.action == "ambush" and cap.plan_ids
    # 抓拍 Agent 改成 engage：舊動作 overridden，開新動作
    evs = team.tick(int(0.5 * S), DRONES, [target(n=20.0)], zone=ZONE)
    for dev in team.new_drone_events:
        lg.on_drone_event(dev, 0.5)
    assert cap.result == "overridden" and "engage" in cap.why
    assert lg.active[cap.drone_id].action == "engage"
    # 不合法的各機決策：記 rejected，不影響目前動作
    from llm_agent.drone_agent import DroneAgentEvent
    lg.on_drone_event(DroneAgentEvent(0, cap.drone_id, "agent_x", "capturer", "capture", "periodic", "llm", 900.0, False,
                                      None, None, error="capturer 不能用 push"), 1.0)
    assert lg.records[-1].result == "rejected" and lg.active[cap.drone_id].action == "engage"
    team.close()
