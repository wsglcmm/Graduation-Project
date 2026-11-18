"""多 Agent：各機 Agent 的角色白名單、規則式決策、LLM 路徑與備援、團隊與動作日誌。"""
import json

import pytest

from fusion import DroneState
from llm_agent.commander import Commander
from llm_agent.drone_actions import ROLE_ACTIONS, parse_drone_action
from llm_agent.drone_agent import DroneAgent, DroneContext, rule_action
from llm_agent.llm_client import ScriptedClient
from llm_agent.team import Team
from planner import CaptureZone, FormationPlanner

IDS = ["uav1", "uav2", "uav3"]
DRONES = [DroneState("uav1", -20, 0, 25), DroneState("uav2", 5, -15, 30), DroneState("uav3", 5, 15, 35)]
ZONE = CaptureZone(35.0, 0.0, 10.0)
S = 1_000_000_000


def target(n=0.0, e=0.0, status="tracking"):
    return {"target_id": 1, "class": "car", "status": status, "position_world_ned_m": {"n": n, "e": e, "d": 0.0},
            "velocity_world_ned_mps": {"n": 0.0, "e": 0.0, "d": 0.0}, "uncertainty_m": 1.0, "observed_by": ["uav1"],
            "last_seen_ns": 0}


def ctx(role, drone=DRONES[0], tgt=None, zone=ZONE, mode="capture"):
    return DroneContext(drone, role, mode, tgt if tgt is not None else target(), zone)


# ---------- 角色白名單 ----------
def test_role_whitelist_rejects_other_roles_actions():
    assert parse_drone_action('{"action": "engage", "reason": "迎上去"}', "capturer").ok
    res = parse_drone_action('{"action": "engage", "reason": "我也想抓拍"}', "herder")
    assert not res.ok and "herder 不能用 engage" in res.error
    assert "不符合動作格式" in parse_drone_action('{"action": "fly_to", "n": 3}', "capturer").error
    assert all("hold" in v for v in ROLE_ACTIONS.values())  # 每個角色都能選最保守的懸停


# ---------- 規則式 ----------
def test_rule_capturer_hysteresis_matches_planner():
    assert rule_action(ctx("capturer", tgt=target(n=0.0)), None).action == "ambush"  # 離捕捉區 35 m
    assert rule_action(ctx("capturer", tgt=target(n=20.0)), "ambush").action == "engage"  # 15 m
    assert rule_action(ctx("capturer", tgt=target(n=10.0)), "engage").action == "engage"  # 25 m：繼續追
    assert rule_action(ctx("capturer", tgt=target(n=10.0)), "ambush").action == "ambush"  # 25 m：沒迎上去過就繼續等
    assert rule_action(ctx("capturer", tgt=target(n=0.0)), "engage").action == "ambush"  # 35 m：放棄


def test_rule_herder_circles_when_on_wrong_side():
    assert rule_action(ctx("herder", drone=DroneState("uav1", -20, 0, 25)), None).action == "push"
    assert rule_action(ctx("herder", drone=DroneState("uav1", 10, 3, 25)), None).action == "circle_behind"
    assert rule_action(ctx("herder", tgt=target(status="lost")), None).action == "hold"


def test_rule_agent_emits_event_only_when_action_changes():
    ag = DroneAgent("uav3")
    evs = ag.tick(0, ctx("capturer", tgt=target(n=0.0)))
    assert [(e.action, e.trigger, e.agent_id) for e in evs] == [("ambush", "role_assigned", "agent_uav3")]
    assert ag.tick(S, ctx("capturer", tgt=target(n=1.0))) == []  # 還是 ambush
    evs = ag.tick(2 * S, ctx("capturer", tgt=target(n=20.0)))
    assert [e.action for e in evs] == ["engage"] and evs[0].trigger == "changed"


# ---------- LLM 路徑 ----------
def test_llm_agent_valid_invalid_and_fallback():
    llm = ScriptedClient(['{"action": "engage", "reason": "目標快到捕捉區"}', '{"action": "push", "reason": "越權"}'])
    ag = DroneAgent("uav3", llm=llm, period_s=1.0)
    evs = ag.tick(0, ctx("capturer", tgt=target(n=20.0)))  # LLM 還沒回來 → 先用規則式
    assert evs[0].planner == "rule" and "等待 LLM" in evs[0].error
    ag.wait_idle()
    evs = ag.tick(S // 2, ctx("capturer", tgt=target(n=20.0)))
    assert (evs[0].planner, evs[0].valid, evs[0].action) == ("llm", True, "engage")
    ag.tick(2 * S, ctx("capturer", tgt=target(n=20.0)))  # 第二次詢問
    ag.wait_idle()
    evs = ag.tick(2 * S + 1, ctx("capturer", tgt=target(n=20.0)))
    assert not evs[0].valid and "capturer 不能用 push" in evs[0].error
    assert ag.current.action == "engage"  # 不合法：沿用上一個合法動作
    msg = json.loads(llm.calls[0][0]["content"].split("\n")[1])
    assert msg["capture_zone"]["target_to_zone_m"] == 15.0 and msg["role"] == "capturer"
    ag.close()


def test_role_change_drops_old_action():
    ag = DroneAgent("uav1")
    ag.tick(0, ctx("capturer", tgt=target(n=20.0)))
    evs = ag.tick(S, ctx("herder"))
    assert evs[0].role == "herder" and evs[0].action in ROLE_ACTIONS["herder"]


# ---------- 團隊 + 規劃器 + 動作日誌 ----------
def test_team_plans_roles_then_actions_and_planner_uses_them():
    team = Team(Commander(IDS, llm=None, strategy="capture"))
    team.tick(0, DRONES, [target()], zone=None)
    acts = team.actions()
    roles = {a.drone_id: a.role for a in team.current.assignments}
    assert set(acts) == set(IDS) and all(acts[d] in ROLE_ACTIONS[roles[d]] for d in IDS)
    fp = FormationPlanner()
    gps = fp.goal_points(team.current, team.current_id, DRONES, [target()], 0, acts)
    assert len(gps) == 3
    held = fp.goal_points(team.current, team.current_id, DRONES, [target()], 1, {"uav1": "hold"})
    g1 = next(g for g in held if g["drone_id"] == "uav1")["position_world_ned_m"]
    assert (g1["n"], g1["e"]) == (-20, 0)  # hold：懸停在原地
    team.close()
