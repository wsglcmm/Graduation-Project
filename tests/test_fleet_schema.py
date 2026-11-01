"""多機 Plan schema 與指揮官 prompt 的測試。"""
import json

import pytest

from fleet_fixtures import DRONES, ENC, IDS, S, target  # noqa: F401
from llm_agent.fleet_prompt import FEW_SHOT, SYSTEM_PROMPT, build_fleet_messages, summarize_state
from llm_agent.fleet_schema import MODES, ROLES, parse_plan


# ---------- schema ----------
def test_valid_plan_parses_with_prose_and_fence():
    r = parse_plan("好的：\n```json\n" + json.dumps(ENC) + "\n```", IDS, [1])
    assert r.ok and r.plan.mode == "encircle"


@pytest.mark.parametrize(
    "bad, msg",
    [
        ({**ENC, "assignments": ENC["assignments"][:2]}, "恰好出現一次"),
        ({**ENC, "assignments": ENC["assignments"] + [{"drone_id": "uav1", "role": "idle"}]}, "恰好出現一次"),
        ({**ENC, "target_id": None}, "必須指定 target_id"),
        ({**ENC, "target_id": 9}, "不存在"),
        ({**ENC, "mode": "attack"}, "mode"),
        ({**ENC, "formation": {"radius_m": 2.0}}, "radius_m"),
        ({**ENC, "formation": {"radius_m": "15"}}, "radius_m"),
        ({**ENC, "reason": ""}, "reason"),
        ({**ENC, "reason": "很長" * 50}, "reason"),
        ({**ENC, "waypoints": [[0, 0]]}, "waypoints"),  # LLM 不得自己塞座標
        ({**ENC, "assignments": [{"drone_id": "uav1", "role": "kamikaze"}] + ENC["assignments"][1:]}, "role"),
    ],
)
def test_invalid_plans_rejected(bad, msg):
    r = parse_plan(json.dumps(bad, ensure_ascii=False), IDS, [1])
    assert not r.ok and msg in r.error


@pytest.mark.parametrize("text", ["", "不知道", "{bad", json.dumps(ENC) + json.dumps(ENC), "[1]"])
def test_garbage_rejected(text):
    assert not parse_plan(text, IDS, [1]).ok


def test_search_without_target_is_valid():
    plan = {"mode": "search", "target_id": None, "assignments": [{"drone_id": i, "role": "searcher"} for i in IDS], "reason": "找"}
    assert parse_plan(json.dumps(plan), IDS, []).ok


# ---------- prompt ----------
@pytest.mark.parametrize("state, cmd, expected", FEW_SHOT)
def test_few_shot_examples_are_valid_plans(state, cmd, expected):
    ids = [d["drone_id"] for d in state["drones"]]
    tids = [t["target_id"] for t in state["targets"] if t["status"] != "lost"]
    assert parse_plan(json.dumps(expected, ensure_ascii=False), ids, tids).ok


def test_prompt_lists_every_mode_and_role():
    assert all(m in SYSTEM_PROMPT for m in MODES) and all(r in SYSTEM_PROMPT for r in ROLES)


def test_messages_alternate_and_end_with_state():
    state = summarize_state(DRONES, [target()], 12 * S)
    msgs = build_fleet_messages(state, "包圍那台車")
    assert [m["role"] for m in msgs] == ["user", "assistant"] * len(FEW_SHOT) + ["user"]
    assert "包圍那台車" in msgs[-1]["content"] and '"uav3"' in msgs[-1]["content"]


def test_state_summary_heading_and_seconds_since_seen():
    s = summarize_state(DRONES, [target(vn=0.0, ve=3.0, seen=10 * S)], 12 * S)
    t = s["targets"][0]
    assert t["heading_deg"] == 90 and t["seconds_since_seen"] == 2.0
