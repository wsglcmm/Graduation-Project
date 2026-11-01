"""規則式指揮官、LLM 客戶端工具與背景指揮迴圈的測試（LLM 一律用 ScriptedClient 模擬）。"""
import json

import pytest

from fleet_fixtures import DRONES, ENC, IDS, S, target  # noqa: F401
from llm_agent.commander import Commander
from llm_agent.fleet_schema import parse_plan
from llm_agent.llm_client import ScriptedClient, load_dotenv
from llm_agent.rule_planner import rule_plan


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nFOO_A=1\nFOO_B='two'\n", encoding="utf-8")
    monkeypatch.setenv("FOO_A", "keep")
    monkeypatch.delenv("FOO_B", raising=False)
    load_dotenv(f)
    import os

    assert os.environ["FOO_A"] == "keep" and os.environ["FOO_B"] == "two"




# ---------- 規則式 ----------
def test_rule_search_when_no_target():
    p = rule_plan(DRONES, [])
    assert p.mode == "search" and {a.role for a in p.assignments} == {"searcher"}


def test_rule_track_when_far_then_encircle_when_close():
    assert rule_plan(DRONES, [target(n=200)]).mode == "track"
    p = rule_plan(DRONES, [target()])
    assert p.mode == "encircle"
    roles = {a.drone_id: a.role for a in p.assignments}
    # 目標往北：uav1 在後方 → chaser；uav2 在西（左前）、uav3 在東（右前）
    assert roles == {"uav1": "chaser", "uav2": "blocker_left", "uav3": "blocker_right"}


@pytest.mark.parametrize("cmd, mode", [("請返航", "return_home"), ("先懸停", "hold"), ("三架包圍它", "encircle"), ("幫我拍照", "hold")])
def test_rule_follows_operator_keywords(cmd, mode):
    assert rule_plan(DRONES, [target(n=200)], cmd).mode == mode


def test_rule_plans_always_pass_validation():
    for tg in ([], [target()], [target(n=300)], [target(status="lost")]):
        p = rule_plan(DRONES, tg)
        tids = [t["target_id"] for t in tg if t["status"] != "lost"]
        assert parse_plan(p.model_dump_json(), IDS, tids).ok


# ---------- 指揮迴圈 ----------
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 0.5
        return self.t


def test_rule_commander_periodic_and_operator_triggers():
    c = Commander(IDS, llm=None, period_s=2.0, clock=FakeClock())
    ev = c.tick(0, DRONES, [target()])
    assert ev[0].trigger == "target_found" and ev[0].planner == "rule" and ev[0].valid
    assert c.tick(1 * S, DRONES, [target()]) == []
    assert c.tick(2 * S, DRONES, [target()])[0].trigger == "periodic"
    c.command("懸停")
    ev = c.tick(int(2.1 * S), DRONES, [target()])
    assert ev[0].trigger == "operator" and c.current.mode == "hold"


def test_llm_commander_accepts_valid_plan_in_background():
    llm = ScriptedClient([json.dumps(ENC)])
    c = Commander(IDS, llm=llm, period_s=2.0, clock=FakeClock())
    assert c.tick(0, DRONES, [target()]) == []  # 已送出，結果還沒回來
    c.wait_idle()
    ev = c.tick(S // 10, DRONES, [target()])
    assert ev[0].planner == "llm" and ev[0].valid and c.current.mode == "encircle" and ev[0].latency_ms > 0
    assert c.current_id == ev[0].plan_id
    sent = llm.calls[0][-1]["content"]
    assert '"uav1"' in sent and '"target_id": 1' in sent
    c.close()


def test_invalid_llm_output_keeps_previous_plan():
    llm = ScriptedClient([json.dumps(ENC), json.dumps({**ENC, "target_id": 99})])
    c = Commander(IDS, llm=llm, period_s=1.0, clock=FakeClock())
    c.tick(0, DRONES, [target()]); c.wait_idle(); c.tick(1, DRONES, [target()])
    first = c.current_id
    c.tick(2 * S, DRONES, [target()]); c.wait_idle()
    ev = c.tick(2 * S + 1, DRONES, [target()])
    assert not ev[0].valid and "不存在" in ev[0].error
    assert c.current_id == first  # 沿用上一個合法計畫
    c.close()


def test_llm_exception_with_no_plan_falls_back_to_rule():
    c = Commander(IDS, llm=ScriptedClient([TimeoutError("逾時")]), clock=FakeClock())
    c.tick(0, DRONES, [target()]); c.wait_idle()
    ev = c.tick(1, DRONES, [target()])
    assert [e.planner for e in ev] == ["llm", "rule"]
    assert not ev[0].valid and "TimeoutError" in ev[0].error
    assert ev[1].valid and c.current is not None
    c.close()


def test_target_lost_triggers_immediately():
    c = Commander(IDS, llm=None, period_s=100.0, clock=FakeClock())
    c.tick(0, DRONES, [target()])
    ev = c.tick(S, DRONES, [target(status="lost")])
    assert ev[0].trigger == "target_lost" and c.current.mode == "search"
