"""端到端 Demo 的整合測試（不開視窗）：規則式與模擬 LLM 兩種指揮官都要能包圍目標。"""
import json
import math

import pytest

from coord_transform.config import load_camera_config, load_error_model
from demo.engine import DemoEngine, is_encircled
from fusion import DroneState
from llm_agent.commander import Commander
from llm_agent.fleet_prompt import summarize_state  # noqa: F401
from llm_agent.rule_planner import rule_plan
from sim2d import RabbitConfig

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAM = load_camera_config(ROOT / "config" / "camera.yaml")
ERR = load_error_model(ROOT / "config" / "error_model.yaml")
IDS = ["uav1", "uav2", "uav3"]


def test_is_encircled_geometry():
    ring = [DroneState(f"u{k}", 15 * math.cos(a), 15 * math.sin(a), 30) for k, a in enumerate((0, 2.1, 4.2))]
    assert is_encircled(ring, 0.0, 0.0, 15.0)
    bunched = [DroneState(f"u{k}", 15 * math.cos(a), 15 * math.sin(a), 30) for k, a in enumerate((0, 0.3, 0.6))]
    assert not is_encircled(bunched, 0.0, 0.0, 15.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rule_commander_encircles_target(seed):
    c = Commander(IDS, llm=None, period_s=2.0)
    eng = DemoEngine(CAM, ERR, c, rabbit=RabbitConfig(seed=seed), seed=seed)
    for _ in range(600):  # 60 秒
        eng.step()
    m = eng.summary()
    assert m["time_to_encircle_s"] is not None and m["time_to_encircle_s"] < 60
    assert m["visible_fraction"] > 0.5
    assert m["min_drone_distance_m"] > 4.0
    assert all(e.valid for e in c.events)


class StateAwareFakeLLM:
    """假 LLM：讀取 prompt 裡的世界狀態，用規則式算出計畫後以 JSON 回傳（模擬「會思考的」LLM 回應格式）。"""

    name = "fake-llm"

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages):
        self.calls += 1
        text = messages[-1]["content"]
        state = json.loads(text[text.index("{"): text.rindex("}") + 1])
        drones = [DroneState(d["drone_id"], d["n"], d["e"], d["alt_m"]) for d in state["drones"]]
        targets = [
            {"target_id": t["target_id"], "status": t["status"], "uncertainty_m": t["uncertainty_m"],
             "position_world_ned_m": {"n": t["n"], "e": t["e"]},
             "velocity_world_ned_mps": {"n": t["speed_mps"] * math.cos(math.radians(t["heading_deg"] or 0)),
                                        "e": t["speed_mps"] * math.sin(math.radians(t["heading_deg"] or 0))}}
            for t in state["targets"]
        ]
        return "好的：\n" + rule_plan(drones, targets).model_dump_json()


def test_llm_commander_drives_the_fleet_end_to_end():
    llm = StateAwareFakeLLM()
    c = Commander(IDS, llm=llm, period_s=2.0)
    eng = DemoEngine(CAM, ERR, c, rabbit=RabbitConfig(seed=0), seed=0)
    for _ in range(600):
        eng.step()
        c.wait_idle()  # 測試中讓背景請求在下一步前完成，結果才可重現
    c.close()
    m = eng.summary()
    assert llm.calls > 10 and m["llm_requests"] > 10
    assert m["llm_valid_rate"] == 1.0
    assert m["time_to_encircle_s"] is not None


def test_demo_cli_headless_writes_outputs(tmp_path):
    from demo.run import main

    m = main(["--headless", "--duration", "5", "--out", str(tmp_path)])
    assert (tmp_path / "metrics.json").exists() and (tmp_path / "events.jsonl").exists()
    assert m["duration_s"] == pytest.approx(5.0)


def test_scripted_operator_commands_take_effect_and_hold_persists():
    from demo.run import feed_scripted, parse_scripted

    scripted = parse_scripted(["30=先懸停", "10=三架包圍那台車"])
    assert scripted == [(10.0, "三架包圍那台車"), (30.0, "先懸停")]
    c = Commander(IDS, llm=None, period_s=2.0)
    eng = DemoEngine(CAM, ERR, c, rabbit=RabbitConfig(seed=0), seed=0)
    for _ in range(600):  # 60 秒
        feed_scripted(eng, scripted)
        eng.step()
    ops = [e for e in c.events if e.trigger == "operator"]
    # 第 10 秒還沒看到車：「包圍」只能先搜尋，且不鎖定，之後發現車就自動包圍
    assert [e.plan["mode"] for e in ops] == ["search", "hold"]
    assert any(e.plan and e.plan["mode"] == "encircle" and e.trigger != "operator" for e in c.events)
    assert all(e.timestamp_ns <= ops[-1].timestamp_ns for e in c.events)  # hold 之後 Agent 不再自己出計畫
    assert c.current.mode == "hold" and c.paused and scripted == []


def test_parse_scripted_rejects_bad_format():
    from demo.run import parse_scripted

    with pytest.raises(SystemExit):
        parse_scripted(["沒有等號"])
