"""動作驗證、解析與 prompt 的測試。核心原則：LLM 輸出不合法時，一律不得轉給飛控。"""
import json

import pytest

from llm_agent import SCHEMA_VERSION, SYSTEM_PROMPT, build_messages, build_request, parse_llm_output
from llm_agent.prompts import FEW_SHOT, prompt_mentions_all_actions


def ok(obj_or_text):
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
    r = parse_llm_output(text)
    assert r.ok, r.error
    return r.action


def rejected(obj_or_text):
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text)
    r = parse_llm_output(text)
    assert not r.ok and r.action is None and r.error
    return r.error


FOLLOW = {"action": "follow", "params": {"track_id": 7, "distance_m": 15.0, "altitude_m": 30.0}}


# ---------- 合法輸出 ----------
@pytest.mark.parametrize("targets, command, expected", FEW_SHOT)
def test_every_few_shot_example_is_valid(targets, command, expected):
    assert ok(expected).action == expected["action"]


def test_int_is_accepted_for_float_fields():
    assert ok({"action": "follow", "params": {"track_id": 7, "distance_m": 15, "altitude_m": 30}})


def test_hover_and_return_home_params_optional():
    assert ok({"action": "hover"}).action == "hover"
    assert ok({"action": "return_home", "params": {}}).action == "return_home"


def test_code_fence_and_surrounding_prose_tolerated():
    assert ok("好的，這是動作：\n```json\n" + json.dumps(FOLLOW) + "\n```\n以上。").action == "follow"


# ---------- 非法輸出一律拒絕 ----------
@pytest.mark.parametrize(
    "bad",
    [
        {"action": "land", "params": {}},  # 不在白名單
        {"action": "follow", "params": {"track_id": 7, "distance_m": 15, "altitude_m": 30, "speed": 9}},  # params 多餘欄位
        {**FOLLOW, "request_id": "forged"},  # 頂層多餘欄位：request_id 不得來自 LLM
        {"action": "follow", "params": {"track_id": 7, "distance_m": 15}},  # 缺 altitude_m
        {"action": "follow", "params": {"track_id": 7, "distance_m": -5, "altitude_m": 30}},
        {"action": "follow", "params": {"track_id": 7, "distance_m": 15, "altitude_m": 0}},
        {"action": "follow", "params": {"track_id": -1, "distance_m": 15, "altitude_m": 30}},
        {"action": "follow", "params": {"track_id": 7, "distance_m": "15", "altitude_m": 30}},  # 字串數字
        {"action": "follow", "params": {"track_id": True, "distance_m": 15, "altitude_m": 30}},  # bool 不是 int
        {"action": "follow", "params": {"track_id": 7.0, "distance_m": 15, "altitude_m": 30}},  # 浮點 id
        {"action": "hover", "params": {"x": 1}},
        {"action": "set_altitude", "params": {}},
        {"action": "search", "params": {"pattern": "spiral", "center_ned_m": {"n": 0, "e": 0}, "radius_m": 10}},
        {"action": "search", "params": {"pattern": "orbit", "center_ned_m": {"n": 0}, "radius_m": 10}},
        {"params": {}},  # 沒有 action
    ],
)
def test_invalid_actions_are_rejected(bad):
    rejected(bad)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_rejected(token):
    rejected('{"action": "set_altitude", "params": {"altitude_m": %s}}' % token)
    rejected('{"action": "search", "params": {"pattern": "orbit", "center_ned_m": {"n": %s, "e": 0}, "radius_m": 5}}' % token)


@pytest.mark.parametrize(
    "text",
    ["", "   ", "我不知道", "{not json}", '{"action": "hover"', "[1, 2, 3]", json.dumps(FOLLOW) + json.dumps(FOLLOW)],
)
def test_garbage_and_multiple_objects_are_rejected(text):
    rejected(text)


def test_error_message_points_to_the_field():
    err = rejected({"action": "follow", "params": {"track_id": 7, "distance_m": -5, "altitude_m": 30}})
    assert "distance_m" in err


def test_two_actions_error_is_explicit():
    assert "多個" in rejected(json.dumps(FOLLOW) + "\n" + json.dumps({"action": "hover"}))


# ---------- 與規格 §7 的對接 ----------
def test_build_request_matches_spec_example():
    req = build_request(ok(FOLLOW), "a1b2c3")
    assert req == {
        "schema_version": SCHEMA_VERSION,
        "request_id": "a1b2c3",
        "action": "follow",
        "params": {"track_id": 7, "distance_m": 15.0, "altitude_m": 30.0},
    }


def test_build_request_hover_has_empty_params():
    assert build_request(ok({"action": "hover"}), "x1")["params"] == {}


def test_build_request_requires_request_id():
    with pytest.raises(ValueError):
        build_request(ok({"action": "hover"}), "")


# ---------- prompt ----------
def test_prompt_lists_every_schema_action():
    assert prompt_mentions_all_actions()


def test_messages_structure():
    targets = [{"track_id": 7, "class": "car", "position_ned_m": {"n": 1.0, "e": 2.0}}]
    msgs = build_messages("追蹤那台車", targets)
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len(msgs) == 1 + 2 * len(FEW_SHOT) + 1
    assert [m["role"] for m in msgs[1:-1]] == ["user", "assistant"] * len(FEW_SHOT)
    assert msgs[-1]["role"] == "user"
    assert "追蹤那台車" in msgs[-1]["content"] and '"track_id": 7' in msgs[-1]["content"]


def test_few_shot_assistant_turns_are_pure_json():
    for m in build_messages("x", [])[1:-1]:
        if m["role"] == "assistant":
            assert json.loads(m["content"])["action"]
