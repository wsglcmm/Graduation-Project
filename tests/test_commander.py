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


# ---------- Anthropic 客戶端（不連網：用 httpx2 的 MockTransport 攔截請求）----------
def test_anthropic_client_request_and_text_extraction(monkeypatch):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")  # anthropic 2.x 使用 httpx2
    from llm_agent.llm_client import AnthropicClient

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
            "content": [{"type": "thinking", "thinking": "...", "signature": "x"}, {"type": "text", "text": '{"mode": "hold"}'}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    sdk = anthropic.Anthropic(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    client = AnthropicClient(sdk_client=sdk, effort="low")
    out = client.complete("系統提示", [{"role": "user", "content": "嗨"}])
    assert out == '{"mode": "hold"}'  # thinking block 被略過，只取 text
    body = seen["body"]
    assert body["model"] == "claude-haiku-5-5" and body["system"] == "系統提示"
    assert body["output_config"] == {"effort": "low"}
    assert "temperature" not in body  # Haiku 5.5 文件建議不要送 temperature


def test_anthropic_client_requires_key(monkeypatch):
    pytest.importorskip("anthropic")
    from llm_agent.llm_client import AnthropicClient

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


# ---------- Gemini 客戶端（不連網：用 httpx 的 MockTransport 攔截請求）----------
def test_gemini_client_request_and_text(monkeypatch):
    pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    from google import genai
    from llm_agent.llm_client import GeminiClient

    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": '{"mode": "hold"}'}]}, "finishReason": "STOP"}]})

    sdk = genai.Client(api_key="test-key", http_options={"httpx_client": httpx.Client(transport=httpx.MockTransport(handler))})
    client = GeminiClient(sdk_client=sdk, thinking="")
    out = client.complete("系統提示", [{"role": "user", "content": "例子"}, {"role": "assistant", "content": "{}"}, {"role": "user", "content": "嗨"}])
    assert out == '{"mode": "hold"}'
    body = seen["body"]
    assert "gemini-3.1-flash-lite" in seen["url"]  # 預設主模型
    assert client.last_model == "gemini-3.1-flash-lite"
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]  # assistant → model
    assert body["systemInstruction"]["parts"][0]["text"] == "系統提示"
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_client_requires_key(monkeypatch):
    pytest.importorskip("google.genai")
    from llm_agent.llm_client import GeminiClient

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiClient()


def test_create_llm_client_rejects_unknown_provider():
    from llm_agent.llm_client import create_llm_client

    with pytest.raises(ValueError, match="AGENT_PROVIDER"):
        create_llm_client("openai")


def _gemini_with(handler, **kw):
    pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    from google import genai
    from llm_agent.llm_client import GeminiClient

    sdk = genai.Client(api_key="test-key", http_options={"httpx_client": httpx.Client(transport=httpx.MockTransport(handler))})
    return GeminiClient(sdk_client=sdk, **kw)


def _ok(httpx, text='{"mode": "hold"}'):
    return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}]})


def test_gemini_falls_back_to_next_model_on_503():
    httpx = pytest.importorskip("httpx")
    urls = []

    def handler(request):
        urls.append(str(request.url))
        if "model-a" in str(request.url):
            return httpx.Response(503, json={"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})
        return _ok(httpx)

    client = _gemini_with(handler, model="model-a", fallbacks="model-b", thinking="")
    assert client.complete("s", [{"role": "user", "content": "hi"}]) == '{"mode": "hold"}'
    assert ["model-a" in urls[0], "model-b" in urls[1]] == [True, True]
    assert client.last_model == "model-b"


def test_gemini_does_not_fall_back_on_bad_key():
    httpx = pytest.importorskip("httpx")
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(400, json={"error": {"code": 400, "message": "API key not valid", "status": "INVALID_ARGUMENT"}})

    client = _gemini_with(handler, model="model-a", fallbacks="model-b", thinking="")
    with pytest.raises(Exception, match="API key"):
        client.complete("s", [{"role": "user", "content": "hi"}])
    assert len(urls) == 1  # 金鑰錯誤換模型也沒用，直接回報


def test_gemini_all_models_busy_raises_last_error():
    httpx = pytest.importorskip("httpx")

    def handler(request):
        return httpx.Response(503, json={"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})

    client = _gemini_with(handler, model="model-a", fallbacks="model-b, model-c", thinking="")
    assert client.models == ["model-a", "model-b", "model-c"]
    with pytest.raises(Exception, match="503"):
        client.complete("s", [{"role": "user", "content": "hi"}])


def test_gemini_drops_unsupported_thinking_level():
    httpx = pytest.importorskip("httpx")
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "thinkingConfig" in body.get("generationConfig", {}):
            return httpx.Response(400, json={"error": {"code": 400, "message": "thinking_level MINIMAL is not supported", "status": "INVALID_ARGUMENT"}})
        return _ok(httpx)

    client = _gemini_with(handler, model="model-a", fallbacks="", thinking="minimal")
    assert client.complete("s", [{"role": "user", "content": "hi"}]) == '{"mode": "hold"}'
    client.complete("s", [{"role": "user", "content": "hi"}])
    assert len(bodies) == 3  # 第一次 400 → 不帶 thinking 重送；之後記住，不再送 thinking

