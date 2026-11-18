"""網頁版 Demo：狀態內容、計畫去重、MAVSDK 指令文字、HTTP / SSE 端點。"""
import json
import threading
import urllib.request

import pytest

from coord_transform.config import load_camera_config, load_error_model
from demo.engine import DemoEngine
from demo.run import ROOT, make_commander
from demo.web import WebDemo
from demo.web_state import _flight_command_text, build_state
from sim2d import EvaderConfig

CAM = load_camera_config(ROOT / "config" / "camera.yaml")
ERR = load_error_model(ROOT / "config" / "error_model.yaml")


def engine(agents="multi", steps=0):
    eng = DemoEngine(CAM, ERR, make_commander("rule", 2.0, "capture", agents), seed=0, evader=EvaderConfig(seed=0),
                     drone_max_speed_mps=5.0)
    for _ in range(steps):
        eng.step()
    return eng


def test_state_has_every_panel():
    eng = engine(steps=400)  # 40 s：已發現目標、開始捕捉
    s = build_state(eng)
    json.dumps(s, ensure_ascii=False)  # 必須可序列化
    assert s["agents"] == "multi" and s["mode"] == "capture" and s["zone"] is not None
    assert {d["role"] for d in s["drones"]} == {"herder", "capturer", "blocker"}
    assert all(len(d["fov"]) == 4 and d["action"] for d in s["drones"])
    assert s["captures"] and s["metrics"]["time_to_capture_s"] is not None
    acts = [a for a in s["actions"] if a["drone_id"]]
    assert acts and all(a["agent_id"].startswith("agent_") for a in acts)
    assert any(a["status_zh"] == "通過" for a in acts)
    eng.commander.close()


def test_plan_feed_skips_identical_periodic_plans():
    eng = engine(steps=400)
    s = build_state(eng)
    assert len(eng.commander.events) == 20  # 每 2 秒一次
    assert len(s["plans"]) == 2  # 只列出「搜尋」與「驅趕抓拍」兩次改變
    eng.commander.close()


def test_flight_command_text():
    fc = {"mavsdk": "offboard.set_position_ned", "north_m": 7.321, "east_m": 0.038, "down_m": -30.0, "also_called": ["offboard.start"]}
    assert _flight_command_text(fc) == "offboard.set_position_ned(N +7.3, E +0.0, D -30.0) ＋ offboard.start"
    assert "讓路中" in _flight_command_text({"mavsdk": None, "deferred": "yield"})
    assert _flight_command_text(None) is None


@pytest.fixture
def server():
    eng = engine()
    demo = WebDemo(eng, speed=50.0, duration_s=5.0).start()
    srv = demo.make_server(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield demo, f"http://127.0.0.1:{srv.server_address[1]}"
    demo.stop()
    srv.shutdown()
    srv.server_close()
    eng.commander.close()


def test_http_endpoints(server):
    demo, base = server
    html = urllib.request.urlopen(base + "/").read().decode("utf-8")
    assert "即時指令翻譯" in html and "EventSource" in html
    req = urllib.request.Request(base + "/command", data=json.dumps({"text": "全部懸停"}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    assert json.loads(urllib.request.urlopen(req).read())["ok"]
    demo.finished.wait(5)
    s = json.loads(urllib.request.urlopen(base + "/state").read())
    assert s["directive"] == "全部懸停" and s["paused"]
    with urllib.request.urlopen(base + "/events") as r:
        assert r.readline().startswith(b"data: {")
    bad = urllib.request.Request(base + "/command", data=b'{"text": "  "}', headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError):
        urllib.request.urlopen(bad)


def test_video_range_and_calibration(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(bytes(range(256)) * 4)  # 1024 bytes 假影片
    out = tmp_path / "video_ground_road01.json"
    eng = engine()
    demo = WebDemo(eng, speed=50.0, duration_s=1.0, video_file=video, ground_out=out).start()
    srv = demo.make_server(port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        req = urllib.request.Request(base + "/video", headers={"Range": "bytes=100-199"})
        with urllib.request.urlopen(req) as r:
            assert r.status == 206 and r.headers["Content-Range"] == "bytes 100-199/1024" and r.read() == bytes(range(100, 200))
        assert "地面標定" in urllib.request.urlopen(base + "/calibrate?video_id=road01").read().decode("utf-8")
        pts = [{"pixel": [0, 0], "ground_ne_m": [10, 0]}, {"pixel": [100, 0], "ground_ne_m": [10, 10]},
               {"pixel": [100, 100], "ground_ne_m": [0, 10]}, {"pixel": [0, 100], "ground_ne_m": [0, 0]},
               {"pixel": [50, 50], "ground_ne_m": [5, 5]}]
        body = json.dumps({"video_id": "road01", "points": pts, "save": True}).encode("utf-8")
        res = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/calibration?x=1", data=body)).read())
        assert res["ok"] and res["rms_error_m"] == pytest.approx(0.0, abs=1e-9) and out.exists()
        from coord_transform.homography import load_video_ground

        assert load_video_ground(out).pixel_to_ground(100, 0) == pytest.approx((10.0, 10.0))
    finally:
        demo.stop()
        srv.shutdown()
        srv.server_close()
        eng.commander.close()


def test_video_scene_relabels_roles():
    eng = engine(steps=400)
    s = build_state(eng, video={"src": "/video", "width": 1920, "height": 1080, "ground_to_pixel": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    assert s["mode_zh"] == "預測攔截抓拍" and s["video"]["t"] == 0.0
    assert {d["role_zh"] for d in s["drones"]} == {"尾隨", "攔截抓拍", "側翼"}
    assert all(r["role_zh"] != "驅趕" for r in s["actions"])
    eng.commander.close()
