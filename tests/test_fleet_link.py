"""C 端與 A 的 SITL 連線：用模仿 A 協定的假伺服器測試（不需要學校伺服器）。"""
import json
import math
import socket
import time

import pytest

pytest.importorskip("websockets")

from fleet_link import FleetLink, LinkError, SitlFleet  # noqa: E402
from fleet_link.fake_server import FakeAServer, FakeConfig, FakeFleet  # noqa: E402

FAST = FakeConfig(climb_mps=30.0)  # 測試用：起飛幾乎瞬間完成
IDS = ["uav1", "uav2", "uav3"]


def gp(did, n, e, d=-30.0, **kw):
    return {"schema_version": "0.2.0", "plan_id": "p1", "drone_id": did, "timestamp_ns": 1,
            "position_world_ned_m": {"n": n, "e": e, "d": d}, **kw}


@pytest.fixture
def server():
    with FakeAServer(FAST, port=0) as srv:
        yield srv


@pytest.fixture
def link(server):
    with FleetLink(server.uri, heartbeat_s=0.5) as lk:
        lk.wait_for_poses(IDS)
        yield lk


# ---------- 假伺服器的安全規則（不經網路）----------
def airborne_fleet():
    f = FakeFleet(FAST)
    for d in f.drones:
        d.mode, d.h = "hold", d.layer
    return f


def test_fake_fleet_layers_push_fence_speed():
    f = airborne_fleet()
    a = f.goal_point(gp("uav1", 10, 0, d=-50, max_speed_mps=10))
    assert a["accepted"] and a["adjusted"] and a["applied_position_world_ned_m"]["d"] == -25.0
    assert "高度層 25" in a["reason"] and "上限 3" in a["reason"]
    b = f.goal_point(gp("uav2", 12, 1))  # 距 uav1 目標點 2.2 m → 推開到 6 m
    p = b["applied_position_world_ned_m"]
    assert b["adjusted"] and math.hypot(p["n"] - 10, p["e"] - 0) == pytest.approx(6.0)
    c = f.goal_point(gp("uav3", 200, -5, d=-35))
    assert c["applied_position_world_ned_m"]["n"] == 80.0 and "邊界" in c["reason"]


def test_fake_fleet_rejects_like_a():
    f = FakeFleet(FAST)  # 全部還在地上
    assert "地上" in f.goal_point(gp("uav1", 0, 0))["reason"]
    f = airborne_fleet()
    assert not f.goal_point(gp("uav9", 0, 0))["accepted"]
    assert not f.goal_point({**gp("uav1", 0, 0), "position_world_ned_m": {"n": 1}})["accepted"]
    assert not f.goal_point({**gp("uav1", 0, 0), "schema_version": "0.1.0"})["accepted"]
    f.command({"cmd": "estop"})
    assert "緊急停止" in f.goal_point(gp("uav1", 0, 0))["reason"]


# ---------- 經過 WebSocket ----------
def test_link_receives_validated_poses_and_origins(link):
    assert sorted(link.poses()) == IDS
    assert link.origins()["uav1"] == (-8.0, -6.0, 0.0)
    assert all(r.pose.status == "landed" for r in link.poses().values())
    bad = {"type": "FleetPose", "data": {"drone_id": "uav1", "extra_field": 1}}
    link._on_message(json.dumps(bad))  # 不符合 §10.2 的不會進到 states()
    assert link.stats.bad_poses == 1


def test_takeoff_then_goals_with_ack_stats(link):
    fleet = SitlFleet(link)
    fleet.ensure_airborne(IDS)
    alts = {s.drone_id: s.alt_m for s in fleet.states()}
    assert alts == pytest.approx({"uav1": 25.0, "uav2": 30.0, "uav3": 35.0}, abs=0.6)
    link.send_goals([gp("uav1", 10, 0, d=-25), gp("uav2", 10, 2), gp("uav9", 0, 0)])
    deadline = time.monotonic() + 3
    while link.stats.acks < 3 and time.monotonic() < deadline:
        time.sleep(0.02)
    s = link.stats.summary()
    assert (s["acks"], s["rejected"], s["adjusted"]) == (3, 1, 1)  # uav2 推開；uav9 拒絕；uav1 原樣接受
    assert "推開" in s["last_reason"]["uav2"] and s["ack_rtt_ms_p50"] is not None


def test_goal_timeout_marks_failsafe(link):
    SitlFleet(link).ensure_airborne(IDS)
    link.send_goals([gp("uav1", -5, -6, d=-25)])
    time.sleep(1.5)  # 超過 1 s 沒有新的 GoalPoint
    r = link.poses()["uav1"]
    assert r.pose.status == "failsafe" and r.failsafe_reason == "goal_timeout"
    assert link.poses()["uav2"].pose.status == "flying"


def test_heartbeat_prevents_agent_timeout(server, link):
    SitlFleet(link).ensure_airborne(IDS)
    time.sleep(3.6)  # 超過 A 的 3 s 無訊息懸停門檻；heartbeat 每 0.5 s 一次
    assert server.fleet.timeout_state is None and link.stats.heartbeats >= 5


def test_command_uses_drone_index_and_reports_failure(link):
    assert link.drone_index("uav3") == 2
    link.command("hold", drone_id="uav2")
    with pytest.raises(LinkError, match="未知指令"):
        link.command("fly_to_moon")


def test_connect_failure_is_clear():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]  # 拿一個沒人在聽的埠
    with pytest.raises(LinkError, match="連不上"):
        FleetLink(f"ws://127.0.0.1:{port}", connect_timeout_s=1.0).start()

