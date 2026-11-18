"""網頁 Demo 每一幀的狀態（純函式，方便測試）：鳥瞰圖、即時指令翻譯、各機角色卡片、動作日誌、抓拍結果、指標。

座標一律 world_ned（n, e，m）；前端畫圖時自己換成螢幕座標（北朝上、東朝右）。
"""
import math
from typing import Optional

from llm_agent.drone_actions import ACTION_ZH

ROLE_ZH = {"chaser": "追擊", "blocker_left": "左翼", "blocker_right": "右翼", "overwatch": "高空監看", "searcher": "搜尋",
           "idle": "待命", "herder": "驅趕", "capturer": "抓拍", "blocker": "封鎖"}
MODE_ZH = {"search": "搜尋", "track": "追蹤", "encircle": "包圍", "capture": "驅趕抓拍", "hold": "懸停", "return_home": "返航"}
TRIGGER_ZH = {"operator": "操作員指令", "periodic": "定期", "target_lost": "目標遺失", "target_found": "發現目標",
              "role_assigned": "接到新角色", "changed": "情況改變"}
# 影片場景的目標不會理無人機、趕不動：同一套 capture 角色，畫面上改用名副其實的名稱（報告也這樣寫）
VIDEO_ROLE_ZH = {"herder": "尾隨", "capturer": "攔截抓拍", "blocker": "側翼"}
VIDEO_MODE_ZH = {"capture": "預測攔截抓拍"}
MAX_PLANS = 12
MAX_ACTIONS = 30


def _r(x, nd=1):
    return None if x is None else round(float(x), nd)


def _plan_row(ev) -> dict:
    p = ev.plan or {}
    return {
        "kind": "plan", "t": _r(ev.timestamp_ns / 1e9), "plan_id": ev.plan_id, "trigger": ev.trigger,
        "trigger_zh": TRIGGER_ZH.get(ev.trigger, ev.trigger), "planner": ev.planner, "latency_ms": _r(ev.latency_ms, 0),
        "valid": ev.valid, "applied": ev.plan_id is not None, "error": ev.error, "operator_command": ev.operator_command,
        "model": ev.model, "mode": p.get("mode"), "mode_zh": MODE_ZH.get(p.get("mode"), p.get("mode")),
        "reason": p.get("reason"),
        "assignments": [{"drone_id": a["drone_id"], "role": a["role"], "role_zh": ROLE_ZH.get(a["role"], a["role"])}
                        for a in p.get("assignments", [])],
    }


def _plan_feed(events) -> list:
    """協調 Agent 的計畫：只列出有意義的（操作員指令、不合法、模式或角色改變），避免每 2 秒重複一樣的計畫洗版。"""
    out, last_key = [], None
    for ev in events:
        p = ev.plan or {}
        key = (p.get("mode"), tuple((a["drone_id"], a["role"]) for a in p.get("assignments", [])))
        if ev.trigger == "operator" or not ev.valid or ev.plan_id is None or key != last_key:
            out.append(_plan_row(ev))
        if ev.valid and ev.plan_id is not None:
            last_key = key
    return out[-MAX_PLANS:]


def _flight_command_text(fc: Optional[dict]) -> Optional[str]:
    """A 實際呼叫的 MAVSDK 函式，轉成一行可讀文字（規格 §10.8(a)）。"""
    if not fc:
        return None
    if fc.get("mavsdk") is None:
        return f"（{fc.get('deferred', '延後')}：讓路中，目標點先記下）"
    args = ", ".join(f"{k[0].upper()} {fc[k]:+.1f}" for k in ("north_m", "east_m", "down_m") if k in fc)
    extra = f" ＋ {', '.join(fc['also_called'])}" if fc.get("also_called") else ""
    return f"{fc['mavsdk']}({args}){extra}"


def _action_row(r) -> dict:
    return {
        "kind": "action", "action_id": r.action_id, "t": _r(r.t_start_s), "t_end": _r(r.t_end_s), "drone_id": r.drone_id,
        "agent_id": r.agent_id, "role": r.role, "role_zh": ROLE_ZH.get(r.role, r.role), "mode": r.mode,
        "action": r.action, "action_zh": ACTION_ZH.get(r.action) if r.action else None, "planner": r.planner,
        "latency_ms": _r(r.latency_ms, 0), "reason": r.reason, "valid": r.valid, "validation_error": r.validation_error,
        "goals_sent": r.goals_sent, "last_goal_ned": r.last_goal_ned, "c_adjustments": r.c_adjustments,
        "acks": r.acks, "accepted": r.accepted, "adjusted": r.adjusted, "rejected": r.rejected,
        "last_ack_reason": r.last_ack_reason, "flight_command": _flight_command_text(r.flight_command),
        "result": r.result, "status_zh": r.status_zh, "why": r.why, "time_to_reach_s": r.time_to_reach_s,
        "consistency": r.command_consistency,
    }


def build_state(engine, backend: str = "sim", link=None, video: Optional[dict] = None) -> dict:
    """engine：demo.engine.DemoEngine；backend：sim（替身）/ sitl（A 的 SITL）；link：fleet_link.FleetLink（sitl 時）。
    video：影片場景的固定資訊（src、寬高、ground → pixel 的 3×3 矩陣、是否暫代 ID…），這裡加上目前的影片時間。"""
    if video is not None:
        return _relabel_video(_build(engine, backend, link, video))
    return _build(engine, backend, link, video)


def _relabel_video(state: dict) -> dict:
    """影片場景：角色與模式改成影片版名稱（只改顯示文字，role / mode 代碼不變，日誌照舊）。"""
    def fix(row):
        if row.get("role") in VIDEO_ROLE_ZH:
            row["role_zh"] = VIDEO_ROLE_ZH[row["role"]]
        if row.get("mode") in VIDEO_MODE_ZH:
            row["mode_zh"] = VIDEO_MODE_ZH[row["mode"]]
    fix(state)
    for d in state["drones"]:
        fix(d)
    for p in state["plans"]:
        fix(p)
        for a in p["assignments"]:
            fix(a)
    for a in state["actions"]:
        fix(a)
    return state


def _build(engine, backend: str, link, video: Optional[dict]) -> dict:
    cmd = engine.commander
    plan = cmd.current
    roles = {a.drone_id: a.role for a in plan.assignments} if plan else {}
    actions = cmd.actions() if engine.multi else {}
    goals = {g["drone_id"]: g["position_world_ned_m"] for g in (engine.goals or [])}
    yielding = link.yielding() if link is not None else {}
    poses = link.poses() if link is not None else {}
    last_action = {}
    for r in engine.ledger.records:
        if r.drone_id:
            last_action[r.drone_id] = r
    drones = []
    for d in engine.sim.states():
        did, role = d.drone_id, roles.get(d.drone_id, "idle")
        rec = last_action.get(did)
        pr = poses.get(did)
        a_status = "替身" if link is None else (pr.pose.status if pr else "—")
        if did in yielding:
            a_status = f"讓路給 {yielding[did]}"
        elif pr is not None and pr.failsafe_reason:
            a_status = f"failsafe（{pr.failsafe_reason}）"
        g = goals.get(did)
        drones.append({
            "id": did, "n": _r(d.n, 2), "e": _r(d.e, 2), "alt": _r(d.alt_m), "yaw": _r(d.yaw_rad, 3),
            "role": role, "role_zh": ROLE_ZH.get(role, role), "action": actions.get(did),
            "action_zh": ACTION_ZH.get(actions.get(did)) if actions.get(did) else None,
            "fov": [[_r(n, 1), _r(e, 1)] for n, e in engine.view.footprint_corners(d)],
            "goal": [_r(g["n"]), _r(g["e"])] if g else None, "a_status": a_status,
            "last": None if rec is None else {"action_id": rec.action_id, "status_zh": rec.status_zh, "result": rec.result,
                                              "why": rec.why, "consistency": rec.command_consistency},
        })
    tgt = engine.target_state
    zone = engine.planner.capture_zone
    caps = engine.metrics.captures
    m = engine.summary()
    devs = getattr(cmd, "drone_events", [])
    vid = None
    if video is not None:
        vid = dict(video, t=round(getattr(engine.rabbit, "video_time_s", 0.0), 3))
    return {
        "video": vid,
        "t": _r(engine.t_ns / 1e9), "backend": backend, "agents": "multi" if engine.multi else "single",
        "planner": cmd.planner_name, "max_speed_mps": engine.planner.cfg.max_speed_mps,
        "mode": plan.mode if plan else None, "mode_zh": MODE_ZH.get(plan.mode) if plan else None,
        "directive": cmd.directive_text, "paused": cmd.paused,
        "arena_radius_m": getattr(engine.rabbit.cfg, "arena_radius_m", 60.0),
        "zone": None if zone is None else {"n": _r(zone.n), "e": _r(zone.e), "r": zone.radius_m},
        "truth": [_r(engine.truth[0], 2), _r(engine.truth[1], 2)] if engine.truth else None,
        "target": None if tgt is None else {
            "status": tgt["status"], "n": _r(tgt["position_world_ned_m"]["n"], 2), "e": _r(tgt["position_world_ned_m"]["e"], 2),
            "unc": _r(tgt["uncertainty_m"], 2), "vn": _r(tgt["velocity_world_ned_mps"]["n"], 2),
            "ve": _r(tgt["velocity_world_ned_mps"]["e"], 2), "observed_by": tgt.get("observed_by", [])},
        "drones": drones,
        "plans": _plan_feed(cmd.events),
        "drone_decisions": [{"t": _r(x.timestamp_ns / 1e9), "drone_id": x.drone_id, "planner": x.planner, "valid": x.valid,
                             "action": x.action, "error": x.error} for x in devs[-20:]],
        "actions": [_action_row(r) for r in engine.ledger.records[-MAX_ACTIONS:]],
        "captures": caps[-5:],
        "metrics": {
            "time_to_capture_s": m["time_to_capture_by_capturer_s"], "captures": m["captures"],
            "zone_entered_at_s": m["zone_entered_at_s"], "visible_fraction": _r(m["visible_fraction"], 3),
            "min_drone_distance_m": _r(m["min_drone_distance_m"] if math.isfinite(m["min_drone_distance_m"]) else None),
            "min_drone_horizontal_m": _r(m["min_drone_horizontal_m"] if m.get("min_drone_horizontal_m") is not None
                                         and math.isfinite(m["min_drone_horizontal_m"]) else None),
            "command_consistency": _r(m["command_consistency"], 3), "action_pass_rate": _r(m["action_pass_rate"], 3),
            "actions": m["actions"]["actions"], "plans": m["plans"],
        },
    }
