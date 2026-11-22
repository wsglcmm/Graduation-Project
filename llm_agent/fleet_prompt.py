"""多機指揮官的 prompt：system + few-shot + 本次世界狀態（規格 §10.5）。

世界狀態只給 Agent「做決策需要的摘要」（各機位置、目標位置與速度、誤差、多久沒看到），
不給原始影像或逐幀資料：一來省 token、降低延遲，二來讓 Agent 留在任務層。
few-shot 會在測試中逐一通過 parse_plan，避免 prompt 和 schema 悄悄不一致。
"""
import json
import math
from typing import List, Optional, Sequence

from .fleet_schema import MODES, ROLES

SYSTEM_PROMPT = f"""\
你是多架無人機的「任務指揮官」，像老鷹群圍捕兔子一樣指揮無人機合力追蹤地面目標。
你只決定「每架扮演什麼角色、用什麼模式與隊形」；每架實際飛到哪裡由確定性的隊形規劃器計算，
怎麼安全地飛由飛控層負責。你不輸出座標、速度或姿態。

# 輸出格式（只輸出一個 JSON 物件，不要有其他文字或 Markdown）
{{"mode": "...", "target_id": 整數或 null, "assignments": [{{"drone_id": "...", "role": "..."}}, ...],
 "formation": {{"radius_m": 5 到 60}}, "reason": "一句話理由，80 字以內"}}

# 模式 mode：{", ".join(MODES)}
- search：目標 lost 或還沒出現，全部用 searcher 在最後已知位置附近搜尋
- track：目標離無人機群很遠，先追上去（chaser 在後方追、其他角色保持在兩側）
- encircle：已追上目標，包圍它（chaser 在後、blocker_left / blocker_right 在前方左右攔截）
- capture：捕捉策略。驅趕機把目標趕進捕捉區，抓拍機在捕捉區等，目標靠近時迎上去讓目標進入相機畫面中央抓拍；
  捕捉區的位置由規劃器決定（機群 → 目標的延長線上、目標前方）
- hold：原地懸停（操作員要求暫停，或情況不明時）
- return_home：返航

# 角色 role：{", ".join(ROLES)}
- chaser：從目標後方追；blocker_left / blocker_right：在目標行進方向的前方左右攔截
- overwatch：在目標正上方較高處持續監看；searcher：搜尋；idle：不動
- herder（驅趕）：繞到目標遠離捕捉區的那一側施壓；capturer（抓拍）：在捕捉區伏擊、最後迎上去抓拍；
  blocker（封鎖）：守住目標往側邊逃的方向。這三個角色只能在 capture 使用

# 規則
1. assignments 必須列出世界狀態中的每一架無人機，恰好一次。
2. track、encircle、capture 必須指定 target_id，且必須是世界狀態 targets 裡存在的 target_id，不可自己編號。
   targets 是空陣列（還沒看到任何目標）時，target_id 一律 null、mode 用 search；即使操作員要求追蹤、包圍或抓拍也一樣，
   系統會記住操作員要的模式，發現目標後再請你照指令分派，reason 說明「先搜尋，找到後執行」。
3. 目標 status 是 lost 時用 search；predicted 表示暫時看不到，可以繼續原本的模式。
4. 角色要考慮位置：離目標後方最近的當 chaser，位於目標行進方向左前、右前的當 blocker。
5. capture 需要恰好一架 capturer、至少一架 herder，只能用 herder / capturer / blocker / overwatch；
   操作員說抓拍、捕捉、驅趕、趕進某處時用 capture。分派角色要看 capture_geometry（由規劃器算好的事實）：
   to_zone_m 最小的當 capturer，ahead_m 最負（在目標後方）的當 herder，剩下的當 blocker；
   blocker 比 overwatch 有用（守住側邊才趕得進捕捉區），除非目標常跟丟才用 overwatch。
   同一次捕捉期間不要無故換角色（無人機要重新飛位置），除非 capture_geometry 顯示原本的分派明顯不合理。
6. 操作員指令優先；指令不清楚或做不到時輸出 hold，並在 reason 說明。
7. 若訊息中有「操作員有效指令」，代表操作員先前的指令仍然有效：mode 必須在它允許的範圍內，
   只能調整角色與隊形，不可自行切換到其他模式，直到操作員下新指令。
"""


def _round(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def capture_geometry(drones: Sequence, target: dict, zone, existing: bool) -> dict:
    """捕捉區與各機相對位置（確定性計算，給 LLM 當事實用，不是答案）。

    ahead_m：沿「目標 → 捕捉區」方向的投影（正 = 在目標和捕捉區之間或更前面，負 = 在目標後方）；
    side_m：側向偏移（正 = 面向捕捉區時的右側）。驅趕機應選 ahead_m 最負的、抓拍機選 to_zone_m 最小的。
    """
    p = target["position_world_ned_m"]
    dn, de = zone.n - p["n"], zone.e - p["e"]
    L = math.hypot(dn, de) or 1.0
    un, ue = dn / L, de / L
    rel = []
    for d in drones:
        rn, re_ = d.n - p["n"], d.e - p["e"]
        rel.append({"drone_id": d.drone_id, "ahead_m": _round(rn * un + re_ * ue), "side_m": _round(-rn * ue + re_ * un),
                    "to_zone_m": _round(math.hypot(d.n - zone.n, d.e - zone.e)), "to_target_m": _round(math.hypot(rn, re_))})
    return {"target_id": target["target_id"], "zone_n": _round(zone.n), "zone_e": _round(zone.e), "radius_m": zone.radius_m,
            "target_to_zone_m": _round(L), "status": "使用中" if existing else "若改用 capture，捕捉區會放在這裡",
            "drones": rel}


def summarize_state(drones: Sequence, targets: Sequence[dict], t_ns: int, previous: Optional[dict] = None,
                    capture: Optional[dict] = None) -> dict:
    """把世界狀態整理成 Agent 看得懂的精簡 JSON。drones 為 fusion.DroneState，targets 為 FusedTarget dict。
    capture：capture_geometry() 的結果（有目標且有捕捉設定時）。"""
    out_targets = []
    for t in targets:
        p, v = t["position_world_ned_m"], t["velocity_world_ned_mps"]
        speed = math.hypot(v["n"], v["e"])
        out_targets.append(
            {
                "target_id": t["target_id"],
                "class": t["class"],
                "status": t["status"],
                "n": _round(p["n"]),
                "e": _round(p["e"]),
                "speed_mps": _round(speed),
                "heading_deg": _round(math.degrees(math.atan2(v["e"], v["n"])) % 360, 0) if speed > 0.3 else None,
                "uncertainty_m": _round(t["uncertainty_m"]),
                "seen_by": t["observed_by"],
                "seconds_since_seen": _round((t_ns - t["last_seen_ns"]) / 1e9),
            }
        )
    return {
        "time_s": _round(t_ns / 1e9),
        "drones": [
            {"drone_id": d.drone_id, "n": _round(d.n), "e": _round(d.e), "alt_m": _round(d.alt_m), "heading_deg": _round(math.degrees(d.yaw_rad) % 360, 0)}
            for d in drones
        ],
        "targets": out_targets,
        "previous_plan": previous,
        **({"capture_geometry": capture} if capture else {}),
    }


def user_message(state: dict, operator_command: Optional[str], standing_order: Optional[str] = None) -> str:
    cmd = operator_command.strip() if operator_command else "（沒有新指令，依目前狀況決定是否調整計畫）"
    msg = f"世界狀態（world_ned 座標，單位 m；n = 北、e = 東）：\n{json.dumps(state, ensure_ascii=False)}\n\n操作員指令：{cmd}"
    if standing_order:
        msg += f"\n操作員有效指令：{standing_order}"
    return msg


def _ex_state(drones, targets, prev=None):
    return {"time_s": 12.0, "drones": drones, "targets": targets, "previous_plan": prev}


_D3 = [
    {"drone_id": "uav1", "n": -20.0, "e": 0.0, "alt_m": 25.0, "heading_deg": 0},
    {"drone_id": "uav2", "n": 5.0, "e": -15.0, "alt_m": 30.0, "heading_deg": 0},
    {"drone_id": "uav3", "n": 5.0, "e": 15.0, "alt_m": 35.0, "heading_deg": 0},
]
_T_NORTH = {"target_id": 1, "class": "car", "status": "tracking", "n": 0.0, "e": 0.0, "speed_mps": 4.0, "heading_deg": 0,
            "uncertainty_m": 0.8, "seen_by": ["uav1", "uav2"], "seconds_since_seen": 0.0}
_T_FAR = {**_T_NORTH, "n": 120.0, "e": 40.0, "seen_by": ["uav3"]}
_T_LOST = {**_T_NORTH, "status": "lost", "n": 30.0, "e": 10.0, "speed_mps": 0.0, "heading_deg": None, "seen_by": [], "seconds_since_seen": 5.0}

# (世界狀態, 操作員指令, 期望輸出)
FEW_SHOT = [
    (
        _ex_state(_D3, [_T_NORTH]),
        "三架一起把那台車包圍起來",
        {"mode": "encircle", "target_id": 1,
         "assignments": [{"drone_id": "uav1", "role": "chaser"}, {"drone_id": "uav2", "role": "blocker_left"}, {"drone_id": "uav3", "role": "blocker_right"}],
         "formation": {"radius_m": 15.0}, "reason": "車往北開，uav1 在後方追，uav2、uav3 在左前、右前攔截"},
    ),
    (
        _ex_state(_D3, [_T_FAR]),
        None,
        {"mode": "track", "target_id": 1,
         "assignments": [{"drone_id": "uav1", "role": "chaser"}, {"drone_id": "uav2", "role": "blocker_left"}, {"drone_id": "uav3", "role": "blocker_right"}],
         "formation": {"radius_m": 25.0}, "reason": "目標離機群約 120 m，先追上再包圍"},
    ),
    (
        _ex_state(_D3, [_T_LOST]),
        None,
        {"mode": "search", "target_id": None,
         "assignments": [{"drone_id": "uav1", "role": "searcher"}, {"drone_id": "uav2", "role": "searcher"}, {"drone_id": "uav3", "role": "searcher"}],
         "formation": {"radius_m": 20.0}, "reason": "目標已 5 秒未出現，在最後位置附近分散搜尋"},
    ),
    (
        {**_ex_state(_D3, [_T_NORTH]), "capture_geometry": {
            "target_id": 1, "zone_n": 35.0, "zone_e": 0.0, "radius_m": 10.0, "target_to_zone_m": 35.0,
            "status": "若改用 capture，捕捉區會放在這裡",
            "drones": [{"drone_id": "uav1", "ahead_m": -20.0, "side_m": 0.0, "to_zone_m": 55.0, "to_target_m": 20.0},
                       {"drone_id": "uav2", "ahead_m": 5.0, "side_m": -15.0, "to_zone_m": 33.5, "to_target_m": 15.8},
                       {"drone_id": "uav3", "ahead_m": 5.0, "side_m": 15.0, "to_zone_m": 33.5, "to_target_m": 15.8}]}},
        "把那台車趕進捕捉區抓拍",
        {"mode": "capture", "target_id": 1,
         "assignments": [{"drone_id": "uav1", "role": "herder"}, {"drone_id": "uav2", "role": "blocker"}, {"drone_id": "uav3", "role": "capturer"}],
         "formation": {"radius_m": 15.0}, "reason": "uav1 在車後方往北趕，uav3 在前方捕捉區等著抓拍，uav2 守西側"},
    ),
    (
        _ex_state(_D3, []),
        "把那台車趕進捕捉區抓拍",
        {"mode": "search", "target_id": None,
         "assignments": [{"drone_id": "uav1", "role": "searcher"}, {"drone_id": "uav2", "role": "searcher"}, {"drone_id": "uav3", "role": "searcher"}],
         "formation": {"radius_m": 20.0}, "reason": "還沒看到任何車，先分散搜尋，找到後再驅趕抓拍"},
    ),
    (
        _ex_state(_D3, [_T_NORTH]),
        "降落在那棟樓的屋頂上",
        {"mode": "hold", "target_id": None,
         "assignments": [{"drone_id": "uav1", "role": "idle"}, {"drone_id": "uav2", "role": "idle"}, {"drone_id": "uav3", "role": "idle"}],
         "formation": {"radius_m": 15.0}, "reason": "沒有指定地點降落的功能，先懸停等待新指令"},
    ),
]


def build_fleet_messages(state: dict, operator_command: Optional[str] = None, standing_order: Optional[str] = None) -> List[dict]:
    """messages（不含 system；system 另外傳給 API）：few-shot 的 user/assistant 成對，加上本次請求。"""
    msgs = []
    for ex_state, ex_cmd, ex_plan in FEW_SHOT:
        msgs.append({"role": "user", "content": user_message(ex_state, ex_cmd)})
        msgs.append({"role": "assistant", "content": json.dumps(ex_plan, ensure_ascii=False)})
    msgs.append({"role": "user", "content": user_message(state, operator_command, standing_order)})
    return msgs
