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
- hold：原地懸停（操作員要求暫停，或情況不明時）
- return_home：返航

# 角色 role：{", ".join(ROLES)}
- chaser：從目標後方追；blocker_left / blocker_right：在目標行進方向的前方左右攔截
- overwatch：在目標正上方較高處持續監看；searcher：搜尋；idle：不動

# 規則
1. assignments 必須列出世界狀態中的每一架無人機，恰好一次。
2. track、encircle 必須指定 target_id，且必須是世界狀態中存在的目標。
3. 目標 status 是 lost 時用 search；predicted 表示暫時看不到，可以繼續原本的模式。
4. 角色要考慮位置：離目標後方最近的當 chaser，位於目標行進方向左前、右前的當 blocker。
5. 操作員指令優先；指令不清楚或做不到時輸出 hold，並在 reason 說明。
6. 若訊息中有「操作員有效指令」，代表操作員先前的指令仍然有效：mode 必須在它允許的範圍內，
   只能調整角色與隊形，不可自行切換到其他模式，直到操作員下新指令。
"""


def _round(x: float, nd: int = 1) -> float:
    return round(float(x), nd)


def summarize_state(drones: Sequence, targets: Sequence[dict], t_ns: int, previous: Optional[dict] = None) -> dict:
    """把世界狀態整理成 Agent 看得懂的精簡 JSON。drones 為 fusion.DroneState，targets 為 FusedTarget dict。"""
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
        _ex_state(_D3, [_T_NORTH]),
        "幫我拍一張照片",
        {"mode": "hold", "target_id": None,
         "assignments": [{"drone_id": "uav1", "role": "idle"}, {"drone_id": "uav2", "role": "idle"}, {"drone_id": "uav3", "role": "idle"}],
         "formation": {"radius_m": 15.0}, "reason": "沒有拍照功能，先懸停等待新指令"},
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
