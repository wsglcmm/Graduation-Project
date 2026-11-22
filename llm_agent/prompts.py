"""Agent 的 prompt 結構：system prompt + few-shot + 世界狀態 + 操作員指令。

本檔只負責「組出 messages」，不呼叫任何 LLM API（呼叫哪個模型、怎麼呼叫是之後的整合工作）。
few-shot 範例會在測試中逐一通過 schema 驗證，確保 prompt 與 schema 不會悄悄漂移。
"""
import json

from .actions import ACTION_NAMES

# ⚠️ 暫定預設值（取自規格 §7 的範例數字），需與 A 的安全限制、老師確認後才能當正式值。
DEFAULT_DISTANCE_M = 15
DEFAULT_ALTITUDE_M = 30

_ACTION_TABLE = """\
| action | params | 說明 |
|---|---|---|
| follow | track_id(int), distance_m(>0), altitude_m(>0) | 追蹤指定目標，保持距離與高度 |
| hover | 無 | 原地懸停 |
| search | pattern("orbit"|"lawnmower"), center_ned_m({"n","e"}), radius_m(>0) | 目標丟失時的搜尋 |
| set_altitude | altitude_m(>0) | 調整飛行高度 |
| return_home | 無 | 返航 |"""

SYSTEM_PROMPT = f"""\
你是無人機的「任務層規劃員」。你只把操作員的自然語言指令轉成「一個」高階動作 JSON；\
你不控制馬達、不輸出速度或姿態，這些由飛控層負責。

# 規則
1. 只輸出一個 JSON 物件，格式為 {{"action": "...", "params": {{...}}}}，不要有任何其他文字、說明或 Markdown。
2. 只能使用下表的動作與欄位，不得自創動作、不得加多餘欄位。
3. track_id 必須來自使用者訊息中「目前偵測到的目標」清單；清單裡沒有或無法確定是哪一個，就不要猜，輸出 hover。
4. 單位：距離與高度為公尺；座標為 NED（n = 北、e = 東），單位公尺。
5. 操作員沒有指定的數值，使用預設：distance_m = {DEFAULT_DISTANCE_M}，altitude_m = {DEFAULT_ALTITUDE_M}。
6. 指令無法用下表的動作表達、模糊不清，或你對安全有疑慮時，輸出 {{"action": "hover"}}。

# 可用動作
{_ACTION_TABLE}
"""

# (目前偵測到的目標, 操作員指令, 期望的動作)
FEW_SHOT = [
    (
        [{"track_id": 7, "class": "car", "position_ned_m": {"n": 41.8, "e": 6.2}}],
        "追蹤那台車，保持 20 公尺，高度 40 公尺",
        {"action": "follow", "params": {"track_id": 7, "distance_m": 20, "altitude_m": 40}},
    ),
    (
        [
            {"track_id": 7, "class": "car", "position_ned_m": {"n": 41.8, "e": 6.2}},
            {"track_id": 9, "class": "person", "position_ned_m": {"n": 12.0, "e": -3.5}},
        ],
        "跟著那個人",
        {
            "action": "follow",
            "params": {"track_id": 9, "distance_m": DEFAULT_DISTANCE_M, "altitude_m": DEFAULT_ALTITUDE_M},
        },
    ),
    (
        [],
        "追蹤那台車",
        {"action": "hover"},  # 沒有偵測到任何目標：不猜 track_id
    ),
    (
        [],
        "以北 40、東 5 為中心，半徑 25 公尺繞圈搜尋",
        {
            "action": "search",
            "params": {"pattern": "orbit", "center_ned_m": {"n": 40, "e": 5}, "radius_m": 25},
        },
    ),
    ([], "降到 20 公尺", {"action": "set_altitude", "params": {"altitude_m": 20}}),
    ([], "回家", {"action": "return_home"}),
    ([], "幫我拍一張照片", {"action": "hover"}),  # 白名單沒有的動作：輸出 hover
]


def _user_message(targets: list, command: str) -> str:
    return (
        "目前偵測到的目標（NED 座標，單位 m）：\n"
        f"{json.dumps(targets, ensure_ascii=False)}\n\n"
        f"操作員指令：{command}"
    )


def build_messages(command: str, targets: list) -> list:
    """組出送給 LLM 的 messages：system + few-shot（user/assistant 成對）+ 本次請求。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex_targets, ex_command, ex_action in FEW_SHOT:
        messages.append({"role": "user", "content": _user_message(ex_targets, ex_command)})
        messages.append({"role": "assistant", "content": json.dumps(ex_action, ensure_ascii=False)})
    messages.append({"role": "user", "content": _user_message(targets, command)})
    return messages


def prompt_mentions_all_actions() -> bool:
    """prompt 的動作表是否涵蓋 schema 的所有動作（供測試使用）。"""
    return all(f"| {name} |" in SYSTEM_PROMPT for name in ACTION_NAMES)
