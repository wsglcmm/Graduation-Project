"""規則式指揮官：不用 LLM 的確定性備援，也是報告裡和 LLM 比較的基準線（baseline）。

用途：
    1. LLM 沒有金鑰、斷網、逾時或輸出不合法時，系統仍然有合法計畫可用。
    2. 離線測試與 CI 不需要呼叫 API。
    3. 報告可以比較「規則 vs LLM」在包圍完成時間、指令合法率上的差異，這比只展示 LLM 更有說服力。
"""
import itertools
import math
from typing import Optional, Sequence

from planner.capture import CaptureConfig, propose_zone

from .fleet_schema import Plan

ENCIRCLE_RANGE_FACTOR = 2.5  # 機群中心離目標 < 半徑 × 這個倍數時，從 track 切換到 encircle


def _slot_offsets(heading: float, radius: float):
    """各角色在目標周圍的位置（相對目標，world n/e）。chaser 在正後方、兩翼在前方 ±60°（三架均分 120°）。"""
    out = {}
    for role, ang in (("chaser", math.pi), ("blocker_left", -math.pi / 3), ("blocker_right", math.pi / 3)):
        a = heading + ang
        out[role] = (radius * math.cos(a), radius * math.sin(a))
    return out


def _parse_operator(cmd: Optional[str]) -> Optional[str]:
    if not cmd:
        return None
    # 捕捉關鍵字放在「追 / 跟」前面：「跟上去抓拍」是捕捉，不是跟隨
    table = (("返航", "return_home"), ("回家", "return_home"), ("懸停", "hold"), ("停", "hold"),
             ("抓拍", "capture"), ("捕捉", "capture"), ("驅趕", "capture"), ("趕", "capture"),
             ("包圍", "encircle"), ("圍", "encircle"), ("搜尋", "search"), ("找", "search"), ("追", "track"), ("跟", "track"))
    for key, mode in table:
        if key in cmd:
            return mode
    return "hold"  # 看不懂的指令：保守懸停


def _vel(tgt: dict):
    v = tgt.get("velocity_world_ned_mps") or {"n": 0.0, "e": 0.0}
    return v["n"], v["e"]


def capture_assignment(drones: Sequence, tn: float, te: float, cfg: CaptureConfig = CaptureConfig(), vel=None) -> list:
    """捕捉策略的角色分派：用和隊形規劃器相同的方法估捕捉區，
    再窮舉排列，選「抓拍機飛到捕捉區 + 驅趕機飛到目標後方 + 封鎖機飛到側面」總距離最短的。"""
    zone = propose_zone(tn, te, drones, cfg, vel=vel)
    un, ue = tn - zone.n, te - zone.e
    r = math.hypot(un, ue) or 1.0
    un, ue = un / r, ue / r
    slots = {"capturer": (zone.n, zone.e),
             "herder": (tn + un * cfg.herd_distance_m, te + ue * cfg.herd_distance_m),
             "blocker": (tn - ue * cfg.block_distance_m, te + un * cfg.block_distance_m)}
    roles = ["capturer", "herder", "blocker"] + ["overwatch"] * max(0, len(drones) - 3)
    if len(drones) == 2:
        roles = ["capturer", "herder"]
    best = min(itertools.permutations(roles[: len(drones)]),
               key=lambda perm: sum(math.hypot(d.n - slots.get(ro, (tn, te))[0], d.e - slots.get(ro, (tn, te))[1])
                                    for d, ro in zip(drones, perm)))
    return [{"drone_id": d.drone_id, "role": ro} for d, ro in zip(drones, best)]


def rule_plan(drones: Sequence, targets: Sequence[dict], operator_command: Optional[str] = None, radius_m: float = 15.0,
              strategy: str = "encircle", previous: Optional[Plan] = None, capture_cfg: Optional[CaptureConfig] = None) -> Plan:
    """strategy：沒有操作員指令時的預設策略，encircle（包圍）或 capture（驅趕 → 抓拍）。
    previous：上一個計畫。同一個目標的 capture 進行中時沿用原本的角色，不要每次重算就換人
    （驅趕機和抓拍機一直交換，策略會完全失效；這是實測發現的問題）。"""
    ids = [d.drone_id for d in drones]
    live = [t for t in targets if t["status"] != "lost"]
    forced = _parse_operator(operator_command)

    def everyone(role: str, mode: str, reason: str, target_id=None) -> Plan:
        return Plan(mode=mode, target_id=target_id, assignments=[{"drone_id": i, "role": role} for i in ids],
                    formation={"radius_m": radius_m}, reason=reason)

    if forced in ("hold", "return_home"):
        return everyone("idle", forced, f"操作員指令：{operator_command[:30]}")
    if not live:
        return everyone("searcher", "search", "目前沒有可追蹤的目標，分散搜尋")

    tgt = min(live, key=lambda t: t["uncertainty_m"])
    p, v = tgt["position_world_ned_m"], tgt["velocity_world_ned_mps"]
    speed = math.hypot(v["n"], v["e"])
    cn, ce = sum(d.n for d in drones) / len(drones), sum(d.e for d in drones) / len(drones)
    heading = math.atan2(v["e"], v["n"]) if speed > 0.3 else math.atan2(p["e"] - ce, p["n"] - cn)
    dist = math.hypot(p["n"] - cn, p["e"] - ce)
    if len(ids) >= 2 and (forced == "capture" or (forced is None and strategy == "capture")):
        keep = (previous is not None and previous.mode == "capture" and previous.target_id == tgt["target_id"]
                and sorted(a.drone_id for a in previous.assignments) == sorted(ids))
        if keep:
            return Plan(mode="capture", target_id=tgt["target_id"], assignments=[a.model_dump() for a in previous.assignments],
                        formation={"radius_m": radius_m}, reason="捕捉進行中，維持原本的角色分工")
        reason = "驅趕機從後方把目標趕向捕捉區，抓拍機在捕捉區等、目標靠近時迎上去抓拍"
        if capture_cfg is not None and capture_cfg.replan_zone_s is not None:  # 影片場景：目標不會反應
            reason = "目標不會理無人機：抓拍機預判它的前進方向先到攔截點，另外兩架跟在後方與側翼保持視野"
        return Plan(mode="capture", target_id=tgt["target_id"], assignments=capture_assignment(drones, p["n"], p["e"], capture_cfg or CaptureConfig(), _vel(tgt)),
                    formation={"radius_m": radius_m}, reason=reason)
    mode = forced if forced in ("track", "encircle", "search") else ("encircle" if dist < ENCIRCLE_RANGE_FACTOR * radius_m else "track")
    if mode == "search":
        return everyone("searcher", "search", "操作員要求搜尋")

    # 角色分派：窮舉排列，選總移動距離最短的（3 架只有 6 種）
    roles = ["chaser", "blocker_left", "blocker_right"] + ["overwatch"] * max(0, len(ids) - 3)
    slots = _slot_offsets(heading, radius_m)
    def slot_pos(role):
        off = slots.get(role, (0.0, 0.0))
        return p["n"] + off[0], p["e"] + off[1]
    best = min(itertools.permutations(roles[: len(ids)]),
               key=lambda perm: sum(math.hypot(d.n - slot_pos(r)[0], d.e - slot_pos(r)[1]) for d, r in zip(drones, perm)))
    hd = int(round(math.degrees(heading) % 360))
    reason = (f"目標朝 {hd}° 移動，距機群 {dist:.0f} m，先追上" if mode == "track"
              else f"目標朝 {hd}° 移動，依最短移動距離分派包圍位置")
    return Plan(mode=mode, target_id=tgt["target_id"],
                assignments=[{"drone_id": d.drone_id, "role": r} for d, r in zip(drones, best)],
                formation={"radius_m": radius_m}, reason=reason)
