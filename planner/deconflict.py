"""防撞第二層：目標點送出前做水平去衝突（C 的確定性規劃器負責，LLM 不參與）。

防撞三層：
    1. 分層：每架固定高度層 25 / 30 / 35 m（規格 §10.6），相鄰層差 5 m
    2. 這裡：(a) 任兩架目標點的水平距離 ≥ min_sep_m（預設 6 m，和 A 安全層的推開距離一致）
             (b) 飛行途中預測會交錯（CPA < min_sep_m）時，優先權低的讓路
    3. A 的安全層：最後把關（推開、3D ≥ 5 m、圍欄、逾時懸停）

為什麼第 1 層不夠：高度層只保證「水平重疊時 3D 距離 = 5 m」，剛好等於安全下限，沒有任何餘裕；
只要有一架因為起飛、讓路或姿態誤差沒維持在自己的高度層，就可能撞上。實測捕捉策略時，
抓拍機飛到目標正上方、驅趕機又追到目標附近，最小 3D 距離只剩 5.3 m（包圍策略 8.6 m）。
目標點水平分開 6 m 後，相鄰層的 3D 距離至少 √(6² + 5²) ≈ 7.8 m。

為什麼在 C 也做、不全交給 A：A 推開目標點時不知道角色，可能把抓拍機推離目標正上方，
抓拍就失敗；C 知道角色，可以讓優先權低的那架讓開（priority 數字越小越重要，不會被移動）。
被移動的目標點都記在 Adjustment，之後進動作日誌（「C 去衝突修改了哪一架、移多少」）。

座標：world_ned 的 n, e，單位 m；高度不動。
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

ROLE_PRIORITY = {  # 數字越小越重要；同優先權時兩架各讓一半
    "capturer": 0,
    "herder": 1, "chaser": 1,
    "blocker": 2, "blocker_left": 2, "blocker_right": 2, "searcher": 2,
    "overwatch": 3, "idle": 3,
}
DEFAULT_PRIORITY = 2


@dataclass(frozen=True)
class Adjustment:
    drone_id: str
    from_ne: Tuple[float, float]
    to_ne: Tuple[float, float]
    because_of: str  # 和哪一架衝突

    @property
    def moved_m(self) -> float:
        return math.dist(self.from_ne, self.to_ne)

    def to_dict(self) -> dict:
        return {"drone_id": self.drone_id, "from_ne": list(self.from_ne), "to_ne": list(self.to_ne),
                "because_of": self.because_of, "moved_m": round(self.moved_m, 2)}


def deconflict_goals(goals: Sequence[dict], roles: Mapping[str, str], min_sep_m: float = 6.0,
                     iterations: int = 8) -> Tuple[List[dict], List[Adjustment]]:
    """回傳（去衝突後的 GoalPoint 列表, 調整紀錄）。輸入不會被修改。

    每一輪檢查所有成對組合：水平距離 < min_sep_m 時，優先權低的那架沿兩點連線往外推到剛好 min_sep_m
    （同優先權各推一半）；兩點重合時沿東西向分開，方向由 drone_id 排序決定（結果可重現）。
    3 架通常 1–2 輪就收斂；iterations 是保險上限。
    """
    if min_sep_m <= 0 or len(goals) < 2:
        return [dict(g) for g in goals], []
    pos: Dict[str, List[float]] = {g["drone_id"]: [g["position_world_ned_m"]["n"], g["position_world_ned_m"]["e"]] for g in goals}
    orig = {k: tuple(v) for k, v in pos.items()}
    blame: Dict[str, str] = {}
    ids = sorted(pos)
    pri = {i: ROLE_PRIORITY.get(roles.get(i, ""), DEFAULT_PRIORITY) for i in ids}
    for _ in range(iterations):
        moved = False
        for a_i, a in enumerate(ids):
            for b in ids[a_i + 1:]:
                pa, pb = pos[a], pos[b]
                dn, de = pb[0] - pa[0], pb[1] - pa[1]
                d = math.hypot(dn, de)
                if d >= min_sep_m - 1e-9:
                    continue
                un, ue = (dn / d, de / d) if d > 1e-6 else (0.0, 1.0)  # 重合：b 往東、a 往西
                gap = min_sep_m - d
                if pri[a] == pri[b]:
                    sa, sb = gap / 2, gap / 2
                elif pri[a] < pri[b]:
                    sa, sb = 0.0, gap
                else:
                    sa, sb = gap, 0.0
                pa[0] -= un * sa
                pa[1] -= ue * sa
                pb[0] += un * sb
                pb[1] += ue * sb
                if sa:
                    blame[a] = b
                if sb:
                    blame[b] = a
                moved = True
        if not moved:
            break
    out, adj = [], []
    for g in goals:
        did = g["drone_id"]
        n, e = pos[did]
        new = dict(g)
        new["position_world_ned_m"] = {**g["position_world_ned_m"], "n": n, "e": e}
        out.append(new)
        if math.dist(orig[did], (n, e)) > 1e-6:
            adj.append(Adjustment(did, orig[did], (n, e), blame.get(did, "")))
    return out, adj


def _intended_velocity(pn: float, pe: float, goal: dict, vmax: float, gain: float) -> Tuple[float, float]:
    gp = goal["position_world_ned_m"]
    dn, de = gp["n"] - pn, gp["e"] - pe
    d = math.hypot(dn, de)
    if d < 1e-6:
        return 0.0, 0.0
    sp = min(vmax, gain * d)
    return dn / d * sp, de / d * sp


def closest_approach(pa: Tuple[float, float], va: Tuple[float, float], pb: Tuple[float, float], vb: Tuple[float, float],
                     horizon_s: float) -> Tuple[float, float]:
    """兩架等速直線前進時，在 [0, horizon_s] 內的最近距離與發生時間（CPA）。輸入 world_ned (n, e)，m 與 m/s。"""
    dn, de = pb[0] - pa[0], pb[1] - pa[1]
    rn, re = vb[0] - va[0], vb[1] - va[1]
    rv2 = rn * rn + re * re
    t = 0.0 if rv2 < 1e-9 else max(0.0, min(horizon_s, -(dn * rn + de * re) / rv2))
    return math.hypot(dn + rn * t, de + re * t), t


def yield_crossings(goals: Sequence[dict], drones: Sequence, roles: Mapping[str, str], min_sep_m: float = 6.0,
                    horizon_s: float = 6.0, vmax: float = 3.0, gain: float = 0.8,
                    margin_m: float = 2.0) -> Tuple[List[dict], List[Adjustment]]:
    """飛行途中的交錯：只看目標點分開還不夠，兩架飛過去的路線可能交叉（實測多數近距離都發生在途中）。

    以目前位置、朝目標點的預期速度（速度上限 vmax、比例 gain，和替身的控制器一致）預測 horizon_s 秒內的 CPA；
    小於 min_sep_m + margin_m 時，優先權低的那架讓路：目標點改成「往自己所在那一側、垂直於對方行進方向」
    橫移 min_sep_m + margin_m，讓出對方的路線；對方照原目標點飛。同優先權時 drone_id 排序後面的讓。

    margin_m 與 horizon_s 是「安全 vs 任務」的取捨（規則式、捕捉策略、10 seeds 實測）：
        觸發 6 m / 預測 4 s    最小水平距離 3.8 m，抓拍 10/10
        觸發 8 m / 預測 6 s    最小水平距離 6.3 m，抓拍  9/10   ← 預設（老師重視防撞，守住 6 m）
        觸發 10 m / 預測 6 s   最小水平距離 10.0 m，抓拍 1/10（讓太多，抓拍機靠不近目標）
    drones 為 fusion.DroneState（world_ned）。每架最多讓一次（取最先發生衝突的那一對）。
    """
    if min_sep_m <= 0 or len(goals) < 2:
        return [dict(g) for g in goals], []
    trigger = min_sep_m + margin_m
    pos = {d.drone_id: (d.n, d.e) for d in drones}
    by_id = {g["drone_id"]: g for g in goals}
    ids = sorted(i for i in by_id if i in pos)
    vel = {i: _intended_velocity(*pos[i], by_id[i], vmax, gain) for i in ids}
    pri = {i: ROLE_PRIORITY.get(roles.get(i, ""), DEFAULT_PRIORITY) for i in ids}
    conflicts = []
    for k, a in enumerate(ids):
        for b in ids[k + 1:]:
            d, t = closest_approach(pos[a], vel[a], pos[b], vel[b], horizon_s)
            if d < trigger:
                conflicts.append((t, a, b))
    new_goal: Dict[str, Tuple[float, float]] = {}
    blame: Dict[str, str] = {}
    for t, a, b in sorted(conflicts):
        yielder, keeper = (b, a) if pri[a] < pri[b] or (pri[a] == pri[b]) else (a, b)
        if yielder in new_goal or keeper in new_goal:
            continue
        kv = vel[keeper]
        sp = math.hypot(*kv)
        if sp > 1e-6:
            perp = (-kv[1] / sp, kv[0] / sp)
        else:  # 對方幾乎不動：直接遠離對方
            dn, de = pos[yielder][0] - pos[keeper][0], pos[yielder][1] - pos[keeper][1]
            r = math.hypot(dn, de) or 1.0
            perp = (dn / r, de / r)
        rel = (pos[yielder][0] - pos[keeper][0], pos[yielder][1] - pos[keeper][1])
        if rel[0] * perp[0] + rel[1] * perp[1] < 0:
            perp = (-perp[0], -perp[1])
        new_goal[yielder] = (pos[yielder][0] + perp[0] * trigger, pos[yielder][1] + perp[1] * trigger)
        blame[yielder] = keeper
    out, adj = [], []
    for g in goals:
        did = g["drone_id"]
        if did not in new_goal:
            out.append(dict(g))
            continue
        old = (g["position_world_ned_m"]["n"], g["position_world_ned_m"]["e"])
        n, e = new_goal[did]
        out.append({**g, "position_world_ned_m": {**g["position_world_ned_m"], "n": n, "e": e}})
        adj.append(Adjustment(did, old, (n, e), blame[did]))
    return out, adj


def min_horizontal_separation(goals: Sequence[dict]) -> Optional[float]:
    """目標點之間的最小水平距離（m）；少於兩個時回傳 None。"""
    ps = [(g["position_world_ned_m"]["n"], g["position_world_ned_m"]["e"]) for g in goals]
    if len(ps) < 2:
        return None
    return min(math.dist(p, q) for i, p in enumerate(ps) for q in ps[i + 1:])
