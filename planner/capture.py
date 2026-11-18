"""捕捉策略（capture）的確定性幾何：Agent 說「uav1 當驅趕機」，這裡算驅趕機要站在哪。

策略（老師 2026-11-16 舉的例子：一架把目標趕進捕捉範圍、一架跟上抓拍）：
    捕捉區 P     capture 開始時決定，之後固定（畫面上畫成圓圈）。方向 = 機群 → 目標的延長線
                 （目標會往遠離無人機的方向逃，順勢趕），中心在目標前方 ambush_ahead_m、限制在場地內。
    herder       驅趕機：站在目標「遠離捕捉區」那一側（u = 捕捉區 → 目標的單位向量）T + u × herd_distance，
                 目標往遠離驅趕機的方向逃，也就是往捕捉區跑（shepherding 的 driving position，Strömbom 2014）。
                 如果驅趕機在錯的一側（捕捉區那側），先繞到側面再切到後方，避免把目標往反方向趕。
    capturer     抓拍機：目標離捕捉區還遠時在 P 上空等（伏擊）；目標進入 engage_radius_m 後迎上去，
                 飛到目標預測位置正上方，讓目標落在相機畫面中央；目標跑出 disengage_radius_m 才回伏擊點（遲滯）。
    blocker      封鎖機：守在目標往側邊偏離的那一側的側後方（偏離方向有遲滯，避免左右抖動），和驅趕機成 V 字。

所有座標為 world_ned 的 (n, e)，單位 m；高度仍由 FormationPlanner 的高度層決定。
"""
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Vec = Tuple[float, float]


@dataclass(frozen=True)
class CaptureConfig:
    ambush_ahead_m: float = 35.0  # 捕捉區中心設在目標前方多遠
    zone_radius_m: float = 10.0  # 捕捉區半徑（畫面顯示、「驅趕成功」判定）
    arena_radius_m: float = 60.0  # 場地半徑（和模擬目標的場地一致）
    zone_margin_m: float = 12.0  # 捕捉區中心離場地邊界至少多遠
    herd_distance_m: float = 12.0
    block_distance_m: float = 14.0
    engage_radius_m: float = 18.0
    disengage_radius_m: float = 30.0  # 迎上去之後，目標離捕捉區超過這個距離才放棄、回伏擊點（遲滯）
    lead_s: float = 1.0
    side_switch_mps: float = 0.3  # 目標側向速度超過這個值才換封鎖邊


@dataclass(frozen=True)
class CaptureZone:
    n: float
    e: float
    radius_m: float

    def contains(self, n: float, e: float) -> bool:
        return math.hypot(n - self.n, e - self.e) <= self.radius_m


def _unit(x: float, y: float) -> Optional[Vec]:
    r = math.hypot(x, y)
    return (x / r, y / r) if r > 1e-6 else None


def _rot(v: Vec, a: float) -> Vec:
    c, s = math.cos(a), math.sin(a)
    return v[0] * c - v[1] * s, v[0] * s + v[1] * c


def propose_zone(tn: float, te: float, drones: Sequence, cfg: CaptureConfig = CaptureConfig(),
                 fallback_heading_rad: float = 0.0) -> CaptureZone:
    """決定捕捉區中心。drones 為 DroneState（world_ned）。

    優先「機群中心 → 目標」的方向；若因場地邊界被壓縮得太近，依序試 ±45°、±90°、±135°，選離目標最遠的。
    """
    cn = sum(d.n for d in drones) / len(drones) if drones else tn
    ce = sum(d.e for d in drones) / len(drones) if drones else te
    base = _unit(tn - cn, te - ce) or (math.cos(fallback_heading_rad), math.sin(fallback_heading_rad))
    limit = cfg.arena_radius_m - cfg.zone_margin_m
    best, best_score = None, -math.inf
    for k, a in enumerate((0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75)):
        dn, de = _rot(base, a * math.pi)
        pn, pe = tn + dn * cfg.ambush_ahead_m, te + de * cfg.ambush_ahead_m
        r = math.hypot(pn, pe)
        if r > limit:
            pn, pe = pn * limit / r, pe * limit / r
        score = math.hypot(pn - tn, pe - te) - 2.0 * k  # 同樣遠時偏好轉得少的方向
        if score > best_score:
            best, best_score = (pn, pe), score
    return CaptureZone(best[0], best[1], cfg.zone_radius_m)


class CaptureGeometry:
    """有狀態：捕捉區與封鎖邊在同一次 capture 期間保持不變。"""

    def __init__(self, cfg: CaptureConfig = CaptureConfig()):
        self.cfg = cfg
        self.zone: Optional[CaptureZone] = None
        self._side = 1.0
        self._target_id: Optional[int] = None
        self._engaged = False

    def reset(self) -> None:
        self.zone, self._target_id, self._side, self._engaged = None, None, 1.0, False

    def ensure_zone(self, target: dict, drones: Sequence, heading_rad: float) -> CaptureZone:
        """capture 開始（或換目標）時決定捕捉區；之後沿用。target 為 FusedTarget dict。"""
        if self.zone is None or self._target_id != target["target_id"]:
            p = target["position_world_ned_m"]
            self.zone = propose_zone(p["n"], p["e"], drones, self.cfg, heading_rad)
            self._target_id = target["target_id"]
            self._engaged = False
        return self.zone

    def goal(self, role: str, target: dict, drone, action: Optional[str] = None) -> Vec:
        """某個角色的目標點 (n, e)。必須先呼叫 ensure_zone。

        action：各機 Agent 選的動作（多 Agent 模式）。None = 單一指揮官模式，由這裡內建的規則決定
        （和規則式各機 Agent 的決策相同，見 llm_agent/drone_agent.rule_action）。
        """
        c, z = self.cfg, self.zone
        p, v = target["position_world_ned_m"], target["velocity_world_ned_mps"]
        tn, te = p["n"] + v["n"] * c.lead_s, p["e"] + v["e"] * c.lead_s
        u = _unit(tn - z.n, te - z.e) or (1.0, 0.0)  # 捕捉區 → 目標
        if role == "capturer" and action in ("ambush", "engage"):
            return (z.n, z.e) if action == "ambush" else (tn, te)
        if role == "herder" and action in ("push", "circle_behind"):
            if action == "push":
                return tn + u[0] * c.herd_distance_m, te + u[1] * c.herd_distance_m
            rel = _unit(drone.n - tn, drone.e - te) or _rot(u, math.pi / 2)
            side = 1.0 if (u[0] * rel[1] - u[1] * rel[0]) >= 0 else -1.0
            fn, fe = _rot(u, side * math.pi / 2)
            return tn + fn * 1.5 * c.herd_distance_m, te + fe * 1.5 * c.herd_distance_m
        if role == "capturer":
            # 遲滯：進入 engage_radius_m 才迎上去，超過 disengage_radius_m 才放棄。
            # 只用一個門檻時，抓拍機一迎上去就把目標嚇回門檻外，於是退回伏擊點、目標又靠近……
            # 5 m/s 實測會這樣來回 70 秒（seed 9：108 s 才抓到）。
            dz = math.hypot(p["n"] - z.n, p["e"] - z.e)
            if dz <= c.engage_radius_m:
                self._engaged = True
            elif dz > c.disengage_radius_m:
                self._engaged = False
            if not self._engaged:
                return z.n, z.e  # 伏擊：在捕捉區上空等
            return tn, te  # 迎上去，飛到目標正上方
        if role == "herder":
            behind = (tn + u[0] * c.herd_distance_m, te + u[1] * c.herd_distance_m)
            rel = _unit(drone.n - tn, drone.e - te)
            if rel is not None and rel[0] * u[0] + rel[1] * u[1] < 0:  # 在捕捉區那側：先繞到側面
                side = 1.0 if (u[0] * rel[1] - u[1] * rel[0]) >= 0 else -1.0
                fn, fe = _rot(u, side * math.pi / 2)
                return tn + fn * 1.5 * c.herd_distance_m, te + fe * 1.5 * c.herd_distance_m
            return behind
        if role == "blocker":
            perp = _rot(u, math.pi / 2)
            lateral = v["n"] * perp[0] + v["e"] * perp[1]
            if abs(lateral) > c.side_switch_mps:
                self._side = 1.0 if lateral > 0 else -1.0
            k = c.block_distance_m
            # 側後方（和驅趕機成 V 字）：擋住側逃，又不會站到目標和捕捉區之間把目標擋回去（實測發現的問題）
            return tn + self._side * perp[0] * k + u[0] * 0.3 * k, te + self._side * perp[1] * k + u[1] * 0.3 * k
        raise ValueError(f"capture 不支援角色 {role!r}")
