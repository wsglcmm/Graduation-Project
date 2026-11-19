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
    # 捕捉區放在目標的哪個方向：fleet = 機群中心 → 目標的延長線；velocity = 目標正在跑的方向；
    # center = 目標 → 場地中心（遠離牆）；auto = 目標在跑（> 0.5 m/s）用 velocity，否則 center。
    # 實驗（20 seeds、3 / 5 m/s）fleet 最好：目標會往遠離無人機的方向逃，順勢趕最省力；
    # center 在 3 m/s 只有 4/20 抓到（要把目標往回趕過整個機群）。
    zone_direction: str = "fleet"
    # 捕捉區重新預測（秒）。None = 同一次 capture 期間固定不動（策略場景：目標會逃，靠驅趕把它趕進去）。
    # 影片場景的目標不會理無人機、也趕不動：每隔這麼久檢查一次，目標在跑（≥ replan_min_speed_mps）
    # 卻不是朝捕捉區前進（夾角 > 60°）時，改放到它新的前進方向（預測攔截）。
    replan_zone_s: Optional[float] = None
    replan_min_speed_mps: float = 1.0


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


def zone_base_direction(tn: float, te: float, drones: Sequence, cfg: CaptureConfig, vel: Optional[Vec] = None,
                        fallback_heading_rad: float = 0.0) -> Vec:
    """捕捉區相對目標的方向（單位向量，world_ned）。見 CaptureConfig.zone_direction。"""
    fallback = (math.cos(fallback_heading_rad), math.sin(fallback_heading_rad))
    cn = sum(d.n for d in drones) / len(drones) if drones else tn
    ce = sum(d.e for d in drones) / len(drones) if drones else te
    fleet = _unit(tn - cn, te - ce)
    center = _unit(-tn, -te)
    moving = _unit(*vel) if vel is not None and math.hypot(*vel) > 0.5 else None
    mode = cfg.zone_direction
    if mode == "velocity":
        return moving or fleet or fallback
    if mode == "center":
        return center or fleet or fallback
    if mode == "auto":
        return moving or center or fleet or fallback
    return fleet or fallback


def propose_zone(tn: float, te: float, drones: Sequence, cfg: CaptureConfig = CaptureConfig(),
                 fallback_heading_rad: float = 0.0, vel: Optional[Vec] = None) -> CaptureZone:
    """決定捕捉區中心。drones 為 DroneState（world_ned）；vel 為目標速度 (vn, ve)。

    方向見 CaptureConfig.zone_direction；若因場地邊界被壓縮得太近，依序試 ±45°、±90°、±135°，選離目標最遠的。
    """
    base = zone_base_direction(tn, te, drones, cfg, vel, fallback_heading_rad)
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
        self._zone_t: Optional[float] = None  # 上次放捕捉區的時間（s）
        self.zone_replans = 0

    def reset(self) -> None:
        self.zone, self._target_id, self._side, self._engaged = None, None, 1.0, False
        self._zone_t = None

    def ensure_zone(self, target: dict, drones: Sequence, heading_rad: float, t_s: Optional[float] = None) -> CaptureZone:
        """capture 開始（或換目標）時決定捕捉區；之後沿用（CaptureConfig.replan_zone_s 有設時會重新預測）。
        target 為 FusedTarget dict；t_s：目前時間（s），重新預測要用。"""
        p = target["position_world_ned_m"]
        v = target.get("velocity_world_ned_mps") or {"n": 0.0, "e": 0.0}
        new = self.zone is None or self._target_id != target["target_id"]
        if not new and self._should_replan(p, v, t_s):
            new = True
            self.zone_replans += 1
        if new:
            if self._parked(p, v, t_s):
                # 不會反應的目標停著（等紅燈、路邊停車）：沒有前進方向可以預測，捕捉區直接放在目標上，抓拍機過去拍
                self.zone = CaptureZone(p["n"], p["e"], self.cfg.zone_radius_m)
            else:
                self.zone = propose_zone(p["n"], p["e"], drones, self.cfg, heading_rad, (v["n"], v["e"]))
            self._target_id = target["target_id"]
            self._engaged = False
            self._zone_t = t_s
        return self.zone

    def _parked(self, p: dict, v: dict, t_s: Optional[float]) -> bool:
        return (self.cfg.replan_zone_s is not None and t_s is not None
                and math.hypot(v["n"], v["e"]) < self.cfg.replan_min_speed_mps)

    def _should_replan(self, p: dict, v: dict, t_s: Optional[float]) -> bool:
        """只在 replan_zone_s 有設時（不會反應的目標）：
        - 目標停著（< replan_min_speed_mps）且不在捕捉區內 → 重新放（放在目標上）
        - 目標在跑、離捕捉區還遠、而且不是朝捕捉區前進（夾角 > 60°）→ 重新預測到它前進的方向"""
        c, z = self.cfg, self.zone
        if c.replan_zone_s is None or t_s is None:
            return False
        if self._zone_t is not None and t_s - self._zone_t < c.replan_zone_s:
            return False
        speed = math.hypot(v["n"], v["e"])
        dz = math.hypot(z.n - p["n"], z.e - p["e"])
        if speed < c.replan_min_speed_mps:
            return dz > c.zone_radius_m  # 停著、但不在捕捉區內（例如捕捉區還在先前預測的前方）→ 改放到目標上
        if dz <= c.engage_radius_m:
            return False
        cos = ((z.n - p["n"]) * v["n"] + (z.e - p["e"]) * v["e"]) / (speed * dz)
        return cos < 0.5

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
