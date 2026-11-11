"""模仿 A 的 agent_server 的假伺服器：協定與安全規則相同，給 C 的測試與離線開發用。

    python -m fleet_link.fake_server              # ws://127.0.0.1:8765
    python -m demo.run --fleet sitl               # 另開一個終端機，Demo 連過來

依據 A 的〈項目 8：多機介面說明〉（2026-11-02）與 11/10 的更新，模仿「看得見的行為」，不是飛控模擬：
    - 出生點 uav1/2/3 = (−8, −6) / (−8, 0) / (−8, 6)，高度層 25 / 30 / 35 m
    - GoalPoint 的 d 一律改成該架高度層；超出 ±80 m 夾到邊界；與其他架目標點水平 < 6 m 推開到 6 m；
      max_speed_mps 上限 3（省略時 2.5）；未知 drone_id、缺座標、schema_version 不符、還在地上、緊急停止中 → 拒絕
    - 單架超過 1.0 s 沒收到 GoalPoint → 懸停，FleetPose status = failsafe（goal_timeout）
    - 完全沒有任何訊息 3 s → 全部懸停；15 s → 全部返航
不模擬 11/10 新增的讓路（yield），等規格 §10.8 定案再補。
"""
import argparse
import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DRONE_IDS = ("uav1", "uav2", "uav3")


@dataclass
class FakeConfig:
    origins: Tuple[Tuple[float, float, float], ...] = ((-8.0, -6.0, 0.0), (-8.0, 0.0, 0.0), (-8.0, 6.0, 0.0))
    layers_m: Tuple[float, ...] = (25.0, 30.0, 35.0)
    fence_m: float = 80.0
    push_m: float = 6.0
    default_speed_mps: float = 2.5
    max_speed_mps: float = 3.0
    goal_timeout_s: float = 1.0
    hover_timeout_s: float = 3.0
    return_timeout_s: float = 15.0
    climb_mps: float = 3.0
    accel_mps2: float = 3.0
    pose_hz: float = 20.0
    sim_dt_s: float = 0.02
    sim_t0_ns: int = 30_000_000_000


@dataclass
class _Drone:
    i: int
    drone_id: str
    origin: Tuple[float, float, float]
    layer: float
    n: float
    e: float
    h: float = 0.0
    vn: float = 0.0
    ve: float = 0.0
    yaw: float = 0.0
    mode: str = "landed"  # landed / takeoff / agent / hold / failsafe / return / land
    failsafe_reason: Optional[str] = None
    goal: Optional[Tuple[float, float, float]] = None  # n, e, speed
    goal_yaw: Optional[float] = None
    goal_t: float = 0.0


class FakeFleet:
    """假機隊的狀態與安全規則（同步，方便單獨測試）。"""

    def __init__(self, cfg: FakeConfig = FakeConfig()):
        self.cfg = cfg
        self.drones = [_Drone(i, did, o, cfg.layers_m[i], o[0], o[1]) for i, (did, o) in enumerate(zip(DRONE_IDS, cfg.origins))]
        self.by_id = {d.drone_id: d for d in self.drones}
        self.sim_ns = cfg.sim_t0_ns
        self.last_msg = time.monotonic()
        self.timeout_state = None
        self.estop = False
        self.min_sep_m = math.inf

    # ---------- GoalPoint ----------
    def goal_point(self, g: dict) -> dict:
        c = self.cfg
        ack = {"schema_version": "0.2.0", "plan_id": g.get("plan_id"), "drone_id": g.get("drone_id"),
               "timestamp_ns": g.get("timestamp_ns"), "accepted": False, "adjusted": False, "reason": None,
               "applied_position_world_ned_m": None}
        d = self.by_id.get(g.get("drone_id"))
        p = g.get("position_world_ned_m") or {}
        vals = [p.get(k) for k in ("n", "e", "d")]
        if d is None:
            ack["reason"] = f"未知的 drone_id {g.get('drone_id')!r}（應為 {'、'.join(DRONE_IDS)}）"
            return ack
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
            ack["reason"] = "position_world_ned_m 需要 n、e、d 三個有限數值"
            return ack
        if g.get("schema_version") != "0.2.0":
            ack["reason"] = "schema_version 必須是 0.2.0"
            return ack
        if self.estop:
            ack["reason"] = "緊急停止中"
            return ack
        if d.mode in ("landed", "takeoff", "land"):
            ack["reason"] = "該架還在地上或起降中"
            return ack
        n, e, dd = vals
        reasons = []
        if abs(-dd - d.layer) > 0.01:
            reasons.append(f"高度 {-dd:.1f} m → 高度層 {d.layer:.0f} m")
        cn, ce = max(-c.fence_m, min(c.fence_m, n)), max(-c.fence_m, min(c.fence_m, e))
        if (cn, ce) != (n, e):
            reasons.append(f"超出飛行範圍，改到邊界 ({cn:.1f}, {ce:.1f})")
            n, e = cn, ce
        for o in self.drones:
            if o is d or o.goal is None:
                continue
            dn, de = n - o.goal[0], e - o.goal[1]
            dist = math.hypot(dn, de)
            if dist < c.push_m:
                if dist < 1e-6:
                    dn, de, dist = 0.0, 1.0, 1.0
                n, e = o.goal[0] + dn / dist * c.push_m, o.goal[1] + de / dist * c.push_m
                reasons.append(f"與 {o.drone_id} 的目標點水平距離 < {c.push_m:.0f} m，推開到 ({n:.1f}, {e:.1f})")
        sp = g.get("max_speed_mps")
        sp = c.default_speed_mps if sp is None else float(sp)
        if sp > c.max_speed_mps:
            reasons.append(f"速度 {sp:g} → 上限 {c.max_speed_mps:g} m/s")
            sp = c.max_speed_mps
        d.goal, d.goal_yaw, d.goal_t = (n, e, sp), g.get("yaw_rad"), time.monotonic()
        d.mode, d.failsafe_reason = "agent", None
        ack.update(accepted=True, adjusted=bool(reasons), reason="；".join(reasons) or None,
                   applied_position_world_ned_m={"n": n, "e": e, "d": -d.layer})
        return ack

    # ---------- 指令 ----------
    def command(self, m: dict) -> dict:
        cmd, idx = m.get("cmd"), m.get("drone")
        targets = self.drones if idx is None else [self.drones[int(idx)]]
        if cmd == "takeoff":
            for d in targets:
                if d.mode == "landed":
                    d.mode = "takeoff"
            return {"ok": True, "cmd": cmd}
        if cmd in ("hold", "return_home", "land"):
            for d in targets:
                if d.mode not in ("landed",):
                    d.mode = {"hold": "hold", "return_home": "return", "land": "land"}[cmd]
                    d.goal = None
            return {"ok": True, "cmd": cmd}
        if cmd == "heartbeat":
            return {"ok": True, "cmd": cmd}
        if cmd == "status":
            return {"ok": True, "cmd": cmd, **self.status()}
        if cmd == "estop":
            self.estop = True
            for d in self.drones:
                if d.mode != "landed":
                    d.mode, d.goal = "hold", None
            return {"ok": True, "cmd": cmd}
        if cmd == "resume":
            self.estop = False
            return {"ok": True, "cmd": cmd}
        return {"ok": False, "reason": f"未知指令 {cmd!r}", "cmd": cmd}

    # ---------- 物理與安全監控 ----------
    def step(self, dt: float) -> None:
        c = self.cfg
        now = time.monotonic()
        self.sim_ns += int(dt * 1e9)
        idle = now - self.last_msg
        if idle > c.return_timeout_s and self.timeout_state != "return":
            self.timeout_state = "return"
            for d in self.drones:
                if d.mode not in ("landed", "land"):
                    d.mode, d.failsafe_reason, d.goal = "return", "agent_timeout", None
        elif c.hover_timeout_s < idle <= c.return_timeout_s and self.timeout_state is None:
            self.timeout_state = "hover"
            for d in self.drones:
                if d.mode == "agent":
                    d.mode, d.failsafe_reason = "failsafe", "agent_timeout"
        elif idle <= c.hover_timeout_s:
            self.timeout_state = None
        for d in self.drones:
            if d.mode == "agent" and now - d.goal_t > c.goal_timeout_s:
                d.mode, d.failsafe_reason = "failsafe", "goal_timeout"
            tn, te, sp, th = d.n, d.e, 0.0, d.h
            if d.mode == "takeoff":
                th = d.layer
                if abs(d.h - d.layer) < 0.3:
                    d.mode = "hold"
            elif d.mode == "agent" and d.goal:
                tn, te, sp = d.goal
                th = d.layer
            elif d.mode in ("hold", "failsafe"):
                th = d.layer if d.h > 0.5 else d.h
            elif d.mode == "return":
                tn, te, sp, th = d.origin[0], d.origin[1], c.default_speed_mps, d.layer
                if math.hypot(d.n - tn, d.e - te) < 0.5:
                    d.mode = "land"
            if d.mode == "land":
                th = 0.0
                if d.h <= 0.05:
                    d.mode, d.h, d.goal = "landed", 0.0, None
            self._move(d, tn, te, sp, th, dt)
        flying = [d for d in self.drones if d.mode != "landed"]
        for i, a in enumerate(flying):
            for b in flying[i + 1:]:
                self.min_sep_m = min(self.min_sep_m, math.dist((a.n, a.e, a.h), (b.n, b.e, b.h)))

    def _move(self, d: _Drone, tn: float, te: float, sp: float, th: float, dt: float) -> None:
        c = self.cfg
        dn, de = tn - d.n, te - d.e
        dist = math.hypot(dn, de)
        want = min(sp, 0.8 * dist)  # 接近時減速
        wn, we = (dn / dist * want, de / dist * want) if dist > 1e-6 else (0.0, 0.0)
        an, ae = (wn - d.vn) / dt, (we - d.ve) / dt
        a = math.hypot(an, ae)
        if a > c.accel_mps2:
            an, ae = an / a * c.accel_mps2, ae / a * c.accel_mps2
        d.vn += an * dt
        d.ve += ae * dt
        d.n += d.vn * dt
        d.e += d.ve * dt
        d.h += max(-c.climb_mps * dt, min(c.climb_mps * dt, th - d.h))
        if d.goal_yaw is not None and d.mode == "agent":
            d.yaw = d.goal_yaw
        elif math.hypot(d.vn, d.ve) > 0.5:
            d.yaw = math.atan2(d.ve, d.vn)

    # ---------- 輸出 ----------
    def fleet_pose(self, d: _Drone) -> dict:
        status = "landed" if d.mode == "landed" else ("failsafe" if d.mode == "failsafe" else "flying")
        return {"type": "FleetPose",
                "data": {"schema_version": "0.2.0", "drone_id": d.drone_id, "timestamp_ns": self.sim_ns,
                         "position_world_ned_m": {"n": d.n, "e": d.e, "d": -d.h},
                         "attitude_rad": {"roll": 0.0, "pitch": 0.0, "yaw": d.yaw},
                         "velocity_world_ned_mps": {"n": d.vn, "e": d.ve, "d": 0.0}, "status": status},
                "mode": d.mode, "failsafe_reason": d.failsafe_reason}

    def status(self) -> dict:
        return {"sim_time": self.sim_ns / 1e9, "estop": self.estop, "timeout_state": self.timeout_state,
                "min_sep_m": None if math.isinf(self.min_sep_m) else round(self.min_sep_m, 3),
                "drones": [{"id": d.i, "drone_id": d.drone_id, "mode": d.mode, "failsafe": d.failsafe_reason,
                            "origin_world_ned_m": list(d.origin), "world_n": d.n, "world_e": d.e, "height_m": d.h,
                            "goal": list(d.goal[:2]) if d.goal else None} for d in self.drones]}

    def all_reached_layer(self, idx: Optional[List[int]] = None) -> bool:
        ds = self.drones if idx is None else [self.drones[i] for i in idx]
        return all(abs(d.h - d.layer) < 0.5 for d in ds)


class FakeAServer:
    """在背景執行緒跑的假 agent_server。port=0 時自動選空的埠（測試用）。"""

    def __init__(self, cfg: FakeConfig = FakeConfig(), host: str = "127.0.0.1", port: int = 8765):
        self.fleet = FakeFleet(cfg)
        self.host, self.port = host, port
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stop: Optional[asyncio.Event] = None
        self.clients: Dict[object, Dict[str, bool]] = {}

    @property
    def uri(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> "FakeAServer":
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), name="fake-a-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(5.0):
            raise RuntimeError("假伺服器啟動失敗")
        return self

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=3.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    async def _main(self) -> None:
        import websockets

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with websockets.serve(self._handler, self.host, self.port, max_size=2 ** 20) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            tasks = [asyncio.ensure_future(self._physics()), asyncio.ensure_future(self._stream())]
            await self._stop.wait()
            for t in tasks:
                t.cancel()

    async def _handler(self, ws) -> None:
        import websockets

        self.clients[ws] = {"state": True, "fleet_pose": True}
        f = self.fleet
        try:
            async for raw in ws:
                f.last_msg = time.monotonic()
                try:
                    m = json.loads(raw)
                except ValueError as e:
                    await ws.send(json.dumps({"type": "reply", "ok": False, "reason": f"JSON 錯誤：{e}"}, ensure_ascii=False))
                    continue
                if m.get("type") == "GoalPoints":
                    r = {"type": "GoalPointAcks", "acks": [f.goal_point(g) for g in m.get("goals", [])]}
                elif m.get("type") == "GoalPoint" or ("cmd" not in m and "drone_id" in m):
                    r = {"type": "GoalPointAck", **f.goal_point(m)}
                elif m.get("cmd") == "stream":
                    on = bool(m.get("on", True))
                    self.clients[ws] = {k: bool(m.get(k, on)) for k in ("state", "fleet_pose")}
                    r = {"type": "reply", "id": m.get("id"), "ok": True, "cmd": "stream"}
                elif m.get("cmd") == "takeoff":
                    f.command(m)
                    idx = m.get("drones")
                    while not f.all_reached_layer(idx):  # 和 A 一樣：全部到高度層才回覆
                        await asyncio.sleep(0.05)
                    r = {"type": "reply", "id": m.get("id"), "ok": True, "cmd": "takeoff"}
                else:
                    r = {"type": "reply", "id": m.get("id"), **f.command(m)}
                await ws.send(json.dumps(r, ensure_ascii=False))
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.pop(ws, None)

    async def _physics(self) -> None:
        dt = self.fleet.cfg.sim_dt_s
        nxt = time.monotonic()
        while True:
            nxt += dt
            self.fleet.step(dt)
            await asyncio.sleep(max(0.0, nxt - time.monotonic()))

    async def _stream(self) -> None:
        import websockets

        period = 1.0 / self.fleet.cfg.pose_hz
        while True:
            await asyncio.sleep(period)
            poses = [json.dumps(self.fleet.fleet_pose(d)) for d in self.fleet.drones]
            state = json.dumps({"type": "state", **self.fleet.status()}, ensure_ascii=False)
            for ws, opt in list(self.clients.items()):
                try:
                    if opt["fleet_pose"]:
                        for p in poses:
                            await ws.send(p)
                    if opt["state"]:
                        await ws.send(state)
                except websockets.ConnectionClosed:
                    pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="模仿 A 的 agent_server（離線開發與測試用）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    srv = FakeAServer(host=args.host, port=args.port).start()
    print(f"[fake_server] {srv.uri}（模仿 A 的 agent_server；Ctrl+C 結束）", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
