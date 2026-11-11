"""C 端連 A 的 agent_server（WebSocket）：收 FleetPose、送 GoalPoint、送指令與 heartbeat。

A 的伺服器在學校的 Linux 主機，只綁 127.0.0.1:8765；從自己電腦用 SSH 通道轉到本機後連 ws://127.0.0.1:8765。

設計：
    - asyncio 在背景執行緒跑，對外是同步 API，Demo 引擎（同步迴圈）可以直接呼叫。
    - 收到的 FleetPose 先用規格 §10.2 的 strict 模型驗證，不合格的計數後丟掉，不會進到融合與規劃。
    - 超過 heartbeat_s 沒送任何訊息就自動送 heartbeat。A 的安全層：完全沒訊息 3 s 全部懸停、15 s 全部返航；
      單架超過 1 s 沒收到 GoalPoint 就懸停（failsafe = goal_timeout）。
    - A 的指令用編號指定無人機（0、1、2），這裡用 drone_id 呼叫，對應表從 state 訊息取得（預設 uavN → N−1）。
"""
import asyncio
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from pydantic import ValidationError

from fusion import DroneState

from .messages import FleetPose, GoalPointAck

DEFAULT_URI = "ws://127.0.0.1:8765"


class LinkError(RuntimeError):
    pass


@dataclass
class PoseRecord:
    pose: FleetPose
    received_s: float  # time.monotonic()
    mode: Optional[str] = None  # A 的補充資訊（外層），C 不依賴
    failsafe_reason: Optional[str] = None
    extra: Dict[str, object] = field(default_factory=dict)


@dataclass
class LinkStats:
    poses: int = 0
    bad_poses: int = 0
    goals_sent: int = 0
    acks: int = 0
    rejected: int = 0
    adjusted: int = 0
    heartbeats: int = 0
    last_reason: Dict[str, str] = field(default_factory=dict)  # drone_id → 最近一次被修改或拒絕的原因
    rtt_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=500))

    def summary(self) -> dict:
        rtt = sorted(self.rtt_ms)
        pct = lambda q: rtt[min(len(rtt) - 1, int(q * len(rtt)))] if rtt else None  # noqa: E731
        return {"poses": self.poses, "bad_poses": self.bad_poses, "goals_sent": self.goals_sent, "acks": self.acks,
                "rejected": self.rejected, "adjusted": self.adjusted, "heartbeats": self.heartbeats,
                "ack_rtt_ms_p50": pct(0.5), "ack_rtt_ms_p99": pct(0.99), "last_reason": dict(self.last_reason)}


class FleetLink:
    def __init__(self, uri: str = DEFAULT_URI, heartbeat_s: float = 1.0, connect_timeout_s: float = 5.0):
        self.uri = uri
        self.heartbeat_s = heartbeat_s
        self.connect_timeout_s = connect_timeout_s
        self.stats = LinkStats()
        self._lock = threading.Lock()
        self._poses: Dict[str, PoseRecord] = {}
        self._state: Optional[dict] = None
        self._index: Dict[str, int] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._next_id = 1
        self._last_send = 0.0
        self._sent_at: Deque[float] = deque(maxlen=200)  # 送出 GoalPoints 的時間，估 ack 來回延遲
        self._closing = False

    # ---------- 連線 ----------
    def start(self) -> "FleetLink":
        self._thread = threading.Thread(target=self._run, name="fleet-link", daemon=True)
        self._thread.start()
        if not self._ready.wait(self.connect_timeout_s + 1.0):
            raise LinkError(f"連不上 {self.uri}（逾時）。A 的 agent_server 有開嗎？SSH 通道有建立嗎？")
        if self._error:
            raise LinkError(f"連不上 {self.uri}：{self._error}") from self._error
        return self

    def close(self) -> None:
        self._closing = True
        if self._loop and self._ws:
            asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
        if self._thread:
            self._thread.join(timeout=3.0)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        except BaseException as e:  # noqa: BLE001
            self._error = self._error or e
            self._ready.set()
        finally:
            self._loop.close()

    async def _main(self) -> None:
        import websockets

        try:
            ws = await asyncio.wait_for(websockets.connect(self.uri, max_size=2 ** 22), self.connect_timeout_s)
        except BaseException as e:  # noqa: BLE001
            self._error = e
            self._ready.set()
            return
        self._ws = ws
        self._last_send = time.monotonic()
        self._ready.set()
        hb = asyncio.ensure_future(self._heartbeat())
        try:
            async for raw in ws:
                self._on_message(raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            hb.cancel()
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(LinkError("連線已中斷"))

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_s / 4)
            if time.monotonic() - self._last_send >= self.heartbeat_s:
                await self._send({"cmd": "heartbeat"})
                self.stats.heartbeats += 1

    async def _send(self, msg: dict) -> None:
        self._last_send = time.monotonic()
        await self._ws.send(json.dumps(msg, ensure_ascii=False))

    # ---------- 收訊息 ----------
    def _on_message(self, raw) -> None:
        try:
            m = json.loads(raw)
        except ValueError:
            return
        t = m.get("type")
        if t == "FleetPose":
            self._on_pose(m)
        elif t == "state":
            with self._lock:
                self._state = m
                for d in m.get("drones", []):
                    if "drone_id" in d and "id" in d:
                        self._index[d["drone_id"]] = int(d["id"])
        elif t in ("GoalPointAck", "GoalPointAcks"):
            self._on_acks(m.get("acks", [m]) if t == "GoalPointAcks" else [m])
        elif t == "reply":
            fut = self._pending.pop(m.get("id"), None)
            if fut is not None and not fut.done():
                fut.set_result(m)

    def _on_pose(self, m: dict) -> None:
        try:
            pose = FleetPose.model_validate(m.get("data"))
        except ValidationError:
            self.stats.bad_poses += 1
            return
        extra = {k: v for k, v in m.items() if k not in ("type", "data", "mode", "failsafe_reason")}
        rec = PoseRecord(pose, time.monotonic(), m.get("mode"), m.get("failsafe_reason"), extra)
        with self._lock:
            self._poses[pose.drone_id] = rec
        self.stats.poses += 1

    def _on_acks(self, acks: List[dict]) -> None:
        if self._sent_at:
            self.stats.rtt_ms.append((time.monotonic() - self._sent_at.popleft()) * 1000)
        for raw in acks:
            a = GoalPointAck.model_validate(raw)
            self.stats.acks += 1
            if not a.accepted:
                self.stats.rejected += 1
            elif a.adjusted:
                self.stats.adjusted += 1
            if a.reason and a.drone_id:
                self.stats.last_reason[a.drone_id] = a.reason

    # ---------- 對外 API（同步） ----------
    def poses(self) -> Dict[str, PoseRecord]:
        with self._lock:
            return dict(self._poses)

    def states(self, max_age_s: float = 1.0) -> List[DroneState]:
        """融合與規劃用的無人機狀態（world_ned）。超過 max_age_s 沒更新的不列入。"""
        now = time.monotonic()
        out = []
        for did, rec in sorted(self.poses().items()):
            if now - rec.received_s > max_age_s:
                continue
            p = rec.pose.position_world_ned_m
            out.append(DroneState(did, p.n, p.e, -p.d, rec.pose.attitude_rad.yaw))
        return out

    def state(self) -> Optional[dict]:
        with self._lock:
            return self._state

    def origins(self) -> Dict[str, tuple]:
        """各機出生點 origin_world_ned_m（§10.1），從 state 訊息取得。"""
        s = self.state() or {}
        return {d["drone_id"]: tuple(d["origin_world_ned_m"]) for d in s.get("drones", []) if d.get("origin_world_ned_m")}

    def wait_for_poses(self, drone_ids, timeout_s: float = 5.0) -> None:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if set(drone_ids) <= set(self.poses()):
                return
            time.sleep(0.05)
        raise LinkError(f"{timeout_s} 秒內沒收到這些無人機的 FleetPose：{sorted(set(drone_ids) - set(self.poses()))}")

    def wait_for_fresh_poses(self, drone_ids, since_s: float, timeout_s: float = 2.0) -> None:
        """等每架都收到 since_s（time.monotonic()）之後的 FleetPose。"""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            recs = self.poses()
            if all(d in recs and recs[d].received_s > since_s for d in drone_ids):
                return
            time.sleep(0.01)
        raise LinkError(f"{timeout_s} 秒內沒收到新的 FleetPose")

    def send_goals(self, goals: List[dict]) -> None:
        """送出一批 GoalPoint（§10.6）。非阻塞；ack 由背景執行緒統計在 stats。"""
        if not goals:
            return
        msg = {"type": "GoalPoints", "goals": [{k: v for k, v in g.items() if v is not None} for g in goals]}
        self._sent_at.append(time.monotonic())
        self.stats.goals_sent += len(goals)
        self._submit(self._send(msg))

    def command(self, cmd: str, drone_id: Optional[str] = None, timeout_s: float = 10.0, **kw) -> dict:
        """送 A 的指令（takeoff / hold / return_home / land / status / estop / resume），等回覆。"""
        msg = {"cmd": cmd, **kw}
        if drone_id is not None:
            msg["drone"] = self.drone_index(drone_id)

        async def go():
            fut = self._loop.create_future()
            cid = self._next_id
            self._next_id += 1
            self._pending[cid] = fut
            await self._send({"id": cid, **msg})
            return await asyncio.wait_for(fut, timeout_s)

        reply = self._submit(go()).result(timeout_s + 1.0)
        if not reply.get("ok", False):
            raise LinkError(f"{cmd} 失敗：{reply.get('reason')}")
        return reply

    def drone_index(self, drone_id: str) -> int:
        with self._lock:
            if drone_id in self._index:
                return self._index[drone_id]
        if drone_id.startswith("uav") and drone_id[3:].isdigit():
            return int(drone_id[3:]) - 1
        raise LinkError(f"不知道 {drone_id!r} 對應 A 的哪一架")

    def min_pair_distance_m(self) -> float:
        """目前任兩架的最小 3D 距離（§10.6 安全要求 ≥ 5 m）。"""
        s = [x for x in self.states()]
        best = math.inf
        for i, a in enumerate(s):
            for b in s[i + 1:]:
                best = min(best, math.dist((a.n, a.e, a.alt_m), (b.n, b.e, b.alt_m)))
        return best

    def _submit(self, coro):
        if self._loop is None or self._ws is None:
            raise LinkError("尚未連線，請先呼叫 start()")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
