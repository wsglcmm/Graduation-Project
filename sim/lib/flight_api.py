"""flight_api：給 C 用的同步飛控介面（內部在背景執行緒跑 MAVSDK 的 asyncio）。

    from flight_api import Drone
    d = Drone()
    d.connect()
    d.takeoff(5.0)
    d.set_velocity(1.0, 0, 0, 0)   # 機體座標 forward/right/down (m/s)、yaw_rate (deg/s)
    d.hover()
    print(d.get_state())
    d.land()
    d.close()

set_velocity 需要持續呼叫（建議 10 Hz 以上）；超過 watchdog_s 秒沒有新指令會自動懸停。

資料來源：位置、速度、姿態都取自 MAVLink ODOMETRY（預設 100 Hz），三者共用同一個
time_usec（Gazebo 模擬時間）。經緯度與 rel_alt_m 取自 position（無時間戳）。
時間對齊：鏡頭 stamp 與 ODOMETRY 時間戳是同一個時鐘（秒）。
    s = d.get_state_at(cam.last_stamp)   # 拍下那張畫面當下的狀態（內插）
"""
import asyncio
import bisect
import math
import threading
import time
from collections import deque

from mavsdk_grpc import System
from mavsdk_grpc.action import ActionError
from mavsdk_grpc.offboard import OffboardError, VelocityBodyYawspeed

ZERO = (0.0, 0.0, 0.0, 0.0)

# 姿態歷史中每筆保存的欄位（姿態到達時，連同當下最新的位置/速度一起存）
HIST_KEYS = ("roll_deg", "pitch_deg", "yaw_deg",
             "rollspeed_deg_s", "pitchspeed_deg_s", "yawspeed_deg_s",
             "north_m", "east_m", "down_m", "height_m",
             "vn_m_s", "ve_m_s", "vd_m_s", "rel_alt_m")


def quat_to_rpy_deg(w, x, y, z):
    """PX4 四元數（機體 FRD -> NED）轉 roll/pitch/yaw（度，ZYX）。"""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

# mavsdk-grpc 的 System.__del__ 在 Python 關閉階段會因 import 失敗而印出錯誤訊息。
# flight_api 會在 close() 自己停掉 mavsdk_server，所以把 __del__ 包成「失敗就略過」。
_orig_system_del = System.__del__


def _safe_system_del(self):
    try:
        _orig_system_del(self)
    except Exception:  # noqa: BLE001
        pass


System.__del__ = _safe_system_del


class Drone:
    def __init__(self, address="udpin://0.0.0.0:14540", grpc_port=50051,
                 max_xy=3.0, max_z=1.0, max_yaw=30.0,
                 min_alt=2.0, max_alt=30.0,
                 watchdog_s=0.5, setpoint_hz=20.0, history_s=10.0, odom_hz=100.0,
                 verbose=True):
        self.address = address
        self.grpc_port = grpc_port
        self.max_xy, self.max_z, self.max_yaw = max_xy, max_z, max_yaw
        self.min_alt, self.max_alt = min_alt, max_alt
        self.watchdog_s = watchdog_s
        self.setpoint_hz = setpoint_hz
        self.odom_hz = odom_hz
        self.verbose = verbose

        self._lock = threading.Lock()
        self._state = {}
        self._hist = deque(maxlen=max(10, int(history_s * odom_hz * 1.2)))
        self._cmd = ZERO
        self._cmd_time = 0.0
        self._offboard = False
        self._watchdog_fired = False
        self._tasks = []
        self.drone = None

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    # ---------- 內部工具 ----------
    def _log(self, msg):
        if self.verbose:
            print(f"[flight_api] {msg}", flush=True)

    def _run(self, coro, timeout=None):
        """在背景 asyncio 迴圈執行 coroutine，並阻塞等結果。"""
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def _set(self, **kw):
        with self._lock:
            self._state.update(kw)

    async def _watch(self, gen, fn):
        try:
            async for v in gen:
                fn(v)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            self._log(f"遙測串流中斷：{e!r}")

    def _clamp(self, fwd, right, down, yaw_rate):
        xy = math.hypot(fwd, right)
        if xy > self.max_xy:
            k = self.max_xy / xy
            fwd, right = fwd * k, right * k
        down = max(-self.max_z, min(self.max_z, down))
        yaw_rate = max(-self.max_yaw, min(self.max_yaw, yaw_rate))
        return (fwd, right, down, yaw_rate)

    def _alt_guard(self, cmd):
        fwd, right, down, yaw = cmd
        alt = self._state.get("rel_alt_m")
        if alt is not None:
            if alt <= self.min_alt and down > 0:
                down = 0.0
            if alt >= self.max_alt and down < 0:
                down = 0.0
        return (fwd, right, down, yaw)

    # ---------- 背景 coroutine ----------
    async def _connect(self, timeout):
        self.drone = System(port=self.grpc_port)
        await self.drone.connect(system_address=self.address)

        async def wait_connected():
            async for s in self.drone.core.connection_state():
                if s.is_connected:
                    return

        async def wait_ready():
            async for h in self.drone.telemetry.health():
                if h.is_global_position_ok and h.is_home_position_ok:
                    return

        await asyncio.wait_for(wait_connected(), timeout)
        self._log("已連線，等待定位就緒…")
        await asyncio.wait_for(wait_ready(), timeout)

        tel = self.drone.telemetry
        for set_rate, hz in ((tel.set_rate_position, 20.0),
                             (tel.set_rate_odometry, self.odom_hz)):
            try:
                await set_rate(hz)
            except Exception as e:  # noqa: BLE001
                self._log(f"設定遙測頻率失敗（不影響使用）：{e!r}")

        def on_pos(p):
            self._set(lat_deg=p.latitude_deg, lon_deg=p.longitude_deg,
                      abs_alt_m=p.absolute_altitude_m, rel_alt_m=p.relative_altitude_m)

        def on_odom(o):
            stamp = o.time_usec / 1e6
            r, p, y = quat_to_rpy_deg(o.q.w, o.q.x, o.q.y, o.q.z)
            av = o.angular_velocity_body
            pb, vb = o.position_body, o.velocity_body  # PX4 實際送 NED（名稱雖為 body）
            with self._lock:
                self._state.update(
                    roll_deg=r, pitch_deg=p, yaw_deg=y,
                    rollspeed_deg_s=math.degrees(av.roll_rad_s),
                    pitchspeed_deg_s=math.degrees(av.pitch_rad_s),
                    yawspeed_deg_s=math.degrees(av.yaw_rad_s),
                    north_m=pb.x_m, east_m=pb.y_m, down_m=pb.z_m, height_m=-pb.z_m,
                    vn_m_s=vb.x_m_s, ve_m_s=vb.y_m_s, vd_m_s=vb.z_m_s,
                    odom_time_usec=o.time_usec, att_timestamp_us=o.time_usec,
                    sim_time=stamp)
                if stamp > 0 and (not self._hist or stamp > self._hist[-1]["stamp"]):
                    snap = {k: self._state.get(k) for k in HIST_KEYS}
                    snap["stamp"] = stamp
                    self._hist.append(snap)

        watchers = [
            (tel.position(), on_pos),
            (tel.odometry(), on_odom),
            (tel.armed(), lambda v: self._set(armed=v)),
            (tel.in_air(), lambda v: self._set(in_air=v)),
            (tel.flight_mode(), lambda v: self._set(flight_mode=str(v))),
            (tel.battery(), lambda b: self._set(battery_percent=b.remaining_percent)),
        ]
        for gen, fn in watchers:
            self._tasks.append(asyncio.ensure_future(self._watch(gen, fn)))
        self._tasks.append(asyncio.ensure_future(self._setpoint_loop()))
        await asyncio.sleep(0.5)  # 讓遙測先填一輪

    async def _setpoint_loop(self):
        period = 1.0 / self.setpoint_hz
        try:
            while True:
                await asyncio.sleep(period)
                if not self._offboard:
                    continue
                with self._lock:
                    cmd, t = self._cmd, self._cmd_time
                    if time.time() - t > self.watchdog_s and cmd != ZERO:
                        cmd = self._cmd = ZERO
                        if not self._watchdog_fired:
                            self._watchdog_fired = True
                            self._log(f"看門狗：{self.watchdog_s}s 未收到新指令，自動懸停")
                cmd = self._alt_guard(cmd)
                try:
                    await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(*cmd))
                except Exception as e:  # noqa: BLE001
                    self._log(f"送出速度指令失敗：{e!r}")
        except asyncio.CancelledError:
            pass

    async def _start_offboard(self):
        await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(*ZERO))
        try:
            await self.drone.offboard.start()
        except OffboardError as e:
            raise RuntimeError(f"進入 Offboard 失敗：{e._result.result}") from e
        self._offboard = True
        self._log("進入 Offboard 模式")

    async def _stop_offboard(self):
        if not self._offboard:
            return
        self._offboard = False
        try:
            await self.drone.offboard.set_velocity_body(VelocityBodyYawspeed(*ZERO))
            await self.drone.offboard.stop()
            self._log("離開 Offboard 模式（PX4 轉為 Hold）")
        except OffboardError as e:
            self._log(f"離開 Offboard 失敗：{e._result.result}")

    async def _wait(self, cond, timeout, what):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(f"等待逾時：{what}")

    async def _wait_armable(self, timeout):
        """等 PX4 自檢通過（health.is_armable）。"""
        async def w():
            async for h in self.drone.telemetry.health():
                if h.is_armable:
                    return
        try:
            await asyncio.wait_for(w(), timeout)
        except asyncio.TimeoutError:
            self._log("等待 is_armable 逾時，仍嘗試解鎖")

    # ---------- 給 C 使用的公開函式 ----------
    def connect(self, timeout=30.0):
        """連線並等到定位就緒。"""
        self._run(self._connect(timeout), timeout * 2 + 5)
        self._log("定位就緒，可以起飛")

    def takeoff(self, alt=5.0, timeout=60.0):
        """解鎖並起飛到相對高度 alt（公尺），到達 95% 後返回。"""
        alt = max(self.min_alt, min(self.max_alt, alt))

        async def go():
            if self._state.get("in_air"):
                raise RuntimeError("無人機已經在空中，請先 land() 或重開模擬")
            await self._wait_armable(30.0)
            await self.drone.action.set_takeoff_altitude(alt)
            for i in range(5):                      # 剛啟動或剛落地時偶爾會被拒，重試
                try:
                    await self.drone.action.arm()
                    break
                except ActionError as e:
                    if i == 4:
                        raise RuntimeError(
                            f"解鎖被拒 5 次（{e._result.result}），請看終端 1 的 "
                            "'Arming denied' / 'Preflight Fail' 訊息") from e
                    self._log(f"解鎖被拒（{e._result.result}），2 秒後重試…")
                    await asyncio.sleep(2.0)
            await self.drone.action.takeoff()
            await self._wait(lambda: self._state.get("rel_alt_m", 0) >= alt * 0.95,
                             timeout, f"爬升到 {alt} m")
        self._run(go())
        self._log(f"已到達 {self._state.get('rel_alt_m', 0):.2f} m")

    def set_velocity(self, fwd, right, down, yaw_rate):
        """機體座標速度指令。回傳實際採用（限幅後）的 (fwd, right, down, yaw_rate)。"""
        if not self._state.get("in_air"):
            raise RuntimeError("尚未起飛，不能下速度指令")
        cmd = self._clamp(float(fwd), float(right), float(down), float(yaw_rate))
        with self._lock:
            self._cmd, self._cmd_time = cmd, time.time()
            self._watchdog_fired = False
        if not self._offboard:
            self._run(self._start_offboard(), 10)
        return cmd

    def hover(self):
        """原地懸停。"""
        if self._offboard:
            with self._lock:
                self._cmd, self._cmd_time = ZERO, time.time()
        else:
            self._run(self.drone.action.hold(), 10)
        self._log("懸停")

    def land(self, timeout=90.0):
        """降落，等到落地且上鎖後返回。"""
        async def go():
            await self._stop_offboard()
            await self.drone.action.land()
            await self._wait(lambda: not self._state.get("in_air", True)
                             and not self._state.get("armed", True), timeout, "落地上鎖")
        self._log("降落中…")
        self._run(go())
        self._log("已落地並上鎖")

    def return_to_launch(self):
        """返航（不等待完成）。"""
        async def go():
            await self._stop_offboard()
            await self.drone.action.return_to_launch()
        self._run(go(), 10)
        self._log("返航中")

    def get_state(self):
        """回傳目前狀態的 dict 複本（附 time 欄位，為本機時間）。"""
        with self._lock:
            s = dict(self._state)
        s["offboard"] = self._offboard
        s["time"] = time.time()
        return s

    def get_state_at(self, stamp, max_gap=0.1):
        """回傳模擬時間 stamp（秒）當下的姿態/位置/速度/角速度（前後兩筆線性內插，yaw 處理 ±180）。

        stamp 通常就是 SimCamera 的 cam.last_stamp。
        超出歷史範圍超過 max_gap 秒（太舊或太新）回傳 None。
        回傳 dict 另含 stamp_used（實際使用的資料時間）與 interpolated。
        """
        with self._lock:
            hist = list(self._hist)
        if not hist:
            return None
        stamps = [h["stamp"] for h in hist]
        i = bisect.bisect_left(stamps, stamp)
        if i == 0 or i == len(hist):
            h = hist[0] if i == 0 else hist[-1]
            if abs(h["stamp"] - stamp) > max_gap:
                return None
            out = dict(h)
            out.update(stamp_used=h["stamp"], stamp=stamp, interpolated=False)
            return out
        a, b = hist[i - 1], hist[i]
        w = (stamp - a["stamp"]) / (b["stamp"] - a["stamp"])
        out = {"stamp": stamp, "stamp_used": stamp, "interpolated": True}
        for k in HIST_KEYS:
            va, vb = a.get(k), b.get(k)
            if va is None or vb is None:
                out[k] = va if w < 0.5 else vb
            elif k == "yaw_deg":
                d = (vb - va + 180.0) % 360.0 - 180.0
                out[k] = (va + w * d + 180.0) % 360.0 - 180.0
            else:
                out[k] = va + w * (vb - va)
        return out

    async def _shutdown(self):
        await self._stop_offboard()
        tasks, self._tasks = self._tasks, []
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def close(self):
        """懸停並結束連線（不會降落）。可重複呼叫。"""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            if self.drone is not None:
                self._run(self._shutdown(), 10)
        finally:
            stop = getattr(self.drone, "_stop_mavsdk_server", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # noqa: BLE001
                    pass
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=2)
            self._log("已關閉連線")
