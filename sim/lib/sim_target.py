"""sim_target：用 Gazebo 的 set_pose service 讓目標（車、人）沿固定路徑移動，進度以模擬時間為準。

    from sim_target import Route, MovingTarget
    route = Route(start=(-30, 3), heading_deg=0).line(60)              # 從 (-30, 3) 往東直走 60 m
    car = MovingTarget("car_blue", route, speed=1.5, clock=truth.latest_stamp)
    car.place_at_start()
    car.start()                  # 背景執行緒約 30 Hz 更新位置
    s = car.state()              # 目前的 x, y, heading, vx, vy, yaw_rate（跟車時當前饋）
    car.stop()

    python sim_target.py car_blue      # 單獨測試：car_blue 繞半徑 10 m 的圓一圈（約 31 秒）

座標：Gazebo 世界 ENU（x 東、y 北）；heading 0 = 朝東、逆時針為正（與 SDF yaw 相同）。
目標是「瞬移」到路徑上的位置（運動學），輪子不會轉，但從正上方看不出差別，軌跡也完全可控。
需要世界載入 UserCommands system（提供 /world/<world>/set_pose），可用 gz service -l | grep set_pose 確認。
"""
import math
import threading
import time
from collections import namedtuple

State = namedtuple("State", "x y heading vx vy yaw_rate s done")


class Route:
    """由直線與圓弧串成的平面路徑；line / arc 會回傳自己，可以串接。"""

    def __init__(self, start=(0.0, 0.0), heading_deg=0.0):
        self.start = (float(start[0]), float(start[1]))
        self.heading0 = math.radians(heading_deg)
        self.segments = []   # (起點 x, y, heading, 長度, 曲率)
        self._end = (self.start[0], self.start[1], self.heading0)

    @property
    def length(self):
        return sum(seg[3] for seg in self.segments)

    def line(self, length):
        return self._add(length, 0.0)

    def arc(self, radius, angle_deg):
        """angle_deg > 0 左轉（逆時針）、< 0 右轉。"""
        return self._add(radius * abs(math.radians(angle_deg)), math.copysign(1.0 / radius, angle_deg))

    def _add(self, length, curvature):
        x, y, h = self._end
        self.segments.append((x, y, h, float(length), curvature))
        self._end = self._pose_on(x, y, h, length, curvature)
        return self

    @staticmethod
    def _pose_on(x, y, h, s, k):
        if abs(k) < 1e-12:
            return x + s * math.cos(h), y + s * math.sin(h), h
        h2 = h + k * s
        return x + (math.sin(h2) - math.sin(h)) / k, y - (math.cos(h2) - math.cos(h)) / k, h2

    def at(self, s):
        """距離起點 s 公尺處的 (x, y, heading, 曲率)；s 超出範圍時停在端點（曲率 0）。"""
        if not self.segments or s <= 0:
            return self.start[0], self.start[1], self.heading0, 0.0
        for x, y, h, length, k in self.segments:
            if s <= length:
                return (*self._pose_on(x, y, h, s, k), k)
            s -= length
        return (*self._end, 0.0)

    def describe(self):
        """給 meta.json 記錄用。"""
        return {"start_enu_m": self.start, "heading_deg": math.degrees(self.heading0),
                "segments": [{"length_m": round(seg[3], 3), "curvature_1_m": round(seg[4], 5)} for seg in self.segments]}


class MovingTarget:
    def __init__(self, name, route, speed, clock, world="default", z=0.0, rate_hz=30.0):
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.msgs10.pose_pb2 import Pose
        from gz.transport13 import Node

        self.name, self.route, self.speed, self.clock, self.z = name, route, float(speed), clock, z
        self.period = 1.0 / rate_hz
        self.service = f"/world/{world}/set_pose"
        self._Pose, self._Boolean, self._node = Pose, Boolean, Node()
        self._t0 = None
        self._stop = threading.Event()
        self._thread = None
        self.fail_count = 0

    @property
    def duration_s(self):
        return self.route.length / self.speed

    def _set_pose(self, x, y, heading):
        req = self._Pose()
        req.name = self.name
        req.position.x, req.position.y, req.position.z = x, y, self.z
        req.orientation.w, req.orientation.z = math.cos(heading / 2), math.sin(heading / 2)
        ok, rep = self._node.request(self.service, req, self._Pose, self._Boolean, 500)
        if not (ok and rep.data):
            self.fail_count += 1
            if self.fail_count == 1:
                print(f"[sim_target] set_pose 失敗：{self.service}（{self.name}）", flush=True)
        return ok and rep.data

    def place_at_start(self):
        """把目標放到路徑起點；成功回傳 True。"""
        x, y, h, _ = self.route.at(0.0)
        return self._set_pose(x, y, h)

    def _progress(self):
        t = self.clock()
        if self._t0 is None or t is None:
            return 0.0
        return max(0.0, self.speed * (t - self._t0))

    def state(self):
        s = self._progress()
        x, y, h, k = self.route.at(s)
        done = s >= self.route.length
        v = 0.0 if done else self.speed
        return State(x, y, h, v * math.cos(h), v * math.sin(h), v * k, s, done)

    def _loop(self):
        while not self._stop.is_set():
            st = self.state()
            self._set_pose(st.x, st.y, st.heading)
            if st.done:
                break
            time.sleep(self.period)

    def start(self):
        """以目前的模擬時間為起點開始移動。"""
        t0 = None
        for _ in range(50):
            t0 = self.clock()
            if t0 is not None:
                break
            time.sleep(0.1)
        if t0 is None:
            raise RuntimeError("拿不到模擬時間（clock() 一直回傳 None）")
        self._t0 = t0
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


if __name__ == "__main__":
    import sys

    from sim_truth import SimTruth

    name = sys.argv[1] if len(sys.argv) > 1 else "car_blue"
    tr = SimTruth(models=(name,))
    car = MovingTarget(name, Route(start=(0.0, -10.0), heading_deg=0.0).arc(10.0, 360.0), 2.0, tr.latest_stamp)
    print(f"放到起點：{car.place_at_start()}；繞圓約 {car.duration_s:.0f} 秒 …", flush=True)
    car.start()
    while not car.state().done:
        time.sleep(1.0)
        t = tr.truth_at(tr.latest_stamp())[name]
        st = car.state()
        print(f"指令 x {st.x:6.2f} y {st.y:6.2f} | 真值 x {t['e']:6.2f} y {t['n']:6.2f}"
              if t else "還沒收到真值", flush=True)
    car.stop()
    print(f"完成；set_pose 失敗 {car.fail_count} 次", flush=True)
