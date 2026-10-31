"""sim_truth：讀 Gazebo 模型的真實位姿（模擬時間），轉成 PX4 使用的 NED / FRD。

    from sim_truth import SimTruth
    truth = SimTruth()
    t = truth.truth_at(cam.last_stamp)                    # {名稱: {"n","e","d","roll_deg","pitch_deg","yaw_deg"}}
    r = truth.relative_body(cam.last_stamp, "car_blue")   # 目標相對無人機：前/右/下（m，只轉 yaw）
    uv = truth.project(cam.last_stamp, "car_blue")        # 目標地面點在畫面上的真實像素 (u, v, 深度)
    raw = truth.raw_since("car_blue", last_ns)            # Gazebo 原始 ENU 串流（錄製用）

座標：Gazebo 是 ENU（x 東、y 北、z 上）、機體 FLU；這裡一律轉成 NED 與機體 FRD：
n = y、e = x、d = -z；姿態轉成 NED 下的 roll/pitch/yaw（度，yaw 0 = 北、90 = 東）。
原點是 Gazebo 世界原點，與 PX4 EKF 的本地原點可能差幾公分；評估座標轉換請用
relative_ned / relative_body / project（都是目標相對無人機），不受原點差異影響。
raw_since 另外保留 Gazebo 原始的 ENU 位置與四元數（不轉換），給規格的 truth.jsonl 用。
"""
import atexit
import bisect
import math
import threading
from collections import deque

from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

DRONE = "x500_mono_cam_down_0"
TARGETS = ("car_blue", "car_red", "person_1", "person_2")

# 下視鏡頭（x500_mono_cam_down）：理想針孔、無畸變
FX = FY = 540.0                 # 640 / tan(1.74 / 2)
CX, CY = 640.0, 480.0
WIDTH, HEIGHT = 1280, 960
CAM_OFFSET_FRD = (0.0, 0.0, -0.10)   # 鏡頭在模型原點上方 0.10 m（FRD 的 z 向下為正）


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def _quat_to_R(w, x, y, z):
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def _enu_flu_to_ned_frd(R):
    """R_ned_frd = T · R_enu_flu · F，T：ENU→NED，F：FRD→FLU。"""
    # T 交換前兩列並把第三列取負；F 把第二、三欄取負
    rows = (R[1], R[0], tuple(-v for v in R[2]))
    return tuple((r[0], -r[1], -r[2]) for r in rows)


def _rpy_deg(R):
    roll = math.atan2(R[2][1], R[2][2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2][0])))
    yaw = math.atan2(R[1][0], R[0][0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _R_from_rpy_deg(r, p, y):
    r, p, y = (math.radians(v) for v in (r, p, y))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr))


class SimTruth:
    def __init__(self, models=(DRONE,) + TARGETS, world="default", history_s=10.0):
        self.topic = f"/world/{world}/pose/info"
        self.models = tuple(models)
        self._lock = threading.Lock()
        self._hist = {m: deque(maxlen=int(history_s * 120)) for m in self.models}
        self.msg_count = 0
        self._node = Node()
        if not self._node.subscribe(Pose_V, self.topic, self._on_pose):
            raise RuntimeError(f"訂閱失敗：{self.topic}")
        atexit.register(self.release)

    def _on_pose(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nsec
        self.msg_count += 1
        for p in msg.pose:
            h = self._hist.get(p.name)
            if h is None:
                continue
            o = p.orientation
            roll, pitch, yaw = _rpy_deg(_enu_flu_to_ned_frd(_quat_to_R(o.w, o.x, o.y, o.z)))
            # 前 7 個欄位給 _at 內插（NED、度）；後面是原始值：stamp_ns、ENU 位置、四元數 wxyz
            rec = (stamp, p.position.y, p.position.x, -p.position.z, roll, pitch, yaw,
                   stamp_ns, p.position.x, p.position.y, p.position.z, o.w, o.x, o.y, o.z)
            with self._lock:
                if not h or stamp > h[-1][0]:
                    h.append(rec)

    # ---------- 查詢 ----------
    def _at(self, name, stamp, max_gap):
        with self._lock:
            h = list(self._hist.get(name, ()))
        if not h:
            return None
        ts = [r[0] for r in h]
        i = bisect.bisect_left(ts, stamp)
        keys = ("n", "e", "d", "roll_deg", "pitch_deg", "yaw_deg")
        if i == 0 or i == len(h):
            r = h[0] if i == 0 else h[-1]
            if abs(r[0] - stamp) > max_gap:
                return None
            out = dict(zip(keys, r[1:]))
            out["stamp_used"] = r[0]
            return out
        a, b = h[i - 1], h[i]
        w = (stamp - a[0]) / (b[0] - a[0])
        out = {k: a[j] + w * (b[j] - a[j]) for j, k in enumerate(keys[:5], start=1)}
        out["yaw_deg"] = _wrap(a[6] + w * _wrap(b[6] - a[6]))
        out["stamp_used"] = stamp
        return out

    def raw_since(self, name, stamp_ns):
        """回傳 stamp_ns 之後（不含）的原始紀錄：[(stamp_ns, (x, y, z), (w, x, y, z)), ...]，Gazebo ENU / FLU。"""
        with self._lock:
            h = list(self._hist.get(name, ()))
        return [(r[7], r[8:11], r[11:15]) for r in h if r[7] > stamp_ns]

    def latest_stamp(self):
        with self._lock:
            s = [h[-1][0] for h in self._hist.values() if h]
        return max(s) if s else None

    def truth_at(self, stamp, max_gap=0.1):
        """回傳 {模型名稱: {"n","e","d","roll_deg","pitch_deg","yaw_deg","stamp_used"}}；查不到為 None。"""
        return {m: self._at(m, stamp, max_gap) for m in self.models}

    def relative_ned(self, stamp, name, max_gap=0.1):
        """目標（模型原點）相對無人機（模型原點）的真實位移，NED（m）。"""
        t, dr = self._at(name, stamp, max_gap), self._at(DRONE, stamp, max_gap)
        if t is None or dr is None:
            return None
        return (t["n"] - dr["n"], t["e"] - dr["e"], t["d"] - dr["d"])

    def relative_body(self, stamp, name, max_gap=0.1):
        """目標相對無人機，轉到機頭方向（只用 yaw，不含傾斜）：(前, 右, 下)。"""
        rel, dr = self.relative_ned(stamp, name, max_gap), self._at(DRONE, stamp, max_gap)
        if rel is None:
            return None
        dn, de, dd = rel
        y = math.radians(dr["yaw_deg"])
        return (dn * math.cos(y) + de * math.sin(y), -dn * math.sin(y) + de * math.cos(y), dd)

    def project(self, stamp, name, height=0.0, max_gap=0.1):
        """把目標上「離地 height 公尺」的點，用真實姿態（含傾斜）投影到下視鏡頭畫面。

        回傳 (u, v, depth_m)；點在鏡頭後方或查不到真值回傳 None。u、v 可能超出畫面範圍。
        """
        t, dr = self._at(name, stamp, max_gap), self._at(DRONE, stamp, max_gap)
        if t is None or dr is None:
            return None
        R = _R_from_rpy_deg(dr["roll_deg"], dr["pitch_deg"], dr["yaw_deg"])   # 機體 FRD → NED
        cam = [dr[k] + sum(R[i][j] * CAM_OFFSET_FRD[j] for j in range(3))
               for i, k in enumerate(("n", "e", "d"))]
        diff = (t["n"] - cam[0], t["e"] - cam[1], t["d"] - height - cam[2])
        pb = [sum(R[j][i] * diff[j] for j in range(3)) for i in range(3)]     # NED → 機體 FRD
        xc, yc, zc = pb[1], -pb[0], pb[2]       # 畫面右 = 機體右、畫面下 = 機體後、光軸 = 機體下
        if zc <= 0.1:
            return None
        return (CX + FX * xc / zc, CY + FY * yc / zc, zc)

    def release(self):
        if not getattr(self, "_released", False):
            self._released = True
            self._node.unsubscribe(self.topic)   # Node 物件保留到程式結束，不在這裡銷毀
