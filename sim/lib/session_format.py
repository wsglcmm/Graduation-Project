"""session_format：模擬紀錄規格（analysis/log_format.md）的共用工具，只用標準函式庫。

    from session_format import JsonlWriter, pose_msg, truth_msg, meta_dict
    w = JsonlWriter("poses.jsonl")
    w.write(pose_msg(t_ns, n, e, d, roll_deg, pitch_deg, yaw_deg))

record_dataset.py（即時錄製）和 tools/convert_session.py（轉換舊 CSV）都用這裡的函式，
確保兩邊寫出的格式完全相同。
"""
import json
import math
from decimal import Decimal

FORMAT_VERSION = 1
SCHEMA_VERSION = "0.1.0"                                               # 規格 §3 Detection
CLASSES = ("car", "person", "truck", "bus", "motorcycle", "bicycle")   # 規格 §3


def wrap_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def euler_to_quat(roll, pitch, yaw):
    """ZYX 歐拉角（弧度）→ 四元數 wxyz（Hamilton），和 C 的 rotations.euler_to_quat 相同。"""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return [cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr, sy * cp * cr - cy * sp * sr]


def quat_to_rpy(w, x, y, z):
    """四元數 wxyz → ZYX 歐拉角（弧度）。"""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def seconds_str_to_ns(stamp):
    """秒字串（例如 "132.2680"）→ 整數奈秒；用 Decimal，同一個字串一定得到同一個整數。"""
    return int(Decimal(stamp.strip()) * 1_000_000_000)


def target_class(name):
    """car_blue → car、person_1 → person；不認得就報錯，避免寫出規格外的類別。"""
    cls = name.split("_")[0]
    if cls not in CLASSES:
        raise ValueError(f"無法從模型名稱 {name!r} 判斷類別（必須是 {CLASSES} 之一）")
    return cls


def pose_msg(t_ns, n, e, d, roll_deg, pitch_deg, yaw_deg, vel_ned=None):
    """規格 §4 Pose。輸入角度為度（flight_api 的單位），輸出弧度並附 quaternion_wxyz。
    只寫 C 的 PoseMessage 允許的欄位（extra="forbid"）。"""
    roll, pitch, yaw = math.radians(roll_deg), math.radians(pitch_deg), wrap_pi(math.radians(yaw_deg))
    msg = {"timestamp_ns": int(t_ns),
           "position_ned_m": {"n": n, "e": e, "d": d},
           "attitude_rad": {"roll": roll, "pitch": pitch, "yaw": yaw},
           "quaternion_wxyz": euler_to_quat(roll, pitch, yaw)}
    if vel_ned is not None and None not in vel_ned:
        msg["velocity_ned_mps"] = dict(zip("ned", vel_ned))
    return msg


def truth_msg(t_ns, name, x, y, z, rpy=None):
    """truth.jsonl 的一行：Gazebo 世界座標 ENU；rpy 為 ENU/FLU 的 (roll, pitch, yaw) 弧度。"""
    msg = {"timestamp_ns": int(t_ns), "name": name, "position_enu_m": {"x": x, "y": y, "z": z}}
    if rpy is not None:
        msg["rpy_rad"] = dict(zip(("roll", "pitch", "yaw"), rpy))
    return msg


def meta_dict(origin_enu, ground_z, drone_name, targets, **extra):
    """meta.json：規格必填欄位 + 其他附加資訊（C 的工具會忽略不認得的欄位）。"""
    return {"format_version": FORMAT_VERSION,
            "ned_origin_enu_m": dict(zip("xyz", origin_enu)),
            "ground_z_enu_m": ground_z,
            "drone_name": drone_name,
            "targets": {n: target_class(n) for n in targets},
            **extra}


class JsonlWriter:
    """一行一筆 JSON，每筆都 flush：錄到一半中斷，已寫的部分仍然完整可讀。"""

    def __init__(self, path):
        self._f = open(path, "w", encoding="utf-8")
        self.count = 0

    def write(self, obj):
        self._f.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
        self._f.flush()
        self.count += 1

    def close(self):
        if not self._f.closed:
            self._f.close()
