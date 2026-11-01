"""多機 Agent 測試共用的假資料。"""
import json
import math

import pytest

from fusion import DroneState

IDS = ["uav1", "uav2", "uav3"]
S = 1_000_000_000
DRONES = [DroneState("uav1", -20, 0, 25), DroneState("uav2", 5, -15, 30), DroneState("uav3", 5, 15, 35)]


def target(n=0.0, e=0.0, vn=4.0, ve=0.0, status="tracking", tid=1, seen=0):
    return {"target_id": tid, "class": "car", "status": status, "position_world_ned_m": {"n": n, "e": e},
            "velocity_world_ned_mps": {"n": vn, "e": ve}, "uncertainty_m": 0.8, "observed_by": ["uav1"], "last_seen_ns": seen}


ENC = {"mode": "encircle", "target_id": 1,
       "assignments": [{"drone_id": "uav1", "role": "chaser"}, {"drone_id": "uav2", "role": "blocker_left"}, {"drone_id": "uav3", "role": "blocker_right"}],
       "formation": {"radius_m": 15.0}, "reason": "包圍"}
