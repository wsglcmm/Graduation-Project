"""C 端與 A 的多機 SITL 連線（規格 §10.2 FleetPose、§10.6 GoalPoint）。"""
from .client import DEFAULT_URI, FleetLink, LinkError, LinkStats, PoseRecord
from .messages import FleetPose, GoalPointAck
from .sitl_fleet import SitlFleet

__all__ = ["DEFAULT_URI", "FleetLink", "LinkError", "LinkStats", "PoseRecord", "FleetPose", "GoalPointAck", "SitlFleet"]
