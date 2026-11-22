"""隊形規劃（C）：Agent 的 Plan（角色）→ 各機目標點 GoalPoint（規格 §10.6）。"""
from .capture import CaptureConfig, CaptureGeometry, CaptureZone, propose_zone
from .formation import FormationConfig, FormationPlanner

__all__ = ["CaptureConfig", "CaptureGeometry", "CaptureZone", "FormationConfig", "FormationPlanner", "propose_zone"]
