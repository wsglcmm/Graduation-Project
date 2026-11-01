"""A 的 3 架 SITL 接上之前的替身（C）：簡化運動學模擬 + 兔子目標路線。

介面刻意和規格 §10.2（FleetPose 輸出）、§10.6（GoalPoint 輸入）相同，
之後換成 A 的 SITL 只需要替換這個模組，Agent、融合、隊形規劃都不用改。
"""
from .fleet_sim import FleetSim, FleetSimConfig
from .scenario import RabbitConfig, RabbitPath

__all__ = ["FleetSim", "FleetSimConfig", "RabbitConfig", "RabbitPath"]
