#!/usr/bin/env python3
"""Phase 0：MAVSDK 基本指令測試 arm -> takeoff -> land"""
import asyncio
from mavsdk_grpc import System

TAKEOFF_ALT = 5.0  # 公尺

async def print_status_text(drone):
    try:
        async for st in drone.telemetry.status_text():
            print(f"[PX4] {st.type}: {st.text}")
    except asyncio.CancelledError:
        return

async def run():
    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    status_task = asyncio.ensure_future(print_status_text(drone))

    print("等待連線...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("-- 已連線")
            break

    print("等待定位就緒...")
    async for h in drone.telemetry.health():
        if h.is_global_position_ok and h.is_home_position_ok:
            print("-- 定位就緒")
            break

    await drone.action.set_takeoff_altitude(TAKEOFF_ALT)
    print("-- Arm"); await drone.action.arm()
    print("-- Takeoff"); await drone.action.takeoff()

    async for pos in drone.telemetry.position():
        print(f"   相對高度 {pos.relative_altitude_m:5.2f} m")
        if pos.relative_altitude_m >= TAKEOFF_ALT * 0.95:
            break
    print("-- 到達高度，懸停 5 秒"); await asyncio.sleep(5)

    print("-- Land"); await drone.action.land()
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("-- 已落地")
            break
    async for armed in drone.telemetry.armed():
        if not armed:
            print("-- 已自動上鎖 (disarmed)，測試完成")
            break
    status_task.cancel()

if __name__ == "__main__":
    asyncio.run(run())
