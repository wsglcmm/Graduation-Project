#!/usr/bin/env bash
# 啟動 Gazebo（無螢幕繪圖）+ PX4 SITL 下視鏡頭機型。請在 tmux px4 內執行：
#   bash /mnt/2disk/vicelead/code/start_sim.sh [world.sdf]
source /mnt/2disk/vicelead/env.sh
micromamba activate px4
WORLD=${1:-$WS/code/worlds/default_targets.sdf}

echo "== 關閉舊的模擬 =="
pkill -u "$USER" -f "gz sim" || true
pkill -u "$USER" -x px4 || true
sleep 2

echo "== 啟動 Gazebo：$WORLD =="
cd "$WS/src/PX4-Autopilot" || exit 1
source build/px4_sitl_default/rootfs/gz_env.sh
gz sim -s -r --headless-rendering -v 3 "$WORLD" > "$WS/tmp/gz.log" 2>&1 &
for i in $(seq 1 60); do
  gz topic -l 2>/dev/null | grep -q "^/world/.*/clock" && break
  sleep 1
done
echo "== Gazebo 已就緒（等了 ${i} 秒），log：$WS/tmp/gz.log =="

echo "== 啟動 PX4 =="
PX4_GZ_WORLD=default HEADLESS=1 make px4_sitl gz_x500_mono_cam_down
