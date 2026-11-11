"""端到端 Demo：鳥瞰圖 + Agent 指令即時面板（今晚可跑的版本）。

    python -m demo.run                          # 規則式指揮官（不需要 API 金鑰）
    python -m demo.run --planner llm            # 接 LLM（預設 Gemini，需要 .env 裡的 GEMINI_API_KEY）
    python -m demo.run --headless --duration 90 # 不開視窗，跑完輸出指標（報告用）
    python -m demo.run --cmd "15=三架一起把那台車包圍起來"  # 第 15 秒自動送出指令（可重複，報告用可重現）

畫面左邊是鳥瞰圖：彩色三角形 = 無人機（半透明框 = 相機視野），灰色 × = 兔子真值，
黑點與圓圈 = 融合 + 卡爾曼後的估計與 uncertainty_m，小空心圓 = 隊形規劃給的目標點。
右邊是 Agent 指令面板：每一筆計畫的時間、規劃者（LLM / 規則）、延遲、模式、角色與理由。
指令輸入：在執行程式的終端機（PowerShell / VS Code 終端機）直接打字按 Enter，支援中文輸入法；
視窗下方的輸入框也可以用，但 matplotlib 不支援輸入法，只能打英文。

目前兔子路線是合成的、無人機是運動學替身；之後換成「B 的影片偵測 + 地面標定」與「A 的 3 架 SITL」，
介面相同（規格 §10），Agent 與融合程式不用改。
"""
import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

from coord_transform.config import load_camera_config, load_error_model
from demo.engine import DemoEngine
from llm_agent.commander import Commander
from sim2d import RabbitConfig

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"uav1": "#2a78d6", "uav2": "#eb6834", "uav3": "#1baf7a"}  # 類別色（已驗證的色盲安全順序）
ROLE_ZH = {"chaser": "追擊", "blocker_left": "左翼", "blocker_right": "右翼", "overwatch": "高空監看", "searcher": "搜尋", "idle": "待命"}
MODE_ZH = {"search": "搜尋", "track": "追蹤", "encircle": "包圍", "hold": "懸停", "return_home": "返航"}
STATUS_ZH = {"tracking": "追蹤中", "predicted": "暫時看不到（外推中）", "lost": "遺失"}


def make_commander(planner: str, period_s: float) -> Commander:
    ids = ["uav1", "uav2", "uav3"]
    if planner == "rule":
        return Commander(ids, llm=None, period_s=period_s)
    from llm_agent.llm_client import create_llm_client, load_dotenv

    load_dotenv(ROOT / ".env")
    return Commander(ids, llm=create_llm_client(), period_s=period_s)  # AGENT_PROVIDER：gemini（預設）/ anthropic


def event_line(ev) -> str:
    t = ev.timestamp_ns / 1e9
    who = "LLM" if ev.planner == "llm" else "規則"
    if ev.trigger == "operator":
        who += "［操作員］"
    if not ev.valid:
        return f"[{t:5.1f}s] {who} {ev.latency_ms:5.0f}ms 不合法：{ev.error[:40]}"
    p = ev.plan
    roles = " ".join(f"{a['drone_id'][-1]}:{ROLE_ZH[a['role']]}" for a in p["assignments"])
    head = f"[{t:5.1f}s] {who} {ev.latency_ms:5.0f}ms  {MODE_ZH[p['mode']]}  {roles}"
    if ev.plan_id is None:  # 已被操作員新指令取代
        head += "（未套用）"
    return head + f"\n          {p['reason']}"


def parse_scripted(items: List[str]) -> List[Tuple[float, str]]:
    """--cmd "秒數=指令" → [(秒數, 指令)]，依時間排序。"""
    out = []
    for item in items or []:
        t, sep, text = item.partition("=")
        if not sep or not text.strip():
            raise SystemExit(f"--cmd 格式應為 秒數=指令，收到：{item!r}")
        out.append((float(t), text.strip()))
    return sorted(out)


def feed_scripted(engine: DemoEngine, scripted: List[Tuple[float, str]]) -> None:
    """到時間的預排指令送給指揮官（會從 scripted 移除）。"""
    while scripted and engine.t_ns / 1e9 >= scripted[0][0]:
        engine.commander.command(scripted.pop(0)[1])


def start_terminal_input(engine: DemoEngine) -> None:
    """背景執行緒讀終端機輸入，每行送一次指令。終端機支援中文輸入法，matplotlib 輸入框不支援。"""
    if sys.stdin is None or not sys.stdin.isatty():
        return

    def loop():
        for line in sys.stdin:
            text = line.strip()
            if text:
                engine.commander.command(text)
                print(f"  → 已送出指令：{text}", flush=True)

    threading.Thread(target=loop, name="terminal-input", daemon=True).start()
    print("可以在這個終端機輸入指令（支援中文），按 Enter 送出；關閉視窗結束。", flush=True)


def run_headless(engine: DemoEngine, duration_s: float, realtime: bool, scripted: List[Tuple[float, str]]) -> None:
    if realtime:  # 照真實時間跑時才有空打字；快轉模式請用 --cmd 預排
        start_terminal_input(engine)
    steps = int(duration_s / engine.dt)
    for _ in range(steps):
        t0 = time.monotonic()
        feed_scripted(engine, scripted)
        for ev in engine.step():
            print(event_line(ev))
        if realtime:  # LLM 模式要照真實時間跑，延遲才有意義
            time.sleep(max(0.0, engine.dt - (time.monotonic() - t0)))
    engine.commander.wait_idle()


def run_gui(engine: DemoEngine, speed: float, scripted: List[Tuple[float, str]]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Circle, Polygon
    from matplotlib.widgets import TextBox

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "Heiti TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(14, 8), facecolor="#fcfcfb")
    ax = fig.add_axes([0.04, 0.12, 0.55, 0.82])
    panel = fig.add_axes([0.62, 0.12, 0.36, 0.82])
    panel.axis("off")
    box_ax = fig.add_axes([0.10, 0.03, 0.49, 0.05])  # 左邊留空間給標籤
    box = TextBox(box_ax, "指令（英文）", initial="")
    box.on_submit(lambda text: (engine.commander.command(text), box.set_val("")) if text.strip() else None)
    start_terminal_input(engine)

    ax.set_facecolor("#fcfcfb")
    ax.set_aspect("equal")
    ax.grid(True, color="#e4e3df", linewidth=1)
    ax.set_xlabel("東 e (m)")
    ax.set_ylabel("北 n (m)")
    lim = engine.rabbit.cfg.arena_radius_m + 30
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    trail_x, trail_y = [], []
    (trail,) = ax.plot([], [], color="#b5b4ae", linewidth=1)
    (truth_pt,) = ax.plot([], [], "x", color="#52514e", markersize=10, mew=2, label="目標真值")
    (est_pt,) = ax.plot([], [], "o", color="#0b0b0b", markersize=6, label="估計位置")
    unc = Circle((0, 0), 1, fill=False, color="#0b0b0b", linewidth=1, linestyle="--")
    ax.add_patch(unc)
    drone_art, fov_art, goal_art = {}, {}, {}
    for did, col in COLORS.items():
        (drone_art[did],) = ax.plot([], [], marker=(3, 0, 0), markersize=14, color=col, linestyle="none", label=did)
        fov_art[did] = Polygon([[0, 0]], closed=True, alpha=0.10, color=col)
        ax.add_patch(fov_art[did])
        (goal_art[did],) = ax.plot([], [], "o", mfc="none", mec=col, markersize=8)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    title = ax.set_title("", loc="left", fontsize=12)
    lines = []

    def update(_):
        for _ in range(max(1, int(round(speed)))):
            feed_scripted(engine, scripted)
            for ev in engine.step():
                lines.insert(0, event_line(ev))
        del lines[9:]
        tn, te = engine.truth
        trail_x.append(te)
        trail_y.append(tn)
        trail.set_data(trail_x[-400:], trail_y[-400:])
        truth_pt.set_data([te], [tn])
        st = engine.target_state
        if st:
            p = st["position_world_ned_m"]
            est_pt.set_data([p["e"]], [p["n"]])
            unc.center, unc.radius = (p["e"], p["n"]), max(st["uncertainty_m"], 0.3)
        for d in engine.sim.states():
            drone_art[d.drone_id].set_data([d.e], [d.n])
            drone_art[d.drone_id].set_marker((3, 0, -math.degrees(d.yaw_rad)))
            fov_art[d.drone_id].set_xy([(e, n) for n, e in engine.view.footprint_corners(d)])
        for g in engine.goals:
            gp = g["position_world_ned_m"]
            goal_art[g["drone_id"]].set_data([gp["e"]], [gp["n"]])
        plan = engine.commander.current
        status = STATUS_ZH[st["status"]] if st else "尚未發現"
        seen = ",".join(st["observed_by"]) if st and st["observed_by"] else "無"
        title.set_text(f"t = {engine.t_ns / 1e9:5.1f} s   目標：{status}（看到的：{seen}）   模式：{MODE_ZH[plan.mode] if plan else '—'}")
        panel.clear()
        panel.axis("off")
        llm = engine.commander.llm
        name = "規則式"
        if llm:
            name = "LLM（" + getattr(llm, "name", "") + "）"
            used = getattr(llm, "last_model", None)
            if used and used != getattr(llm, "model", used):
                name += f"　備援中：{used}"
        panel.text(0, 1.0, f"Agent 指令面板　規劃者：{name}", fontsize=12, weight="bold", va="top")
        c = engine.commander
        if c.directive:
            lock = "（Agent 暫停，等待新指令）" if c.paused else ""
            panel.text(0, 0.955, f"操作員有效指令：{c.directive_text}　→ {MODE_ZH[c.directive]}{lock}", fontsize=9, color="#b4380b", va="top")
        panel.text(0, 0.92, "\n".join(lines) or "（等待第一個計畫）", fontsize=9, va="top", linespacing=1.45)
        m = engine.summary()
        enc = f"{m['time_to_encircle_s']:.1f} s" if m["time_to_encircle_s"] else "尚未"
        panel.text(0, 0.02, f"包圍完成：{enc}　視野內比例：{m['visible_fraction']:.0%}　最小機間距離：{m['min_drone_distance_m']:.1f} m",
                   fontsize=9, color="#52514e")

    anim = FuncAnimation(fig, update, interval=int(engine.dt * 1000), cache_frame_data=False)
    fig._anim = anim  # 防止被垃圾回收
    plt.show()


def main(argv=None):
    ap = argparse.ArgumentParser(description="多機包圍 Demo（鳥瞰圖 + Agent 指令面板）")
    ap.add_argument("--planner", choices=("rule", "llm"), default="rule")
    ap.add_argument("--period", type=float, default=2.0, help="多久問一次 Agent（秒）")
    ap.add_argument("--headless", action="store_true", help="不開視窗，跑完輸出指標")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--speed", type=float, default=1.0, help="畫面模式的播放倍速")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="指標與事件輸出資料夾（預設 output/demo/<時間>）")
    ap.add_argument("--cmd", action="append", default=[], metavar="秒數=指令", help="預排操作員指令，可重複（例如 15=三架包圍那台車）")
    ap.add_argument("--fleet", choices=("sim", "sitl"), default="sim",
                    help="sim = 內建替身；sitl = 連 A 的 3 架 SITL（agent_server，或 python -m fleet_link.fake_server）")
    ap.add_argument("--uri", default="ws://127.0.0.1:8765", help="A 的 agent_server 位址（--fleet sitl 時使用）")
    args = ap.parse_args(argv)

    cam = load_camera_config(ROOT / "config" / "camera.yaml")
    err = load_error_model(ROOT / "config" / "error_model.yaml")
    scripted = parse_scripted(args.cmd)
    commander = make_commander(args.planner, args.period)
    link, fleet = None, None
    if args.fleet == "sitl":
        from fleet_link import FleetLink, SitlFleet

        link = FleetLink(args.uri).start()
        fleet = SitlFleet(link)
        fleet.ensure_airborne(commander.drone_ids)
        print(f"已連上 {args.uri}：{', '.join(s.drone_id for s in fleet.states())}", flush=True)
    engine = DemoEngine(cam, err, commander, rabbit=RabbitConfig(seed=args.seed), seed=args.seed, fleet=fleet)
    try:
        if args.headless:
            run_headless(engine, args.duration, realtime=args.planner == "llm" or link is not None, scripted=scripted)
        else:
            run_gui(engine, 1.0 if link is not None else args.speed, scripted)  # SITL 照真實時間跑，不能快轉
    finally:
        commander.close()
        if link is not None:
            print("連線統計：" + json.dumps(link.stats.summary(), ensure_ascii=False))
            link.close()

    out = Path(args.out) if args.out else ROOT / "output" / "demo" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    summary = engine.summary()
    (out / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out / "events.jsonl", "w", encoding="utf-8") as f:
        for ev in commander.events:
            f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
    print("\n指標：" + json.dumps(summary, ensure_ascii=False))
    print(f"輸出位置：{out}")
    return summary


if __name__ == "__main__":
    main()
