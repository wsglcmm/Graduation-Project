"""端到端 Demo：鳥瞰圖 + Agent 指令即時面板（今晚可跑的版本）。

    python -m demo.run                          # 規則式指揮官（不需要 API 金鑰）
    python -m demo.run --planner llm            # 接 LLM（預設 Gemini，需要 .env 裡的 GEMINI_API_KEY）
    python -m demo.run --headless --duration 90 # 不開視窗，跑完輸出指標（報告用）
    python -m demo.run --cmd "15=三架一起把那台車包圍起來"  # 第 15 秒自動送出指令（可重複，報告用可重現）
    python -m demo.run --target evader --strategy capture   # 策略場景：會逃的目標 + 驅趕 → 抓拍（無人機 3 m/s）

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
from sim2d import EvaderConfig, RabbitConfig

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"uav1": "#2a78d6", "uav2": "#eb6834", "uav3": "#1baf7a"}  # 類別色（已驗證的色盲安全順序）
ROLE_ZH = {"chaser": "追擊", "blocker_left": "左翼", "blocker_right": "右翼", "overwatch": "高空監看", "searcher": "搜尋", "idle": "待命",
           "herder": "驅趕", "capturer": "抓拍", "blocker": "封鎖"}
MODE_ZH = {"search": "搜尋", "track": "追蹤", "encircle": "包圍", "capture": "驅趕抓拍", "hold": "懸停", "return_home": "返航"}
STATUS_ZH = {"tracking": "追蹤中", "predicted": "暫時看不到（外推中）", "lost": "遺失"}


def make_commander(planner: str, period_s: float, strategy: str = "encircle", agents: str = "single",
                   drone_planner: str = "rule", drone_period_s: float = 6.0):
    """建立指揮層。

    agents：single = 單一指揮官（對照組）；multi = 協調 Agent + 每架一個各機 Agent（老師 11/16 的建議）。
    planner：協調 Agent（或單一指揮官）用 LLM 還是規則式。drone_planner：各機 Agent 用 LLM 還是規則式。
    strategy：規則式在沒有操作員指令時的預設策略（LLM 模式時只影響失效時的規則式備援）。
    """
    ids = ["uav1", "uav2", "uav3"]
    llm = drone_llms = None
    if planner == "llm" or (agents == "multi" and drone_planner == "llm"):
        from llm_agent.llm_client import create_llm_client, load_dotenv

        load_dotenv(ROOT / ".env")  # AGENT_PROVIDER：gemini（預設）/ anthropic
        llm = create_llm_client() if planner == "llm" else None
        if agents == "multi" and drone_planner == "llm":
            drone_llms = {d: create_llm_client() for d in ids}  # 每架各自一個連線，互不阻塞
    commander = Commander(ids, llm=llm, period_s=period_s, strategy=strategy)
    if agents == "single":
        return commander
    from llm_agent.team import Team

    return Team(commander, drone_llms=drone_llms, drone_period_s=drone_period_s)


def drone_event_line(dev) -> str:
    """各機 Agent 的一筆決策。"""
    from llm_agent.drone_actions import ACTION_ZH

    t = dev.timestamp_ns / 1e9
    who = f"{dev.drone_id} {ROLE_ZH.get(dev.role, dev.role)}Agent（{'LLM' if dev.planner == 'llm' else '規則'}）"
    if not dev.valid:
        return f"[{t:5.1f}s] {who} 不合法、未送出：{(dev.error or '')[:50]}"
    lat = f" {dev.latency_ms:.0f}ms" if dev.planner == "llm" else ""
    return f"[{t:5.1f}s] {who}{lat} → {dev.action}（{ACTION_ZH.get(dev.action, '')}）：{dev.reason}"


def action_line(r) -> str:
    """動作日誌一筆有結果時的一行摘要：哪一架、角色、驗證結果、原因。"""
    mark = {"succeeded": "✓", "failed": "✗", "rejected": "⛔"}.get(r.result, "·")
    t = r.t_start_s
    who = f"{r.drone_id}（{ROLE_ZH.get(r.role, r.role or '—')}）" if r.drone_id else "（整個計畫）"
    extra = f"，A 修改 {r.adjusted} 次" if r.adjusted else ""
    if r.command_consistency is not None:
        extra += f"，指令一致 {r.command_consistency:.0%}"
    return f"[{t:5.1f}s] {mark} 動作 {r.action_id} {who} {r.status_zh}：{r.why}{extra}"


def capture_line(c) -> str:
    who = f"{c.drone_id}（{ROLE_ZH.get(c.role, c.role or '—')}）"
    return f"[{c.t_s:5.1f}s] ★ 抓拍成功：{who} 目標在畫面中央（偏移 {c.pixel_offset[0]:+.0f}, {c.pixel_offset[1]:+.0f} px，約 {c.target_px:.0f} px）"


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
    if getattr(ev, "note", None):
        head += f"\n          {ev.note}"
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


def load_video_scene(args):
    """影片場景：讀地面標定與 B 的偵測，回傳（VideoTarget 或 None、給網頁的影片資訊、標定檔存檔位置）。"""
    import numpy as np

    from coord_transform.homography import GroundHomography, load_video_ground
    from demo.video_scene import load_video_target

    vdir = Path(args.video)
    video_id = vdir.name
    ground_path = Path(args.ground) if args.ground else ROOT / "config" / f"video_ground_{video_id}.json"
    det = vdir / "detections.jsonl"
    if args.calibrate:
        if not args.video_file:
            raise SystemExit("--calibrate 需要 --video-file（背景影片）")
        args.web = True
        return None, None, ground_path
    if not det.exists():
        raise SystemExit(f"找不到 {det}（B 的偵測輸出）")
    first = json.loads(det.read_text(encoding="utf-8").splitlines()[0])
    if ground_path.exists():
        ground, provisional = load_video_ground(ground_path), False
    elif args.ground_scale:  # 暫定：畫面中心為原點、北朝上，只為了先看流程
        center = (first.get("image_width", 1920) / 2, first.get("image_height", 1080) / 2)
        ground, provisional = GroundHomography.from_scale(args.ground_scale, center, video_id=video_id), True
    else:
        raise SystemExit(f"找不到地面標定檔 {ground_path}。先用標定工具：python -m demo.run --video {vdir} "
                         f"--video-file <影片.mp4> --calibrate，或暫時加 --ground-scale 0.05 只看流程")
    target, tracks, provisional_ids = load_video_target(det, ground, args.video_track)
    print(f"影片場景：{video_id}，{len(tracks)} 條軌跡，目標 #{target.track.track_id}（{target.track.cls}，"
          f"{target.track.duration_s:.0f} s）" + ("；track_id 為 C 的暫代關聯" if provisional_ids else "")
          + ("；地面標定為暫定值" if provisional else ""), flush=True)
    info = {"src": "/video", "video_id": video_id, "width": first["image_width"], "height": first["image_height"],
            "ground_to_pixel": np.linalg.inv(ground.H).tolist(), "track_id": target.track.track_id,
            "provisional_ids": provisional_ids, "provisional_ground": provisional} if args.video_file else None
    return target, info, ground_path


def print_step_events(engine: DemoEngine) -> None:
    """把這一步新的事件印在終端機（網頁模式也同時保留文字紀錄）。"""
    for ev in engine.commander.events[-1:]:
        if ev.timestamp_ns == engine.t_ns:
            print(event_line(ev))
    for dev in engine.new_drone_events:
        print(drone_event_line(dev))
    for cap in engine.new_captures:
        print(capture_line(cap))
    for rec in engine.ledger.newly_resolved:
        print(action_line(rec))


def run_headless(engine: DemoEngine, duration_s: float, realtime: bool, scripted: List[Tuple[float, str]]) -> None:
    if realtime:  # 照真實時間跑時才有空打字；快轉模式請用 --cmd 預排
        start_terminal_input(engine)
    steps = int(duration_s / engine.dt)
    for _ in range(steps):
        t0 = time.monotonic()
        feed_scripted(engine, scripted)
        for ev in engine.step():
            print(event_line(ev))
        for dev in engine.new_drone_events:
            print(drone_event_line(dev))
        for cap in engine.new_captures:
            print(capture_line(cap))
        for rec in engine.ledger.newly_resolved:
            print(action_line(rec))
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
    zone_art = Circle((0, 0), 1, fill=True, alpha=0.12, color="#b4380b", visible=False, label="捕捉區")
    ax.add_patch(zone_art)
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
            for dev in engine.new_drone_events:
                lines.insert(0, drone_event_line(dev))
            for cap in engine.new_captures:
                lines.insert(0, capture_line(cap))
            for rec in engine.ledger.newly_resolved:
                lines.insert(0, action_line(rec))
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
        zone = engine.planner.capture_zone
        zone_art.set_visible(zone is not None)
        if zone is not None:
            zone_art.center, zone_art.radius = (zone.e, zone.n), zone.radius_m
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
        cap = f"{m['time_to_capture_s']:.1f} s" if m["time_to_capture_s"] else "尚未"
        panel.text(0, 0.02, f"抓拍：{cap}　包圍完成：{enc}　視野內比例：{m['visible_fraction']:.0%}　最小機間距離：{m['min_drone_distance_m']:.1f} m",
                   fontsize=9, color="#52514e")

    anim = FuncAnimation(fig, update, interval=int(engine.dt * 1000), cache_frame_data=False)
    fig._anim = anim  # 防止被垃圾回收
    plt.show()


SITL_HINT = ("檢查：(1) 用假伺服器測試時，要先在另一個終端機執行 python -m fleet_link.fake_server 並保持開著；"
             "(2) 連 A 的 SITL 時，A 的 agent_server 要先啟動，SSH 通道（-L 8765:127.0.0.1:8765）要建立且視窗不能關；"
             "(3) --uri 是否正確")


def connect_sitl(uri: str, commander: Commander, max_speed_mps: float = 3.0):
    """連 A 的 agent_server 並確認 3 架都在空中。失敗時印一行原因 + 檢查清單後結束，不印 traceback。"""
    from fleet_link import FleetLink, LinkError, SitlFleet

    link = None
    try:
        link = FleetLink(uri).start()
        fleet = SitlFleet(link, max_speed_mps=max_speed_mps)
        fleet.ensure_airborne(commander.drone_ids)
    except LinkError as e:
        if link is not None:
            link.close()
        commander.close()
        raise SystemExit(f"\n[連線失敗] {e}\n{SITL_HINT}") from None
    print(f"已連上 {uri}：{', '.join(s.drone_id for s in fleet.states())}", flush=True)
    return link, fleet


def main(argv=None):
    ap = argparse.ArgumentParser(description="多機包圍 Demo（鳥瞰圖 + Agent 指令面板）")
    ap.add_argument("--planner", choices=("rule", "llm"), default="rule")
    ap.add_argument("--period", type=float, default=2.0, help="多久問一次 Agent（秒）")
    ap.add_argument("--agents", choices=("multi", "single"), default="multi",
                    help="multi = 協調 Agent + 每架一個各機 Agent；single = 單一指揮官（對照組）")
    ap.add_argument("--drone-planner", choices=("rule", "llm"), default="rule", help="各機 Agent 用規則式或 LLM（--agents multi）")
    ap.add_argument("--drone-period", type=float, default=6.0, help="各機 Agent 多久問一次 LLM（秒）；3 架同時呼叫，免費額度建議 ≥ 6")
    ap.add_argument("--headless", action="store_true", help="不開視窗，跑完輸出指標")
    ap.add_argument("--duration", type=float, default=90.0)
    ap.add_argument("--speed", type=float, default=1.0, help="畫面模式的播放倍速")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="指標與事件輸出資料夾（預設 output/demo/<時間>）")
    ap.add_argument("--cmd", action="append", default=[], metavar="秒數=指令", help="預排操作員指令，可重複（例如 15=三架包圍那台車）")
    ap.add_argument("--fleet", choices=("sim", "sitl"), default="sim",
                    help="sim = 內建替身；sitl = 連 A 的 3 架 SITL（agent_server，或 python -m fleet_link.fake_server）")
    ap.add_argument("--uri", default="ws://127.0.0.1:8765", help="A 的 agent_server 位址（--fleet sitl 時使用）")
    ap.add_argument("--max-speed", type=float, default=3.0,
                    help="A 的速度上限 m/s（--fleet sitl 時使用），要和 A 的 agent_server --max-speed 相同")
    ap.add_argument("--target", choices=("rabbit", "evader"), default="rabbit",
                    help="rabbit = 不理無人機的兔子路線；evader = 會逃的目標（策略場景，驅趕才有意義）")
    ap.add_argument("--strategy", choices=("encircle", "capture"), default=None,
                    help="規則式的預設策略：包圍 / 驅趕抓拍（預設 encircle；--video 時預設 capture）")
    ap.add_argument("--drone-speed", type=float, default=None,
                    help="替身無人機速度上限 m/s（預設：evader 時 3，對齊 A 的 SITL；rabbit 時 8）")
    ap.add_argument("--web", action="store_true", help="網頁版 Demo 畫面（即時指令翻譯、角色卡片、動作日誌），瀏覽器開 http://127.0.0.1:8000/")
    ap.add_argument("--port", type=int, default=8000, help="--web 的埠")
    ap.add_argument("--no-browser", action="store_true", help="--web 時不要自動開瀏覽器")
    ap.add_argument("--video", default=None, metavar="資料夾",
                    help="影片場景：B 的偵測資料夾（內含 detections.jsonl），例如 data/videos/road01；目標改用影片中的車")
    ap.add_argument("--video-file", default=None, help="背景影片 mp4（--web 時疊圖用），例如 \"data/videos/road 01_1080p50.mp4\"")
    ap.add_argument("--ground", default=None, help="地面標定檔（預設 config/video_ground_<影片資料夾名>.json）")
    ap.add_argument("--ground-scale", type=float, default=None, metavar="m/px",
                    help="沒有標定檔時暫用的比例（只看流程，距離不準，畫面會標示「暫定」）")
    ap.add_argument("--video-track", type=int, default=None, help="指定影片中當目標的 track_id（預設：路口 40 m 內、真的在移動的時間最長的車）")
    ap.add_argument("--calibrate", action="store_true", help="開地面標定工具（需要 --video 與 --video-file）")
    args = ap.parse_args(argv)
    if args.strategy is None:
        args.strategy = "capture" if args.video else "encircle"  # 影片場景的重點是預測攔截 → 抓拍

    cam = load_camera_config(ROOT / "config" / "camera.yaml")
    err = load_error_model(ROOT / "config" / "error_model.yaml")
    scripted = parse_scripted(args.cmd)
    commander = make_commander(args.planner, args.period, args.strategy, args.agents, args.drone_planner, args.drone_period)
    link, fleet = None, None
    if args.fleet == "sitl":
        link, fleet = connect_sitl(args.uri, commander, args.max_speed)
    evader = EvaderConfig(seed=args.seed) if args.target == "evader" else None
    speed = args.drone_speed or (3.0 if evader is not None else None)
    video_target, video_info, capture_cfg, ground_out = None, None, None, None
    if args.video:
        video_target, video_info, ground_out = load_video_scene(args)
        if video_target is not None:
            from planner.capture import CaptureConfig

            # 影片目標不會逃、也趕不動：捕捉區放在它前進的方向，轉彎時每 3 s 重新預測（預測攔截）
            capture_cfg = CaptureConfig(zone_direction="velocity", replan_zone_s=3.0)
            speed = args.drone_speed or 5.0
    engine = DemoEngine(cam, err, commander, rabbit=RabbitConfig(seed=args.seed), seed=args.seed, fleet=fleet,
                        evader=evader, drone_max_speed_mps=speed, target_source=video_target, capture_cfg=capture_cfg)
    if video_info is not None:
        video_info["speed"] = 1.0 if link is not None else args.speed
    try:
        if args.web:
            from demo.web import run_web

            run_web(engine, 1.0 if link is not None else args.speed, scripted, port=args.port,
                    backend="sitl" if link is not None else "sim", link=link, open_browser=not args.no_browser,
                    on_step=print_step_events, duration_s=args.duration if args.headless else None, video=video_info,
                    video_file=args.video_file, ground_out=ground_out,
                    path=f"/calibrate?video_id={Path(args.video).name}" if args.calibrate else "/")
        elif args.headless:
            llm_used = args.planner == "llm" or (args.agents == "multi" and args.drone_planner == "llm")
            run_headless(engine, args.duration, realtime=llm_used or link is not None, scripted=scripted)
        else:
            run_gui(engine, 1.0 if link is not None else args.speed, scripted)  # SITL 照真實時間跑，不能快轉
    finally:
        commander.close()
        if link is not None:
            print("連線統計：" + json.dumps(link.stats.summary(), ensure_ascii=False))
            link.close()

    out = Path(args.out) if args.out else ROOT / "output" / "demo" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    engine.ledger.finish(engine.t_ns / 1e9)
    engine.ledger.write(out)  # actions.jsonl / actions.csv：動作日誌與驗證
    if engine.multi:
        with open(out / "drone_events.jsonl", "w", encoding="utf-8") as f:  # 各機 Agent 的每一筆決策
            for dev in commander.drone_events:
                f.write(json.dumps(dev.to_dict(), ensure_ascii=False) + "\n")
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
