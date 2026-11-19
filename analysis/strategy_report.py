"""把 demo.benchmark 的 runs.csv 整理成報告用的表格與圖（捕捉策略比較、速度敏感度、操作員指令劇本）。

    python -m analysis.strategy_report --strategy <e1>/runs.csv --speed <e2>/runs.csv --operator <e3>/runs.csv \
        --out analysis/results

輸出：
    strategy_benchmark.md        三張表 + 重現指令，可直接貼進報告
    fig_capture_rate.png         各策略抓拍成功率（單一指揮官 / 多 Agent）
    fig_capture_time.png         抓拍所需時間的分佈（每個點是一個 seed）

「抓拍成功」以抓拍機（capturer）完成抓拍為準（規格 §10.10）；包圍策略沒有抓拍機角色，改用任一架完成抓拍
（time_to_capture_s）當參考，表格會標明。
"""
import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Dict, List, Optional

STRATEGY_ZH = {"evader_capture": "驅趕抓拍（自主）", "command_capture": "驅趕抓拍（操作員下指令）",
               "evader_encircle": "包圍（對照）"}
OPERATOR_ZH = {"autonomous": "自主追蹤", "command_encircle": "下指令包圍", "hold_resume": "懸停後恢復",
               "return_home": "返航"}
AGENTS_ZH = {"single": "單一指揮官", "multi": "多 Agent"}

# 參考調色盤（dataviz skill references/palette.md）的類別色前兩格；文字一律用中性色
SERIES = {"single": "#2a78d6", "multi": "#eb6834"}
TEXT, TEXT2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path) -> List[dict]:
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(v) -> Optional[float]:
    if v in (None, "", "None", "nan"):
        return None
    x = float(v)
    return None if math.isnan(x) else x


def mean_sd(vals: List[float], fmt: str = "{:.1f}") -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(statistics.stdev(vals))}"


def pct(k: int, n: int) -> str:
    return f"{k}/{n}（{k / n:.0%}）" if n else "—"


def group(rows: List[dict], *keys) -> Dict[tuple, List[dict]]:
    out: Dict[tuple, List[dict]] = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def capture_key(scenario: str) -> str:
    return "time_to_capture_s" if scenario == "evader_encircle" else "time_to_capture_by_capturer_s"


def strategy_table(rows: List[dict]) -> str:
    lines = ["| 策略 | 指揮架構 | 抓拍成功 | 抓拍時間 (s) | 中位數 (s) | 趕進捕捉區 | 視野內比例 | 最小水平距離 (m) | 最小 3D 距離 (m) | 指令一致率 | 動作到達率 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    g = group(rows, "scenario_base", "agents")
    for scn in STRATEGY_ZH:
        for ag in ("single", "multi"):
            rs = g.get((scn, ag))
            if not rs:
                continue
            t = [x for x in (num(r[capture_key(scn)]) for r in rs) if x is not None]
            zone = [num(r["zone_entered_at_s"]) for r in rs]
            hmin = [num(r["min_drone_horizontal_m"]) for r in rs]
            dmin = [num(r["min_drone_distance_m"]) for r in rs]
            zone_txt = "—（無捕捉區）" if scn == "evader_encircle" else pct(sum(z is not None for z in zone), len(rs))
            lines.append(" | ".join([
                f"| {STRATEGY_ZH[scn]}", AGENTS_ZH[ag], pct(len(t), len(rs)) + ("＊" if scn == "evader_encircle" else ""),
                mean_sd(t), f"{statistics.median(t):.1f}" if t else "—", zone_txt,
                mean_sd([num(r["visible_fraction"]) * 100 for r in rs], "{:.0f}") + "%",
                f"{min(h for h in hmin if h is not None):.1f}（平均 {statistics.mean(h for h in hmin if h is not None):.1f}）",
                f"{min(d for d in dmin if d is not None):.1f}",
                mean_sd([num(r["command_consistency"]) * 100 for r in rs if num(r["command_consistency"]) is not None], "{:.0f}") + "%",
                mean_sd([num(r["action_pass_rate"]) * 100 for r in rs if num(r["action_pass_rate"]) is not None], "{:.0f}") + "% |",
            ]))
    lines.append("")
    lines.append("＊包圍策略沒有抓拍機角色，這一欄是「任一架」完成抓拍判定（中央 20%、≥ 40 px、連續 2 s）的次數，只當參考。")
    return "\n".join(lines)


def speed_table(rows_by_speed: Dict[float, List[dict]]) -> str:
    lines = ["| 速度上限 (m/s) | 策略 | 抓拍成功 | 抓拍時間 (s) | 趕進捕捉區 | 最小水平距離 (m) |", "|---|---|---|---|---|---|"]
    for spd in sorted(rows_by_speed):
        g = group([r for r in rows_by_speed[spd] if r["agents"] == "multi"], "scenario_base")
        for scn in ("evader_capture", "evader_encircle"):
            rs = g.get((scn,))
            if not rs:
                continue
            t = [x for x in (num(r[capture_key(scn)]) for r in rs) if x is not None]
            zone = [num(r["zone_entered_at_s"]) for r in rs]
            hmin = [x for x in (num(r["min_drone_horizontal_m"]) for r in rs) if x is not None]
            lines.append(f"| {spd:g} | {STRATEGY_ZH[scn]} | {pct(len(t), len(rs))}{'＊' if scn == 'evader_encircle' else ''} | "
                         f"{mean_sd(t)} | {'—' if scn == 'evader_encircle' else pct(sum(z is not None for z in zone), len(rs))} | "
                         f"{min(hmin):.1f} |")
    lines.append("")
    lines.append("多 Agent（協調 Agent + 各機 Agent，規則式）。目標逃跑速度上限 2.2 m/s。")
    return "\n".join(lines)


def operator_table(rows: List[dict]) -> str:
    lines = ["| 劇本 | 指揮架構 | 次數 | 計畫合法率 | 指令回應 (s) | 包圍完成 (s) | 恢復後重新包圍 (s) | 暫停期間自行出計畫 | 最小水平距離 (m) |",
             "|---|---|---|---|---|---|---|---|---|"]
    g = group(rows, "scenario_base", "agents")
    for scn in OPERATOR_ZH:
        for ag in ("single", "multi"):
            rs = g.get((scn, ag))
            if not rs:
                continue
            f = lambda k: [x for x in (num(r[k]) for r in rs) if x is not None]  # noqa: E731
            enc = f("time_to_encircle_s")
            lines.append(" | ".join([
                f"| {OPERATOR_ZH[scn]}", AGENTS_ZH[ag], str(len(rs)), mean_sd([v * 100 for v in f("plan_valid_rate")], "{:.0f}") + "%",
                mean_sd(f("operator_response_s"), "{:.2f}"),
                mean_sd(enc) + (f"（{len(rs) - len(enc)} 次未完成）" if len(enc) < len(rs) else ""),
                mean_sd(f("reencircle_s")), str(int(sum(f("agent_plans_while_paused")))),
                f"{min(f('min_drone_horizontal_m')):.1f} |"]))
    lines.append("")
    lines.append("目標為不會反應的兔子路線（rabbit）；「暫停期間自行出計畫」應為 0（懸停 / 返航時 Agent 不能自己改計畫）。")
    return "\n".join(lines)


def _style(ax, plt):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def figures(rows: List[dict], out: Path) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    g = group(rows, "scenario_base", "agents")
    scns = [s for s in STRATEGY_ZH if any((s, a) in g for a in ("single", "multi"))]
    files = []

    # 圖 1：抓拍成功率（水平長條，兩個系列）
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.75 * len(scns)), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    h = 0.34
    for i, s in enumerate(reversed(scns)):
        for j, ag in enumerate(("single", "multi")):
            rs = g.get((s, ag), [])
            if not rs:
                continue
            k = sum(num(r[capture_key(s)]) is not None for r in rs)
            y = i + (h / 2 + 0.02 if j == 0 else -h / 2 - 0.02)
            ax.barh(y, k / len(rs) * 100, height=h, color=SERIES[ag], label=AGENTS_ZH[ag] if i == 0 else None)
            ax.text(k / len(rs) * 100 + 1.5, y, f"{k}/{len(rs)}", va="center", fontsize=8, color=TEXT2)
    ax.set_yticks(range(len(scns)))
    ax.set_yticklabels([STRATEGY_ZH[s] for s in reversed(scns)], color=TEXT, fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
    _style(ax, plt)
    ax.set_title("抓拍成功率（120 s 內）", loc="left", color=TEXT, fontsize=11, pad=24)
    ax.text(0, 1.03, "包圍策略沒有抓拍機，以任一架完成抓拍計", transform=ax.transAxes, fontsize=8, color=TEXT2)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_capture_rate.png", facecolor=SURFACE)
    plt.close(fig)
    files.append("fig_capture_rate.png")

    # 圖 2：抓拍時間分佈（每個點一個 seed；只有驅趕抓拍）
    cap = [s for s in scns if s != "evader_encircle"]
    fig, ax = plt.subplots(figsize=(7.2, 0.9 + 0.75 * len(cap)), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    for i, s in enumerate(reversed(cap)):
        for j, ag in enumerate(("single", "multi")):
            t = [x for x in (num(r[capture_key(s)]) for r in g.get((s, ag), [])) if x is not None]
            if not t:
                continue
            y0 = i + (0.17 if j == 0 else -0.17)
            jit = [((k * 37) % 11 - 5) / 5 * 0.07 for k in range(len(t))]  # 固定的小抖動，避免重疊
            ax.scatter(t, [y0 + d for d in jit], s=14, color=SERIES[ag], edgecolors=SURFACE, linewidths=0.6,
                       label=AGENTS_ZH[ag] if i == 0 else None, zorder=3)
            med = statistics.median(t)
            ax.plot([med, med], [y0 - 0.13, y0 + 0.13], color=TEXT, linewidth=1.6, zorder=4)
    ax.set_yticks(range(len(cap)))
    ax.set_yticklabels([STRATEGY_ZH[s] for s in reversed(cap)], color=TEXT, fontsize=9)
    ax.set_xlabel("抓拍機完成抓拍的時間 (s)", color=TEXT2, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    _style(ax, plt)
    ax.set_title("抓拍所需時間（每點 = 一個 seed，黑線 = 中位數）", loc="left", color=TEXT, fontsize=11, pad=12)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig_capture_time.png", facecolor=SURFACE)
    plt.close(fig)
    files.append("fig_capture_time.png")
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description="benchmark 結果 → 報告用表格與圖")
    ap.add_argument("--strategy", required=True, help="策略比較的 runs.csv（evader_capture / evader_encircle / command_capture）")
    ap.add_argument("--speed", nargs="*", default=[], help="其他速度上限的 runs.csv（速度敏感度）")
    ap.add_argument("--operator", default=None, help="操作員指令劇本的 runs.csv（rabbit）")
    ap.add_argument("--out", default="analysis/results")
    ap.add_argument("--commit", default="", help="寫進報告的程式版本（git commit）")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    strat = load(args.strategy)
    by_speed: Dict[float, List[dict]] = {}
    for r in strat + [r for p in args.speed for r in load(p)]:
        if r.get("drone_speed_mps") not in (None, ""):
            by_speed.setdefault(float(r["drone_speed_mps"]), []).append(r)
    n_seeds = len({r["seed"] for r in strat})
    spd = sorted({float(r["drone_speed_mps"]) for r in strat if r.get("drone_speed_mps")})
    md = [
        "# 捕捉策略實驗結果（規則式 Agent）", "",
        f"程式版本 `{args.commit}`；每個組合 {n_seeds} 個 seed（目標逃跑模型、起始位置、雜訊都由 seed 決定）。"
        f"策略比較的無人機速度上限 {', '.join(f'{s:g}' for s in spd)} m/s（對齊 A 的 agent_server --max-speed），"
        "目標是會逃的模擬目標（離最近無人機越近逃得越快，上限 2.2 m/s，半徑 60 m 圓形場地）。這是我們設計的簡化模型，不代表真實車輛或行人。", "",
        "## 表 1：捕捉策略比較", "", strategy_table(strat), "",
    ]
    if len(by_speed) > 1:
        md += ["## 表 2：速度上限敏感度", "", speed_table(by_speed), ""]
    if args.operator:
        md += ["## 表 3：操作員指令劇本（指令理解與安全行為）", "", operator_table(load(args.operator)), ""]
    figs = figures(strat, out)
    md += ["## 圖", ""] + [f"![{f}]({f})" for f in figs] + ["",
           "## 名詞", "",
           "- **抓拍成功**：抓拍機畫面中，目標中心落在中央 20% × 20%、像素尺寸 ≥ 40 px，連續 2 s（規格 §10.10），用目標真值判定。",
           "- **趕進捕捉區**：目標進入捕捉區（半徑 10 m）的比例；捕捉區由規劃器決定，同一次捕捉期間固定。",
           "- **指令一致率**：無人機速度方向與「指向目標點」夾角 < 45°（或已到達）的時間比例，衡量飛控是否照指令飛。",
           "- **動作到達率**：動作日誌中在期限內到達目標點的比例（規格 §10.11）；目標點跟著逃跑目標移動，所以低於指令一致率。",
           "- **單一指揮官 / 多 Agent**：單一指揮官直接分派三架的角色；多 Agent 是協調 Agent 分派角色、每架各機 Agent 從角色白名單選動作。",
           ""]
    (out / "strategy_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"輸出：{out}")


if __name__ == "__main__":
    main()
