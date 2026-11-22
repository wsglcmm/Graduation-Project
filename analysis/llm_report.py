"""LLM 版本 vs 規則式的對照表（同劇本、同 seed），以及 LLM 輸出品質（合法率、錯誤種類、延遲）。

    python -m analysis.llm_report --llm <benchmark 輸出資料夾> [...] --rule <規則式 runs.csv> [...] --out analysis/results

--llm 的資料夾是 demo.benchmark --planner llm 的輸出（裡面有 runs.csv 與 runs/<劇本>/events.jsonl）。
規則式只取和 LLM 相同的 seed 比較。錯誤分成「模型輸出不合法」與「API 失敗（429 額度、503 / 504）」：
後者是免費額度或伺服器問題，不是模型理解錯，報告要分開寫。
"""
import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from analysis.strategy_report import AGENTS_ZH, load, mean_sd, num, pct

SCN_ZH = {"switch_capture": "中途改口抓拍", "command_encircle": "下令包圍", "hold_resume": "懸停後恢復", "return_home": "返航"}


def classify(err: Optional[str]) -> str:
    if not err:
        return "合法"
    if "429" in err:
        return "API：429 額度用完"
    if any(k in err for k in ("503", "504", "UNAVAILABLE", "DEADLINE", "逾時", "Timeout")):
        return "API：503 / 504 伺服器忙或逾時"
    if "target_id" in err and "不存在" in err:
        return "模型：指定了不存在的目標"
    if "capturer" in err or "herder" in err:
        return "模型：捕捉角色數量不對"
    if "格式" in err or "Extra inputs" in err or "Field required" in err or "JSON" in err:
        return "模型：JSON 格式不符"
    return "模型：其他 — " + err[:30]


def scan_events(dirs: List[Path], name: str) -> collections.Counter:
    """name：events.jsonl（協調 Agent）或 drone_events.jsonl（各機 Agent）。只統計真的送給 LLM 的請求。"""
    c: collections.Counter = collections.Counter()
    for d in dirs:
        for f in sorted((d / "runs").glob(f"*/{name}")):
            for line in f.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                if e.get("planner") != "llm":
                    continue
                c[classify(None if e.get("valid") else e.get("error"))] += 1
    return c


def latencies(dirs: List[Path], name: str) -> List[float]:
    out = []
    for d in dirs:
        for f in sorted((d / "runs").glob(f"*/{name}")):
            for line in f.read_text(encoding="utf-8").splitlines():
                e = json.loads(line)
                if e.get("planner") == "llm" and e.get("valid") and e.get("latency_ms"):
                    out.append(float(e["latency_ms"]))
    return out


def quality_table(title: str, c: collections.Counter, lat: List[float]) -> List[str]:
    tot = sum(c.values())
    api = sum(v for k, v in c.items() if k.startswith("API"))
    model_bad = sum(v for k, v in c.items() if k.startswith("模型"))
    answered = tot - api
    lines = [f"**{title}**：送出 {tot} 次請求", "",
             "| 項目 | 次數 | 比例 |", "|---|---|---|",
             f"| 合法並通過白名單驗證 | {c['合法']} | {c['合法'] / tot:.1%}（佔全部請求） |",
             f"| 模型有回應時的合法率 | {c['合法']}/{answered} | **{c['合法'] / answered:.1%}** |" if answered else "| 模型有回應時的合法率 | — | — |"]
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        if k != "合法":
            lines.append(f"| {k} | {v} | {v / tot:.1%} |")
    if lat:
        q = sorted(lat)
        lines.append(f"| 延遲（合法回應） | 平均 {statistics.mean(q):.0f} ms | 中位數 {statistics.median(q):.0f} ms、p95 {q[int(0.95 * (len(q) - 1))]:.0f} ms |")
    lines.append("")
    lines.append(f"不合法的 {model_bad + api} 次都沒有送出：協調 Agent 沿用上一個合法計畫、或自動重送，重送仍失敗才改用規則式；"
                 "各機 Agent 等待或失敗時先用規則式。所以 LLM 出錯不會讓無人機失控。")
    return lines


def compare_table(llm_rows: List[dict], rule_rows: List[dict]) -> List[str]:
    seeds = {r["seed"] for r in llm_rows}
    rule_rows = [r for r in rule_rows if r["seed"] in seeds]
    lines = ["| 劇本 | 指揮架構 | 決策 | 次數 | 下令後抓拍 (s) | 包圍完成 (s) | 指令回應 (s) | 暫停期間自行出計畫 | 最小 3D 距離 (m) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for scn in SCN_ZH:
        for ag in ("single", "multi"):
            for planner, rows in (("規則式", rule_rows), ("LLM", llm_rows)):
                rs = [r for r in rows if r["scenario_base"] == scn and r["agents"] == ag
                      and (planner == "規則式" or r.get("drone_planner") in (None, "", "rule"))]
                if not rs:
                    continue
                f = lambda k: [x for x in (num(r.get(k)) for r in rs) if x is not None]  # noqa: E731
                cap = f("capture_after_command_s")
                enc = f("time_to_encircle_s")
                lines.append(" | ".join([
                    f"| {SCN_ZH[scn]}", AGENTS_ZH[ag], planner, str(len(rs)),
                    (mean_sd(cap) + (f"（{len(rs) - len(cap)} 次未抓到）" if len(cap) < len(rs) else "")) if scn == "switch_capture" else "—",
                    mean_sd(enc) + (f"（{len(rs) - len(enc)} 次未完成）" if enc and len(enc) < len(rs) else "") if enc else "—",
                    mean_sd(f("operator_response_s"), "{:.2f}"), str(int(sum(f("agent_plans_while_paused")))),
                    f"{min(f('min_drone_distance_m')):.1f} |"]))
    # 各機 Agent 也用 LLM 的那一組
    rs = [r for r in llm_rows if r.get("drone_planner") == "llm"]
    if rs:
        cap = [x for x in (num(r.get("capture_after_command_s")) for r in rs) if x is not None]
        lines.append(f"| 中途改口抓拍 | 多 Agent | LLM（各機 Agent 也用 LLM） | {len(rs)} | {mean_sd(cap)} | — | "
                     f"{mean_sd([x for x in (num(r.get('operator_response_s')) for r in rs) if x is not None], '{:.2f}')} | "
                     f"{int(sum(num(r.get('agent_plans_while_paused')) or 0 for r in rs))} | "
                     f"{min(num(r['min_drone_distance_m']) for r in rs):.1f} |")
    lines += ["", f"規則式只取和 LLM 相同的 seed（{', '.join(sorted(seeds))}）。「暫停期間自行出計畫」應為 0。"]
    return lines


ROLE_ZH = {"herder": "驅", "capturer": "拍", "blocker": "封", "overwatch": "高"}


def capture_roles(d: Path, run: str) -> List[str]:
    """一次執行中，capture 計畫（有套用的）出現過的角色分派，例如「驅高拍」= uav1 驅趕、uav2 高空、uav3 抓拍。"""
    seq = []
    f = d / "runs" / run / "events.jsonl"
    for line in f.read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        p = e.get("plan") or {}
        if p.get("mode") == "capture" and e.get("plan_id") is not None:
            r = "".join(ROLE_ZH.get(a["role"], "?") for a in sorted(p["assignments"], key=lambda a: a["drone_id"]))
            if not seq or seq[-1] != r:
                seq.append(r)
    return seq


def improvement_table(versions: List[tuple]) -> List[str]:
    """versions：[(標籤, [資料夾…]), …]。同一個版本、同一種架構一列；另列 seed 0–2 的平均，和改善前同 seed 比較。"""
    lines = ["| 版本 | 指揮架構 | 下令後抓拍 (s) | 抓到 | seed 0–2 平均 (s) | 抓拍期間的角色分派（uav1 uav2 uav3） | 角色鎖定擋下 | API 失敗 |",
             "|---|---|---|---|---|---|---|---|"]
    for label, dirs in versions:
        for ag in ("single", "multi"):
            rows = [(d, r) for d in dirs for r in load(d / "runs.csv")
                    if r["scenario_base"] == "switch_capture" and r["agents"] == ag and r.get("drone_planner") in (None, "", "rule")]
            if not rows:
                continue
            rows.sort(key=lambda x: int(x[1]["seed"]))
            cap = [x for x in (num(r.get("capture_after_command_s")) for _, r in rows) if x is not None]
            first3 = [x for x in (num(r.get("capture_after_command_s")) for _, r in rows if int(r["seed"]) <= 2) if x is not None]
            runs = [(d, f"switch_capture{'_multi' if ag == 'multi' else ''}_s{r['seed']}") for d, r in rows]
            roles = [capture_roles(d, run) for d, run in runs]
            api = locks = 0
            for d, run in runs:
                for line in (d / "runs" / run / "events.jsonl").read_text(encoding="utf-8").splitlines():
                    e = json.loads(line)
                    locks += "角色鎖定" in (e.get("note") or "")
                    if e.get("planner") == "llm" and not e.get("valid") and classify(e.get("error")).startswith("API"):
                        api += 1
            n3 = sum(int(r["seed"]) <= 2 for _, r in rows)
            lines.append(f"| {label} | {AGENTS_ZH[ag]} | {mean_sd(cap)} | {pct(len(cap), len(rows))} | "
                         f"{mean_sd(first3)}{'' if len(first3) == n3 else f'（{n3 - len(first3)} 次未抓到）'} | "
                         f"{'；'.join('→'.join(x) if x else '（沒進入抓拍）' for x in roles)} | {locks} | {api} |")
    lines += ["", "角色代號：驅 = 驅趕、拍 = 抓拍、封 = 封鎖、高 = 高空監看，依序是 uav1、uav2、uav3。"
              "規則式同劇本 seed 0–2：下令後 17.4 s（3/3）。"]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description="LLM 版本 vs 規則式對照與 LLM 輸出品質")
    ap.add_argument("--llm", nargs="+", required=True, help="demo.benchmark --planner llm 的輸出資料夾")
    ap.add_argument("--rule", nargs="+", required=True, help="規則式的 runs.csv（同劇本）")
    ap.add_argument("--model", default="gemini-3.1-flash-lite（失敗改用 gemini-3.5-flash-lite）")
    ap.add_argument("--commit", default="")
    ap.add_argument("--out", default="analysis/results")
    ap.add_argument("--notes", default="", help="附加在最後的觀察（Markdown）")
    ap.add_argument("--after", nargs="*", default=[],
                    help="改善後重跑的資料夾，可寫成「標籤=資料夾」（和 --llm 的 switch_capture 比較）")
    args = ap.parse_args(argv)
    dirs = [Path(d) for d in args.llm]
    llm_rows = [r for d in dirs for r in load(d / "runs.csv")]
    rule_rows = [r for p in args.rule for r in load(p)]
    coord_dirs = [d for d in dirs if any(r.get("drone_planner") != "llm" for r in load(d / "runs.csv"))]
    drone_dirs = [d for d in dirs if any(r.get("drone_planner") == "llm" for r in load(d / "runs.csv"))]
    md = ["# LLM 版本實測（協調 Agent / 各機 Agent）", "",
          f"模型 {args.model}；程式版本 `{args.commit}`；每組 3 個 seed，照真實時間執行。"
          "無人機速度上限 5 m/s（策略場景）。協調 Agent 每 5 s 問一次 LLM，操作員下指令時立刻問。", "",
          "## 表 A：LLM vs 規則式（同劇本、同 seed）", ""] + compare_table(llm_rows, rule_rows) + ["",
          "## 表 B：LLM 輸出品質", ""]
    md += quality_table("協調 Agent（第四步：各機 Agent 用規則式）", scan_events(coord_dirs, "events.jsonl"),
                        latencies(coord_dirs, "events.jsonl")) + [""]
    if drone_dirs:
        md += quality_table("協調 Agent（第五步：同時有 3 架各機 Agent 在問 LLM）", scan_events(drone_dirs, "events.jsonl"),
                            latencies(drone_dirs, "events.jsonl")) + [""]
        md += quality_table("各機 Agent（第五步，每 10 s 一次 × 3 架）", scan_events(drone_dirs, "drone_events.jsonl"),
                            latencies(drone_dirs, "drone_events.jsonl")) + [""]
    if args.after:
        versions = [("改善前 d40be5c", dirs)]
        for item in args.after:
            label, _, path = item.rpartition("=")
            versions.append((label or Path(path).name, [Path(path)]))
        md += ["## 表 C：改善驗證（中途改口抓拍）", "",
               "改善內容：`a578198` 把捕捉區與各機相對位置放進給 LLM 的世界狀態；`ea159d0` LLM 先搜尋時保留操作員的抓拍意圖；"
               "`22e9476` 同一次捕捉期間鎖定角色。", ""]
        md += improvement_table(versions) + [""]
    if args.notes:
        md += ["## 觀察", "", Path(args.notes).read_text(encoding="utf-8")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "llm_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
