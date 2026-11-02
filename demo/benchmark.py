"""批次執行固定劇本，輸出報告用的比較表（規則式 vs LLM）。

    python -m demo.benchmark                                  # 規則式，全部劇本，seed 0 1 2（快轉，幾秒鐘）
    python -m demo.benchmark --planner llm --period 5         # LLM，照真實時間跑（每個劇本 × seed 約 1.5 分鐘）
    python -m demo.benchmark --scenario hold_resume --seeds 0 1 2 3 4

輸出（output/benchmark/<時間>_<planner>/）：
    runs.csv        每次執行一列（劇本、seed、全部指標）
    summary.md      依劇本彙整的平均 ± 標準差，可直接貼進報告
    runs/<劇本>_s<seed>/metrics.json、events.jsonl

除了 engine.summary() 的指標，另外計算：
    plan_valid_rate / plan_latency_ms_mean   不分規劃者（LLM 或規則式）的計畫合法率與延遲
    agent_plans_while_paused   懸停 / 返航期間 Agent 自己產生的計畫數（應為 0）
    operator_response_s        從下指令到對應計畫生效的時間（平均）
    reencircle_s               劇本指定的時間點起，多久重新完成包圍（例如懸停後恢復）
"""
import argparse
import csv
import json
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

from coord_transform.config import load_camera_config, load_error_model
from demo.engine import DemoEngine
from demo.run import ROOT, feed_scripted, make_commander
from demo.scenarios import SCENARIOS, Scenario
from sim2d import RabbitConfig

LOCKING = ("hold", "return_home")
REPORT_KEYS = (  # summary.md 的欄位順序（key, 標題, 格式）
    ("time_to_encircle_s", "包圍完成 (s)", "{:.1f}"),
    ("encircled_fraction", "包圍時間比例", "{:.0%}"),
    ("visible_fraction", "視野內比例", "{:.0%}"),
    ("estimate_rmse_m", "估計誤差 RMSE (m)", "{:.2f}"),
    ("min_drone_distance_m", "最小機間距離 (m)", "{:.1f}"),
    ("plan_valid_rate", "計畫合法率", "{:.0%}"),
    ("plan_latency_ms_mean", "延遲平均 (ms)", "{:.0f}"),
    ("operator_response_s", "指令回應 (s)", "{:.2f}"),
    ("reencircle_s", "恢復後重新包圍 (s)", "{:.1f}"),
    ("agent_plans_while_paused", "暫停期間自行出計畫", "{:.0f}"),
)


def operator_metrics(events, commands) -> dict:
    """從 AgentEvent 算出操作員相關指標。events 為 AgentEvent 物件或 dict。"""
    evs = [e if isinstance(e, dict) else e.to_dict() for e in events]
    paused = False
    while_paused = 0
    for e in sorted(evs, key=lambda e: e["timestamp_ns"]):
        applied = e["valid"] and e["plan_id"] is not None
        if e["trigger"] == "operator" and applied:
            paused = e["plan"]["mode"] in LOCKING
        elif paused and applied:
            while_paused += 1
    responses = []
    for t_cmd, text in commands:
        ok = [e for e in evs if e["trigger"] == "operator" and e["valid"] and e["plan_id"] is not None
              and e["operator_command"] == text and e["timestamp_ns"] / 1e9 >= t_cmd]
        if ok:
            responses.append(min(e["timestamp_ns"] for e in ok) / 1e9 - t_cmd)
    planned = [e for e in evs if e["plan_id"] is not None or not e["valid"]]  # 被取代（valid 但未套用）的不算
    return {"plan_valid_rate": (sum(e["valid"] for e in planned) / len(planned)) if planned else None,
            "plan_latency_ms_mean": statistics.mean(e["latency_ms"] for e in planned) if planned else None,
            "agent_plans_while_paused": while_paused,
            "operator_response_s": statistics.mean(responses) if responses else None,
            "operator_commands_answered": f"{len(responses)}/{len(commands)}"}


def run_once(scn: Scenario, seed: int, planner: str, period_s: float, cam, err, out_dir: Optional[Path] = None) -> dict:
    commander = make_commander(planner, period_s)
    engine = DemoEngine(cam, err, commander, rabbit=RabbitConfig(seed=seed), seed=seed)
    scripted = sorted(scn.commands)
    realtime = planner == "llm"  # LLM 要照真實時間跑，延遲才有意義
    try:
        for _ in range(int(round(scn.duration_s / engine.dt))):
            t0 = time.monotonic()
            feed_scripted(engine, scripted)
            engine.step()
            if realtime:
                time.sleep(max(0.0, engine.dt - (time.monotonic() - t0)))
        commander.wait_idle()
    finally:
        commander.close()
    m = engine.summary()
    m.update(operator_metrics(commander.events, scn.commands))
    if scn.reencircle_from_s is not None:
        later = [t for t in m["encircle_onsets_s"] if t >= scn.reencircle_from_s]
        m["reencircle_s"] = later[0] - scn.reencircle_from_s if later else None
    m.update({"scenario": scn.name, "seed": seed, "planner": planner})
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
        with open(out_dir / "events.jsonl", "w", encoding="utf-8") as f:
            for ev in commander.events:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")
    return m


def _cell(values: List[Optional[float]], fmt: str) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return "—"
    miss = f"（{len(values) - len(vals)} 次未達成）" if len(vals) < len(values) else ""
    if len(vals) == 1:
        return fmt.format(vals[0]) + miss
    return f"{fmt.format(statistics.mean(vals))} ± {fmt.format(statistics.stdev(vals))}" + miss


def summarize(results: List[dict], planner: str) -> str:
    lines = [f"# Benchmark：{'LLM' if planner == 'llm' else '規則式'}", "",
             "數值為各 seed 的平均 ± 標準差；「未達成」表示該次執行沒有完成包圍。", ""]
    lines.append("| 劇本 | 次數 | " + " | ".join(t for _, t, _ in REPORT_KEYS) + " |")
    lines.append("|---" * (len(REPORT_KEYS) + 2) + "|")
    for name in dict.fromkeys(r["scenario"] for r in results):
        rs = [r for r in results if r["scenario"] == name]
        cells = [_cell([r.get(k) for r in rs], fmt) for k, _, fmt in REPORT_KEYS]
        lines.append(f"| {name} | {len(rs)} | " + " | ".join(cells) + " |")
    lines += ["", "劇本說明：", ""] + [f"- **{s.name}**：{s.description}" for s in SCENARIOS.values() if any(r["scenario"] == s.name for r in results)]
    return "\n".join(lines) + "\n"


def _flat(m: dict) -> Dict[str, object]:
    out = {}
    for k, v in m.items():
        if isinstance(v, dict):
            out.update({f"{k}.{kk}": vv for kk, vv in v.items()})
        else:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="固定劇本批次比較（規則式 vs LLM）")
    ap.add_argument("--planner", choices=("rule", "llm"), default="rule")
    ap.add_argument("--scenario", nargs="+", default=["all"], help=f"劇本名稱，或 all：{', '.join(SCENARIOS)}")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--period", type=float, default=2.0, help="多久問一次 Agent（秒）；LLM 建議 5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    names = list(SCENARIOS) if args.scenario == ["all"] else args.scenario
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"沒有這些劇本：{unknown}（可用：{', '.join(SCENARIOS)}）")
    cam = load_camera_config(ROOT / "config" / "camera.yaml")
    err = load_error_model(ROOT / "config" / "error_model.yaml")
    out = Path(args.out) if args.out else ROOT / "output" / "benchmark" / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.planner}"

    results = []
    total = len(names) * len(args.seeds)
    for name in names:
        for seed in args.seeds:
            print(f"[{len(results) + 1}/{total}] {name} seed={seed} ...", flush=True)
            results.append(run_once(SCENARIOS[name], seed, args.planner, args.period, cam, err, out / "runs" / f"{name}_s{seed}"))

    out.mkdir(parents=True, exist_ok=True)
    rows = [_flat(r) for r in results]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "runs.csv", "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig：Excel 開啟中文不亂碼
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    text = summarize(results, args.planner)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print("\n" + text + f"\n輸出位置：{out}")
    return results


if __name__ == "__main__":
    main()
