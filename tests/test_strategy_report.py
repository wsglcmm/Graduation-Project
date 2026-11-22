"""報告用表格：成功率、平均 ± 標準差、包圍策略改用任一架抓拍、未達成不計入時間。"""
import csv

from analysis.strategy_report import main, mean_sd, pct

COLS = ["scenario_base", "agents", "seed", "drone_speed_mps", "time_to_capture_by_capturer_s", "time_to_capture_s",
        "zone_entered_at_s", "visible_fraction", "min_drone_horizontal_m", "min_drone_distance_m",
        "command_consistency", "action_pass_rate"]


def write(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def row(scn, ag, seed, cap=None, any_cap=None, zone=None):
    return {"scenario_base": scn, "agents": ag, "seed": seed, "drone_speed_mps": 5.0,
            "time_to_capture_by_capturer_s": "" if cap is None else cap, "time_to_capture_s": "" if any_cap is None else any_cap,
            "zone_entered_at_s": "" if zone is None else zone, "visible_fraction": 0.9, "min_drone_horizontal_m": 6.0 + seed,
            "min_drone_distance_m": 8.0, "command_consistency": 0.9, "action_pass_rate": 0.5}


def test_helpers():
    assert pct(3, 4) == "3/4（75%）"
    assert mean_sd([30.0, 40.0]) == "35.0 ± 7.1"
    assert mean_sd([]) == "—"


def test_strategy_table_counts_failures_and_encircle_reference(tmp_path):
    rows = [row("evader_capture", "multi", 0, cap=30, zone=20), row("evader_capture", "multi", 1, cap=40, zone=25),
            row("evader_capture", "multi", 2), row("evader_encircle", "multi", 0, any_cap=50),
            row("evader_encircle", "multi", 1)]
    write(tmp_path / "runs.csv", rows)
    main(["--strategy", str(tmp_path / "runs.csv"), "--out", str(tmp_path / "out")])
    md = (tmp_path / "out" / "strategy_benchmark.md").read_text(encoding="utf-8")
    cap_line = next(l for l in md.splitlines() if l.startswith("| 驅趕抓拍（自主）"))
    assert "2/3（67%）" in cap_line and "35.0 ± 7.1" in cap_line and "6.0（平均 7.0）" in cap_line
    enc_line = next(l for l in md.splitlines() if l.startswith("| 包圍（對照）"))
    assert "1/2（50%）＊" in enc_line and "無捕捉區" in enc_line
    assert (tmp_path / "out" / "fig_capture_rate.png").stat().st_size > 1000


def test_llm_error_classification():
    from analysis.llm_report import classify

    assert classify(None) == "合法"
    assert classify("ClientError: 429 RESOURCE_EXHAUSTED") == "API：429 額度用完"
    assert classify("ServerError: 503 UNAVAILABLE") == "API：503 / 504 伺服器忙或逾時"
    assert classify("target_id=1 不存在，目前的目標：[]") == "模型：指定了不存在的目標"
    assert classify("不符合動作格式：data: Extra inputs are not permitted") == "模型：JSON 格式不符"
