"""固定劇本批次比較的測試：操作員相關指標的計算，以及 CLI 輸出。"""
import json

from demo.benchmark import main, operator_metrics


def ev(t, trigger, mode, cmd=None, valid=True, applied=True, latency=10.0):
    return {"timestamp_ns": int(t * 1e9), "plan_id": "p" if (valid and applied) else None, "trigger": trigger,
            "planner": "llm", "latency_ms": latency, "valid": valid, "plan": {"mode": mode} if valid else None,
            "error": None, "operator_command": cmd}


def test_counts_agent_plans_while_paused_and_response_time():
    events = [
        ev(1, "periodic", "search"),
        ev(11.5, "operator", "encircle", "包圍"),
        ev(20.8, "operator", "hold", "先懸停"),
        ev(25, "periodic", "encircle"),            # 懸停期間 Agent 自己出計畫 → 違規
        ev(26, "periodic", "encircle", applied=False),  # 被取代、沒有套用 → 不算
        ev(31, "operator", "encircle", "繼續包圍"),
        ev(36, "periodic", "encircle"),            # 恢復後正常
        ev(40, "periodic", None, valid=False),
    ]
    m = operator_metrics(events, [(10, "包圍"), (20, "先懸停"), (30, "繼續包圍"), (50, "沒有回應的指令")])
    assert m["agent_plans_while_paused"] == 1
    assert abs(m["operator_response_s"] - 1.1) < 1e-9 and m["operator_commands_answered"] == "3/4"
    assert m["plan_valid_rate"] == 6 / 7  # 7 筆計入（被取代的不算），1 筆不合法


def test_benchmark_cli_rule_planner(tmp_path):
    results = main(["--scenario", "hold_resume", "return_home", "--seeds", "0", "--out", str(tmp_path)])
    assert [r["scenario"] for r in results] == ["hold_resume", "return_home"]
    for r in results:
        assert r["agent_plans_while_paused"] == 0 and r["plan_valid_rate"] == 1.0
        assert r["operator_commands_answered"] in ("3/3", "2/2")
    assert results[0]["reencircle_s"] is not None
    assert (tmp_path / "runs.csv").exists() and "| hold_resume | 1 |" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    m = json.loads((tmp_path / "runs" / "return_home_s0" / "metrics.json").read_text(encoding="utf-8"))
    assert m["scenario"] == "return_home"
