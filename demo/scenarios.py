"""報告用的固定測試劇本：同一個劇本、同一組 seed，規則式與 LLM 才能公平比較。

每個劇本 = 時長 + 預排的操作員指令（秒數, 指令）。指令用中文，規則式指揮官的關鍵字也看得懂。
新增劇本時只要加進 SCENARIOS；名稱會出現在 benchmark 的輸出表格。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    duration_s: float
    commands: List[Tuple[float, str]] = field(default_factory=list)
    reencircle_from_s: Optional[float] = None  # 量「從這個時間起，多久重新完成包圍」（例如恢復指令的時間）
    target: str = "rabbit"  # rabbit：不理無人機的兔子；evader：會逃的目標（策略場景，無人機速度用 A 的 3 m/s）
    strategy: str = "encircle"  # 規則式沒有操作員指令時的預設策略
    capture_from_s: Optional[float] = None  # 量「從這個時間起，多久完成抓拍」（例如中途改下捕捉指令的時間）


SCENARIOS: Dict[str, Scenario] = {
    s.name: s
    for s in (
        Scenario("autonomous", "沒有操作員指令，Agent 自己搜尋、追蹤、包圍", 90.0),
        Scenario("command_encircle", "第 10 秒下令包圍", 90.0, [(10.0, "三架一起把那台車包圍起來")]),
        Scenario(
            "hold_resume",
            "包圍 → 懸停 → 恢復包圍：檢查懸停期間 Agent 不會自己出計畫，以及恢復後多快重新包圍",
            100.0,
            [(15.0, "三架一起把那台車包圍起來"), (40.0, "先懸停"), (60.0, "繼續包圍那台車")],
            reencircle_from_s=60.0,
        ),
        Scenario("return_home", "包圍 → 返航：返航後 Agent 不得再改變計畫", 75.0, [(15.0, "三架一起把那台車包圍起來"), (50.0, "請返航")]),
        # 策略場景：同一個會逃的目標，比較不同捕捉策略（報告「策略比較」）
        Scenario("evader_encircle", "會逃的目標，預設策略包圍（對照組）", 120.0, target="evader", strategy="encircle"),
        Scenario("evader_capture", "會逃的目標，預設策略驅趕 → 抓拍", 120.0, target="evader", strategy="capture"),
        Scenario("command_capture", "會逃的目標，第 10 秒下令「把那台車趕進捕捉區抓拍」（測 Agent 能否把自然語言轉成捕捉策略）",
                 120.0, [(10.0, "把那台車趕進捕捉區抓拍")], target="evader"),
        # command_capture 的指令在發現目標（約 18 s）之前，結果會和 evader_capture 相同；
        # 這個劇本先包圍、第 35 秒才改口，測「中途換策略」：Agent 要重新分派角色，量下令後多久抓拍
        Scenario("switch_capture", "會逃的目標，先包圍，第 35 秒改口「改成把那台車趕進捕捉區抓拍」（測中途換策略）",
                 150.0, [(35.0, "改成把那台車趕進捕捉區抓拍")], target="evader", capture_from_s=35.0),
    )
}
