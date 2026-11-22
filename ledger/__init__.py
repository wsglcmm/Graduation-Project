"""動作日誌與驗證（C）：Agent 動作 → 白名單驗證 → 飛控指令 → A 的回覆 → 實際是否達成。"""
from .action_ledger import RESULT_ZH, RESULTS, ActionLedger, ActionRecord, VerifyConfig

__all__ = ["RESULT_ZH", "RESULTS", "ActionLedger", "ActionRecord", "VerifyConfig"]
