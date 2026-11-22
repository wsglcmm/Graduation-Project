"""把 LLM 的文字輸出解析並驗證成動作。失敗時回傳可直接餵回 LLM 的錯誤訊息（供重試）。"""
import json
from dataclasses import dataclass
from typing import Optional

from pydantic import ValidationError

from .actions import ACTION_ADAPTER


@dataclass(frozen=True)
class ParseResult:
    ok: bool
    action: Optional[object] = None  # 驗證通過的動作模型
    error: Optional[str] = None  # ok=False 時的原因（可餵回 LLM）


def _format_errors(e: ValidationError) -> str:
    return "；".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors())


def parse_llm_output(text: str) -> ParseResult:
    """從 LLM 輸出取出「唯一一個」JSON 物件並驗證。

    容忍前後的說明文字與 ```json 圍欄，但文字中出現第二個 JSON 物件一律拒絕
    （一次只能下一個動作）。任何不合法的輸出都不會轉給飛控。
    """
    if not isinstance(text, str) or not text.strip():
        return ParseResult(False, error="輸出為空。請只輸出一個 JSON 物件。")

    start = text.find("{")
    if start < 0:
        return ParseResult(False, error="找不到 JSON 物件。請只輸出一個 JSON 物件。")

    try:
        obj, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as e:
        return ParseResult(False, error=f"JSON 格式錯誤：{e.msg}（位置 {e.pos}）。")

    if "{" in text[end:]:
        return ParseResult(False, error="輸出包含多個 JSON 物件。一次只能輸出一個動作。")
    if not isinstance(obj, dict):
        return ParseResult(False, error="輸出必須是 JSON 物件（不是陣列或其他型別）。")

    try:
        action = ACTION_ADAPTER.validate_python(obj)
    except ValidationError as e:
        return ParseResult(False, error=f"不符合動作格式：{_format_errors(e)}")
    return ParseResult(True, action=action)
