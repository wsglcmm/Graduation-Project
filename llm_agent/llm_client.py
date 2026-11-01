"""LLM 呼叫層：把「哪家模型、怎麼呼叫」隔離在這裡，Agent 其他程式只看 complete(system, messages) -> str。

預設用 Anthropic 的 Claude Haiku 5.5（官方標示延遲最低，適合 0.5–2 Hz 的任務層決策）。
金鑰只從環境變數 ANTHROPIC_API_KEY 讀取；可以放在專案根目錄的 .env（已在 .gitignore，絕對不 commit）。

可用環境變數：
    ANTHROPIC_API_KEY   必填（使用 LLM 模式時）
    AGENT_MODEL         預設 claude-haiku-5-5
    AGENT_EFFORT        預設 low（降低延遲）；若 API 不接受，設成空字串即可不送這個參數
    AGENT_TIMEOUT_S     預設 10
"""
import os
from pathlib import Path
from typing import List, Optional, Protocol

DEFAULT_MODEL = "claude-haiku-5-5"


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, messages: List[dict]) -> str: ...


def load_dotenv(path=".env") -> None:
    """極簡 .env 讀取：KEY=VALUE 一行一個；不覆蓋已存在的環境變數。不另外安裝 python-dotenv。"""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class AnthropicClient:
    def __init__(self, model: Optional[str] = None, max_tokens: int = 1024, timeout_s: Optional[float] = None,
                 effort: Optional[str] = None, sdk_client=None):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("未安裝 anthropic 套件：python -m pip install anthropic") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("找不到 ANTHROPIC_API_KEY：請放在專案根目錄的 .env（不要 commit）")
        self.model = model or os.environ.get("AGENT_MODEL", DEFAULT_MODEL)
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s if timeout_s is not None else float(os.environ.get("AGENT_TIMEOUT_S", "10"))
        self.effort = effort if effort is not None else os.environ.get("AGENT_EFFORT", "low")
        self._client = sdk_client or anthropic.Anthropic()  # sdk_client 供測試注入
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, messages: List[dict]) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=messages, timeout=self.timeout_s)
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        resp = self._client.messages.create(**kwargs)
        # content 是 block 列表，可能先有 thinking block；只取 text block
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


class ScriptedClient:
    """測試與離線展示用：依序回傳預先寫好的輸出（字串或例外）。"""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: List[List[dict]] = []
        self.name = "scripted"

    def complete(self, system: str, messages: List[dict]) -> str:
        self.calls.append(messages)
        out = self.outputs.pop(0) if self.outputs else '{"mode": "hold"}'
        if isinstance(out, Exception):
            raise out
        return out
