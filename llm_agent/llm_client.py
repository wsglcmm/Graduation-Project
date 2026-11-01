"""LLM 呼叫層：把「哪家模型、怎麼呼叫」隔離在這裡，Agent 其他程式只看 complete(system, messages) -> str。

支援兩家（用 AGENT_PROVIDER 切換，預設 gemini，依與老師的約定）：
    gemini      Google Gemini API（google-genai 套件），預設 gemini-3.5-flash-lite（官方標示最快、有免費額度）
    anthropic   Claude API（anthropic 套件），預設 claude-haiku-5-5（需預付額度）
金鑰只從環境變數讀取；可以放在專案根目錄的 .env（已在 .gitignore，絕對不 commit）。

可用環境變數：
    AGENT_PROVIDER      gemini（預設）/ anthropic
    GEMINI_API_KEY      provider = gemini 時必填（也接受 GOOGLE_API_KEY）
    ANTHROPIC_API_KEY   provider = anthropic 時必填
    AGENT_MODEL         覆寫預設模型
    AGENT_TIMEOUT_S     預設 10
    AGENT_EFFORT        （anthropic）預設 low；API 不接受時設成空字串
    AGENT_THINKING      （gemini）思考程度 minimal / low / medium / high；預設不送，用模型預設
"""
import os
from pathlib import Path
from typing import List, Optional, Protocol

DEFAULT_MODEL = "claude-haiku-5-5"  # anthropic 預設
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


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
        self.model = model or os.environ.get("AGENT_MODEL") or DEFAULT_MODEL
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


class GeminiClient:
    """Google Gemini API。few-shot 的 assistant 回合對應 Gemini 的 "model" 角色；強制 JSON 輸出。"""

    def __init__(self, model: Optional[str] = None, timeout_s: Optional[float] = None, thinking: Optional[str] = None, sdk_client=None):
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise RuntimeError("未安裝 google-genai 套件：python -m pip install google-genai") from e
        if sdk_client is None and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError("找不到 GEMINI_API_KEY：請到 Google AI Studio 建立金鑰，放在專案根目錄的 .env（不要 commit）")
        self._types = types
        self.model = model or os.environ.get("AGENT_MODEL") or DEFAULT_GEMINI_MODEL
        self.timeout_s = timeout_s if timeout_s is not None else float(os.environ.get("AGENT_TIMEOUT_S", "10"))
        self.thinking = thinking if thinking is not None else os.environ.get("AGENT_THINKING", "")
        self._client = sdk_client or genai.Client()  # SDK 自動讀 GEMINI_API_KEY / GOOGLE_API_KEY
        self.name = f"gemini:{self.model}"

    def complete(self, system: str, messages: List[dict]) -> str:
        t = self._types
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
        cfg = dict(system_instruction=system, response_mime_type="application/json",
                   http_options=t.HttpOptions(timeout=int(self.timeout_s * 1000)))  # SDK 的 timeout 單位是毫秒
        if self.thinking:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_level=self.thinking.upper())
        resp = self._client.models.generate_content(model=self.model, contents=contents, config=t.GenerateContentConfig(**cfg))
        return resp.text or ""


def create_llm_client(provider: Optional[str] = None):
    """依 AGENT_PROVIDER（或參數）建立客戶端。"""
    provider = (provider or os.environ.get("AGENT_PROVIDER") or "gemini").strip().lower()
    if provider == "gemini":
        return GeminiClient()
    if provider == "anthropic":
        return AnthropicClient()
    raise ValueError(f"不支援的 AGENT_PROVIDER：{provider!r}（可用 gemini / anthropic）")


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
