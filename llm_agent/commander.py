"""多機指揮迴圈：決定何時問 Agent、在背景呼叫 LLM、驗證、記錄 AgentEvent（規格 §10.7）。

為什麼要背景執行：LLM 一次要 1–數秒，模擬與隊形規劃是 10 Hz，不能等。
所以同一時間只送一個請求，結果回來才驗證、套用；這段期間無人機照上一個合法計畫飛。

觸發時機（trigger）：
    periodic       每 period_s 秒一次
    operator       操作員輸入新指令（立即）
    target_lost    目標狀態變成 lost（立即）
    target_found   目標從 lost / 不存在 變成看得到（立即）

失效處理（安全原則）：LLM 例外、逾時或輸出不合法 → 記錄 valid=false，**沿用上一個合法計畫**；
還沒有任何計畫時改用規則式計畫，確保無人機一定有合法指令。
"""
import itertools
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .fleet_prompt import SYSTEM_PROMPT, build_fleet_messages, summarize_state
from .fleet_schema import Plan, parse_plan
from .rule_planner import rule_plan


@dataclass
class AgentEvent:
    timestamp_ns: int
    plan_id: Optional[str]
    trigger: str
    planner: str
    latency_ms: float
    valid: bool
    plan: Optional[dict]
    error: Optional[str] = None
    operator_command: Optional[str] = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Commander:
    drone_ids: Sequence[str]
    llm: Optional[object] = None  # 有 complete(system, messages) 的物件；None = 只用規則式
    period_s: float = 2.0
    clock: Callable[[], float] = time.monotonic  # 量測延遲用的牆鐘
    current: Optional[Plan] = None
    current_id: Optional[str] = None
    events: List[AgentEvent] = field(default_factory=list)

    def __post_init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1) if self.llm is not None else None
        self._pending: Optional[tuple] = None  # (future, trigger, t_ns, start, cmd)
        self._last_ask_ns: Optional[int] = None
        self._last_status: dict = {}
        self._ids = (f"p{i:04d}" for i in itertools.count(1))
        self._pending_cmd: Optional[str] = None

    @property
    def planner_name(self) -> str:
        return "llm" if self.llm is not None else "rule"

    def command(self, text: str) -> None:
        """操作員輸入自然語言指令；下一次 tick 立即觸發。"""
        self._pending_cmd = text

    def _trigger(self, t_ns: int, targets: Sequence[dict]) -> Optional[str]:
        status = {t["target_id"]: t["status"] for t in targets}
        trig = None
        if self._pending_cmd is not None:
            trig = "operator"
        elif any(s == "lost" and self._last_status.get(i) != "lost" for i, s in status.items()):
            trig = "target_lost"
        elif any(s != "lost" and self._last_status.get(i, "lost") == "lost" for i, s in status.items()):
            trig = "target_found"
        elif self._last_ask_ns is None or (t_ns - self._last_ask_ns) / 1e9 >= self.period_s:
            trig = "periodic"
        self._last_status = status
        return trig

    def _accept(self, plan: Plan, ev_kw: dict) -> AgentEvent:
        self.current, self.current_id = plan, next(self._ids)
        ev = AgentEvent(plan_id=self.current_id, valid=True, plan=plan.model_dump(), **ev_kw)
        self.events.append(ev)
        return ev

    def _reject(self, error: str, ev_kw: dict) -> AgentEvent:
        ev = AgentEvent(plan_id=None, valid=False, plan=None, error=error, **ev_kw)
        self.events.append(ev)
        return ev

    def tick(self, t_ns: int, drones: Sequence, targets: Sequence[dict]) -> List[AgentEvent]:
        """每個模擬步呼叫一次。回傳這一步新產生的事件（可能為空）。"""
        new: List[AgentEvent] = []
        target_ids = [t["target_id"] for t in targets if t["status"] != "lost"]

        # 1) 收背景 LLM 的結果
        if self._pending and self._pending[0].done():
            fut, trig, t0_ns, start, cmd = self._pending
            self._pending = None
            kw = dict(timestamp_ns=t_ns, trigger=trig, planner="llm", latency_ms=(self.clock() - start) * 1000, operator_command=cmd)
            try:
                res = parse_plan(fut.result(), self.drone_ids, target_ids)
                new.append(self._accept(res.plan, kw) if res.ok else self._reject(res.error, kw))
            except Exception as e:  # 網路、金鑰、逾時都在這裡
                new.append(self._reject(f"{type(e).__name__}: {e}"[:200], kw))
            if not new[-1].valid and self.current is None:
                new.append(self._rule(t_ns, drones, targets, trig, cmd, note="LLM 失敗，改用規則式"))

        # 2) 決定要不要問
        trig = self._trigger(t_ns, targets)
        if trig is None or (self._pending is not None):
            return new
        cmd, self._pending_cmd = self._pending_cmd, None
        self._last_ask_ns = t_ns
        if self.llm is None:
            new.append(self._rule(t_ns, drones, targets, trig, cmd))
        else:
            prev = self.current.model_dump() if self.current else None
            msgs = build_fleet_messages(summarize_state(drones, targets, t_ns, prev), cmd)
            self._pending = (self._pool.submit(self.llm.complete, SYSTEM_PROMPT, msgs), trig, t_ns, self.clock(), cmd)
        return new

    def _rule(self, t_ns, drones, targets, trig, cmd, note: Optional[str] = None) -> AgentEvent:
        start = self.clock()
        plan = rule_plan(drones, targets, cmd)
        kw = dict(timestamp_ns=t_ns, trigger=trig, planner="rule", latency_ms=(self.clock() - start) * 1000, operator_command=cmd)
        ev = self._accept(plan, kw)
        if note:
            ev.error = note
        return ev

    def wait_idle(self, timeout_s: float = 30.0) -> None:
        """測試與離線模式用：等背景請求完成。"""
        if self._pending:
            wait([self._pending[0]], timeout=timeout_s)  # 不在這裡丟例外：例外由 tick() 記成 AgentEvent

    def close(self) -> None:
        if self._pool:
            self._pool.shutdown(wait=False, cancel_futures=True)
