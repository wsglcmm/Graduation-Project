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

操作員指令優先（人在迴路中）：
    - 操作員指令產生的計畫成為「有效指令」（directive），直到操作員下新指令為止。
    - hold / return_home：定期與目標事件觸發全部暫停，不再問 Agent，避免 AI 推翻人的懸停或返航。
    - search：之後的計畫 mode 只能是 search（只在下指令時看得到目標才鎖定；看不到目標時的 search 是不得不然，不鎖）；track / encircle：只能在 track、encircle、search（目標遺失時）之間切換。
      違反的計畫視為不合法，沿用上一個計畫。
    - 操作員指令的請求失敗（例外或不合法）會自動重送，最多 MAX_OPERATOR_RETRIES 次；
      仍失敗就改用規則式解讀，保證操作員的指令一定有回應。
    - 背景請求進行中操作員下了新指令：舊請求的結果標記「已被取代」，不套用。
"""
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

from .fleet_prompt import SYSTEM_PROMPT, build_fleet_messages, summarize_state
from .fleet_schema import Plan, parse_plan
from .rule_planner import rule_plan

MAX_OPERATOR_RETRIES = 2
LOCKING_MODES = ("hold", "return_home")  # 操作員下這些指令後，Agent 暫停到下一個操作員指令
ALLOWED_UNDER = {"search": ("search",), "track": ("track", "encircle", "search"), "encircle": ("track", "encircle", "search")}
SUPERSEDED = "已被操作員新指令取代，未套用"


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
    model: Optional[str] = None  # 實際回應的模型（有備援模型時才看得出差別）；規則式為 None

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
        self._lock = threading.Lock()  # command() 可能來自終端機輸入執行緒
        self._op_retries = 0
        self.directive: Optional[str] = None  # 操作員有效指令對應的 mode
        self.directive_text: Optional[str] = None

    @property
    def planner_name(self) -> str:
        return "llm" if self.llm is not None else "rule"

    @property
    def paused(self) -> bool:
        """操作員下了 hold / return_home：Agent 暫停，直到下一個操作員指令。"""
        return self.directive in LOCKING_MODES

    def command(self, text: str) -> None:
        """操作員輸入自然語言指令；下一次 tick 立即觸發。執行緒安全。"""
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._pending_cmd = text
            self._op_retries = 0

    def _take_cmd(self) -> Optional[str]:
        with self._lock:
            cmd, self._pending_cmd = self._pending_cmd, None
            return cmd

    def _has_cmd(self) -> bool:
        with self._lock:
            return self._pending_cmd is not None

    def _requeue(self, cmd: str) -> bool:
        """操作員指令失敗時重送；操作員已經下了更新的指令就不重送。"""
        with self._lock:
            if self._pending_cmd is not None or self._op_retries >= MAX_OPERATOR_RETRIES:
                return False
            self._op_retries += 1
            self._pending_cmd = cmd
            return True

    def _violates_directive(self, plan: Plan) -> Optional[str]:
        allowed = ALLOWED_UNDER.get(self.directive)
        if allowed and plan.mode not in allowed:
            return f"違反操作員指令「{self.directive_text}」：mode 只能是 {'/'.join(allowed)}，收到 {plan.mode}"
        return None

    def standing_order(self) -> Optional[str]:
        """給 prompt 的一句話：目前仍然有效的操作員指令與限制。"""
        if self.directive is None:
            return None
        allowed = ALLOWED_UNDER.get(self.directive, (self.directive,))
        return f"「{self.directive_text}」（mode 只能是 {' / '.join(allowed)}）"

    def _trigger(self, t_ns: int, targets: Sequence[dict]) -> Optional[str]:
        status = {t["target_id"]: t["status"] for t in targets}
        trig = None
        if self._has_cmd():
            trig = "operator"
        elif self.paused:
            trig = None  # 操作員要求懸停 / 返航：不讓 Agent 自己改變
        elif any(s == "lost" and self._last_status.get(i) != "lost" for i, s in status.items()):
            trig = "target_lost"
        elif any(s != "lost" and self._last_status.get(i, "lost") == "lost" for i, s in status.items()):
            trig = "target_found"
        elif self._last_ask_ns is None or (t_ns - self._last_ask_ns) / 1e9 >= self.period_s:
            trig = "periodic"
        self._last_status = status
        return trig

    def _accept(self, plan: Plan, ev_kw: dict, has_target: bool = True) -> AgentEvent:
        self.current, self.current_id = plan, next(self._ids)
        if ev_kw.get("trigger") == "operator":
            self.directive, self.directive_text = plan.mode, ev_kw.get("operator_command")
            if plan.mode == "search" and not has_target:
                # 例如「包圍那台車」但目前看不到車 → 只能先搜尋；這不是操作員要「只搜尋」，不鎖定
                self.directive = self.directive_text = None
        ev = AgentEvent(plan_id=self.current_id, valid=True, plan=plan.model_dump(), **ev_kw)
        self.events.append(ev)
        return ev

    def _superseded(self, plan: Plan, ev_kw: dict) -> AgentEvent:
        """格式合法，但操作員在請求期間下了新指令：記錄下來（valid = 通過驗證），plan_id 為 None 代表沒有套用。"""
        ev = AgentEvent(plan_id=None, valid=True, plan=plan.model_dump(), error=SUPERSEDED, **ev_kw)
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
            new.extend(self._collect(t_ns, drones, targets, target_ids))

        # 2) 決定要不要問
        if self._pending is not None:
            return new
        trig = self._trigger(t_ns, targets)
        if trig is None:
            return new
        cmd = self._take_cmd() if trig == "operator" else None
        self._last_ask_ns = t_ns
        if self.llm is None:
            # 規則式：沒有新指令時沿用操作員有效指令的文字，讓規則式也遵守它
            new.append(self._rule(t_ns, drones, targets, trig, cmd, rule_cmd=cmd or self.directive_text))
        else:
            prev = self.current.model_dump() if self.current else None
            standing = None if trig == "operator" else self.standing_order()
            msgs = build_fleet_messages(summarize_state(drones, targets, t_ns, prev), cmd, standing)
            self._pending = (self._pool.submit(self.llm.complete, SYSTEM_PROMPT, msgs), trig, t_ns, self.clock(), cmd)
        return new

    def _collect(self, t_ns: int, drones: Sequence, targets: Sequence[dict], target_ids: List[int]) -> List[AgentEvent]:
        fut, trig, _, start, cmd = self._pending
        self._pending = None
        kw = dict(timestamp_ns=t_ns, trigger=trig, planner="llm", latency_ms=(self.clock() - start) * 1000, operator_command=cmd)
        out: List[AgentEvent] = []
        try:
            text = fut.result()
            kw["model"] = getattr(self.llm, "last_model", None)
            res = parse_plan(text, self.drone_ids, target_ids)
            if not res.ok:
                out.append(self._reject(res.error, kw))
            elif trig != "operator" and self._has_cmd():
                out.append(self._superseded(res.plan, kw))
            elif trig != "operator" and (err := self._violates_directive(res.plan)):
                out.append(self._reject(err, kw))
            else:
                out.append(self._accept(res.plan, kw, has_target=bool(target_ids)))
        except Exception as e:  # 網路、金鑰、逾時都在這裡
            out.append(self._reject(f"{type(e).__name__}: {e}"[:200], kw))
        if out[-1].valid:
            return out
        if trig == "operator":
            if self._requeue(cmd):
                out[-1].error += f"（自動重送 {self._op_retries}/{MAX_OPERATOR_RETRIES}）"
            elif not self._has_cmd():
                out.append(self._rule(t_ns, drones, targets, trig, cmd, note="LLM 重試仍失敗，改用規則式解讀操作員指令"))
        elif self.current is None:
            out.append(self._rule(t_ns, drones, targets, trig, cmd, note="LLM 失敗，改用規則式"))
        return out

    def _rule(self, t_ns, drones, targets, trig, cmd, note: Optional[str] = None, rule_cmd: Optional[str] = None) -> AgentEvent:
        start = self.clock()
        plan = rule_plan(drones, targets, rule_cmd if rule_cmd is not None else cmd)
        kw = dict(timestamp_ns=t_ns, trigger=trig, planner="rule", latency_ms=(self.clock() - start) * 1000, operator_command=cmd)
        ev = self._accept(plan, kw, has_target=any(t["status"] != "lost" for t in targets))
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
