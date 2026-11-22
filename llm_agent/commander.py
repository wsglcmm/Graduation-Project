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
    - 定期 / 目標事件的請求連續失敗 LLM_FAIL_FALLBACK 次（例如額度用完一直 429）：由規則式照有效指令接手，LLM 恢復後交回。
    - 背景請求進行中操作員下了新指令：舊請求的結果標記「已被取代」，不套用。
"""
import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .fleet_prompt import SYSTEM_PROMPT, build_fleet_messages, summarize_state
from .fleet_schema import Plan, parse_plan
from .rule_planner import _parse_operator, rule_plan

MAX_OPERATOR_RETRIES = 2
LLM_FAIL_FALLBACK = 3  # LLM 連續失敗這麼多次，就由規則式接手（照操作員有效指令）
LOCKING_MODES = ("hold", "return_home")  # 操作員下這些指令後，Agent 暫停到下一個操作員指令
ALLOWED_UNDER = {"search": ("search",), "track": ("track", "encircle", "search"), "encircle": ("track", "encircle", "search"),
                 "capture": ("capture", "search")}
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
    note: Optional[str] = None  # 合法但被 C 調整時的說明（例如抓拍期間角色鎖定）

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Commander:
    drone_ids: Sequence[str]
    llm: Optional[object] = None  # 有 complete(system, messages) 的物件；None = 只用規則式
    period_s: float = 2.0
    strategy: str = "encircle"  # 沒有操作員指令時規則式用的預設策略：encircle / capture
    capture_cfg: Optional[object] = None  # planner.capture.CaptureConfig；和規劃器用同一份，角色分派才會對上捕捉區
    clock: Callable[[], float] = time.monotonic  # 量測延遲用的牆鐘
    current: Optional[Plan] = None
    current_id: Optional[str] = None
    events: List[AgentEvent] = field(default_factory=list)

    def __post_init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1) if self.llm is not None else None
        self._pending: Optional[tuple] = None  # (future, trigger, t_ns, start, cmd)
        self._last_ask_ns: Optional[int] = None
        self._last_status: dict = {}
        self._role_lock: Optional[Tuple[Optional[int], Dict[str, str]]] = None  # (target_id, 各機角色)
        self._llm_fail_streak = 0  # LLM 連續失敗次數（API 失敗或不合法）
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

    def _capture_geometry(self, drones: Sequence, targets: Sequence[dict], zone) -> Optional[dict]:
        """給 LLM 的捕捉區事實：capture 進行中用規劃器的捕捉區；否則算出「如果現在改用 capture」的捕捉區。"""
        if self.capture_cfg is None:
            return None
        live = [t for t in targets if t["status"] != "lost"]
        if not live:
            return None
        cur = self.current.target_id if self.current is not None else None
        tgt = next((t for t in live if t["target_id"] == cur), live[0])
        existing = zone is not None and self.current is not None and self.current.mode == "capture"
        if not existing:
            from planner.capture import propose_zone

            p, v = tgt["position_world_ned_m"], tgt.get("velocity_world_ned_mps") or {"n": 0.0, "e": 0.0}
            zone = propose_zone(p["n"], p["e"], drones, self.capture_cfg, 0.0, (v["n"], v["e"]))
        from .fleet_prompt import capture_geometry

        return capture_geometry(drones, tgt, zone, existing)

    def _lock_roles(self, plan: Plan, trigger: str) -> Tuple[Plan, Optional[str]]:
        """同一次捕捉期間鎖定角色（使用者 11/22 選擇方案 A）。

        11/22 LLM 實測：知道捕捉區後，LLM 在抓拍途中會依最新位置一直換角色，無人機要重新飛位置，
        抵銷了好處；規則式在同一次捕捉期間本來就固定角色。所以：
        - capture 開始（或操作員下新指令、換目標）時，照 LLM / 規則式的分派，並記住
        - 之後同一個目標的 capture 計畫若改了角色，改回記住的分派，並在事件記一筆 note（不算不合法）
        - 離開 capture 模式就解除
        """
        if plan.mode != "capture":
            self._role_lock = None
            return plan, None
        locked = self._role_lock
        if trigger == "operator" or locked is None or locked[0] != plan.target_id or set(locked[1]) != set(self.drone_ids):
            self._role_lock = (plan.target_id, {a.drone_id: a.role for a in plan.assignments})
            return plan, None
        proposed = {a.drone_id: a.role for a in plan.assignments}
        if proposed == locked[1]:
            return plan, None
        changed = "、".join(f"{d} {proposed[d]}→{locked[1][d]}" for d in sorted(proposed) if proposed[d] != locked[1][d])
        fixed = plan.model_copy(update={"assignments": [a.model_copy(update={"role": locked[1][a.drone_id]})
                                                        for a in plan.assignments]})
        return fixed, f"角色鎖定：同一次捕捉期間維持原分派（提議的 {changed} 未採用）"

    def _accept(self, plan: Plan, ev_kw: dict, has_target: bool = True) -> AgentEvent:
        plan, note = self._lock_roles(plan, ev_kw.get("trigger", ""))
        self.current, self.current_id = plan, next(self._ids)
        if ev_kw.get("trigger") == "operator":
            self.directive, self.directive_text = plan.mode, ev_kw.get("operator_command")
            if plan.mode == "search":
                # 例如「包圍那台車」但目前看不到車 → 只能先搜尋；這不是操作員要「只搜尋」。
                # 記住操作員真正要的模式（包圍 / 追蹤 / 捕捉，都允許先 search），發現目標後照指令執行；
                # 看不出要什麼就不鎖定。之前直接不鎖定，結果發現目標後回到預設策略，指令被忘記（實測發現）。
                # 不再看 has_target：11/22 LLM 實測，目標是 predicted（短暫沒看到）時 LLM 也選了 search，
                # 結果指令被鎖成「只能 search」，整次都沒有抓拍。操作員本來就要搜尋時才鎖定 search。
                intent = _parse_operator(self.directive_text)
                if intent in ("track", "encircle", "capture"):
                    self.directive = intent
                elif not (has_target or intent == "search"):
                    self.directive = self.directive_text = None
                # 其他情況（看得到目標、指令也不是追 / 包圍 / 捕捉）：LLM 判斷操作員就是要搜尋，維持 search
        ev = AgentEvent(plan_id=self.current_id, valid=True, plan=plan.model_dump(), note=note, **ev_kw)
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

    def tick(self, t_ns: int, drones: Sequence, targets: Sequence[dict], zone=None) -> List[AgentEvent]:
        """每個模擬步呼叫一次。回傳這一步新產生的事件（可能為空）。
        zone：規劃器目前的捕捉區（capture 進行中才有）；給 LLM 的 capture_geometry 用。"""
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
            geo = self._capture_geometry(drones, targets, zone)
            msgs = build_fleet_messages(summarize_state(drones, targets, t_ns, prev, geo), cmd, standing)
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
            self._llm_fail_streak = 0
            return out
        self._llm_fail_streak += 1
        if trig == "operator":
            if self._requeue(cmd):
                out[-1].error += f"（自動重送 {self._op_retries}/{MAX_OPERATOR_RETRIES}）"
            elif not self._has_cmd():
                out.append(self._rule(t_ns, drones, targets, trig, cmd, note="LLM 重試仍失敗，改用規則式解讀操作員指令"))
        elif self.current is None:
            out.append(self._rule(t_ns, drones, targets, trig, cmd, note="LLM 失敗，改用規則式"))
        elif self._llm_fail_streak >= LLM_FAIL_FALLBACK:
            # 11/22 實測：免費額度用完（429）後 LLM 一直失敗，只能沿用最後一個計畫（剛好是 search），
            # 操作員要的抓拍整次都沒有執行。連續失敗就由規則式照操作員有效指令接手，LLM 恢復後再交回。
            out.append(self._rule(t_ns, drones, targets, trig, cmd, rule_cmd=self.directive_text,
                                  note=f"LLM 連續 {self._llm_fail_streak} 次失敗，這一輪改用規則式（照操作員有效指令）"))
        return out

    def _rule(self, t_ns, drones, targets, trig, cmd, note: Optional[str] = None, rule_cmd: Optional[str] = None) -> AgentEvent:
        start = self.clock()
        plan = rule_plan(drones, targets, rule_cmd if rule_cmd is not None else cmd, strategy=self.strategy, previous=self.current,
                         capture_cfg=self.capture_cfg)
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
