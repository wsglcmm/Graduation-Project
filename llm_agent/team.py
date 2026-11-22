"""多 Agent 團隊：協調 Agent（選策略、分角色）+ 每架一個各機 Agent（選自己的動作）。

    操作員自然語言 ─→ 協調 Agent（commander.Commander，規格 §10.5 Plan）
                          │ 每架的角色
                          ▼
                     各機 Agent × 3（drone_agent.DroneAgent，角色專屬動作白名單）
                          │ 每架的動作
                          ▼
                     確定性規劃器（planner/）→ GoalPoint → A

介面刻意和 Commander 相容（tick / command / current / current_id / events / close / wait_idle），
Demo 引擎只要多讀 actions() 與 drone_events 就能切換「單一指揮官 / 多 Agent」做比較。
"""
from typing import Dict, List, Optional, Sequence

from planner.capture import CaptureConfig

from .commander import Commander
from .drone_agent import DroneAgent, DroneAgentEvent, DroneContext


class Team:
    def __init__(self, commander: Commander, drone_llms: Optional[Dict[str, object]] = None, drone_period_s: float = 6.0,
                 capture_cfg: CaptureConfig = CaptureConfig()):
        self.commander = commander
        llms = drone_llms or {}
        self.agents: Dict[str, DroneAgent] = {
            did: DroneAgent(did, llm=llms.get(did), period_s=drone_period_s, capture_cfg=capture_cfg)
            for did in commander.drone_ids}
        self.drone_events: List[DroneAgentEvent] = []
        self.new_drone_events: List[DroneAgentEvent] = []

    @property
    def capture_cfg(self) -> Optional[CaptureConfig]:
        return self.commander.capture_cfg

    @capture_cfg.setter
    def capture_cfg(self, cfg: CaptureConfig) -> None:
        """協調 Agent 與各機 Agent 用同一份捕捉設定（DemoEngine 會設成和規劃器相同）。"""
        self.commander.capture_cfg = cfg
        for ag in self.agents.values():
            ag.capture_cfg = cfg

    # ---- 和 Commander 相容的部分 ----
    @property
    def drone_ids(self):
        return self.commander.drone_ids

    @property
    def current(self):
        return self.commander.current

    @property
    def current_id(self):
        return self.commander.current_id

    @property
    def events(self):
        return self.commander.events

    @property
    def planner_name(self) -> str:
        return self.commander.planner_name

    @property
    def directive(self):
        return self.commander.directive

    @property
    def directive_text(self):
        return self.commander.directive_text

    @property
    def paused(self) -> bool:
        return self.commander.paused

    def command(self, text: str) -> None:
        self.commander.command(text)

    def tick(self, t_ns: int, drones: Sequence, targets: Sequence[dict], zone=None) -> list:
        """先讓協調 Agent 決定角色，再讓每架的 Agent 決定動作。回傳協調 Agent 的事件；各機事件放在 new_drone_events。"""
        events = self.commander.tick(t_ns, drones, targets, zone)
        self.new_drone_events = []
        plan = self.commander.current
        if plan is not None:
            by_id = {d.drone_id: d for d in drones}
            tgt = next((t for t in targets if t["target_id"] == plan.target_id), None)
            for a in plan.assignments:
                d = by_id.get(a.drone_id)
                if d is None:
                    continue
                ctx = DroneContext(d, a.role, plan.mode, tgt, zone if plan.mode == "capture" else None,
                                   [m for m in drones if m.drone_id != a.drone_id])
                self.new_drone_events.extend(self.agents[a.drone_id].tick(t_ns, ctx))
        self.drone_events.extend(self.new_drone_events)
        return events

    def actions(self) -> Dict[str, str]:
        """每架目前的動作（給規劃器）。"""
        return {did: ag.current.action for did, ag in self.agents.items() if ag.current is not None}

    def wait_idle(self, timeout_s: float = 30.0) -> None:
        self.commander.wait_idle(timeout_s)
        for ag in self.agents.values():
            ag.wait_idle(timeout_s)

    def close(self) -> None:
        self.commander.close()
        for ag in self.agents.values():
            ag.close()
