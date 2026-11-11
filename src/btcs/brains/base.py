"""The brain interface.

Everything that decides what an organism does implements this one method. The
mock brain and the LLM brain are interchangeable behind it, which is the whole
reason the simulation can be built, tuned, and tested before any money is spent.

The interface is also, deliberately, the shape the MCP tool surface will take in
Phase 2: a percept goes out, a decision comes back, and no simulation array ever
crosses the boundary in either direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from btcs.world.agents import Agent
from btcs.world.types import Decision, Percept

if TYPE_CHECKING:
    from btcs.sim.service import SimulationService


@dataclass(frozen=True, slots=True)
class TurnBridge:
    """A brain's handle on the world during one open turn.

    Exists so a deliberating brain can look things up mid-thought without ever
    holding an agent id or being able to reach another organism. The runner
    builds one per turn and drops it when the turn closes.

    Note what is *not* here: no action methods. A brain proposes its action by
    returning a Decision, and the runner dispatches it. That keeps "one physical
    act per turn" enforced in a single place rather than trusted to every brain.
    """

    service: "SimulationService"
    token: str

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one deliberation tool. Refusals come back as data.

        Only the read-and-remember tools are routed here. An action name is
        refused rather than executed, because actions must flow through the
        runner's dispatch so the turn accounting stays in one place.
        """
        from btcs.sim.service import ErrorCode, ToolError

        allowed = {"sense_gradient", "recall", "remember"}
        if name not in allowed:
            return {
                "ok": False,
                "error": "rejected",
                "detail": f"{name} is not a deliberation tool.",
            }

        try:
            match name:
                case "sense_gradient":
                    return self.service.sense_gradient(self.token)
                case "recall":
                    return self.service.recall(self.token, arguments.get("key"))
                case "remember":
                    return self.service.remember(
                        self.token, arguments.get("key", ""), arguments.get("content", "")
                    )
        except ToolError as refusal:
            return refusal.to_dict()
        return {"ok": False, "error": str(ErrorCode.REJECTED), "detail": "unreachable"}

    def memory(self) -> dict[str, str]:
        """This organism's memory, undegraded, for building the prompt.

        Degradation is applied when the *organism* recalls; the prompt builder
        reads the true store and the percept's degradation flags tell the
        organism its recall is unreliable. Corrupting it twice would compound.
        """
        agent = self.service.resolve(self.token)
        return self.service.memory_of(agent.id)


@runtime_checkable
class Brain(Protocol):
    """Turns perception into action, and reports what the thought cost."""

    @property
    def name(self) -> str:
        """Identifier recorded in the run log."""
        ...

    def decide(
        self, percept: Percept, agent: Agent, bridge: TurnBridge | None = None
    ) -> Decision:
        """Choose an action.

        The returned `Decision.tokens` is not decorative: the engine converts it
        into internal entropy via law L3. An implementation that under-reports
        its token use is cheating at thermodynamics.

        `bridge` is present only while the organism's turn is open. A brain that
        does not deliberate mid-turn -- the mock brain -- may ignore it.
        """
        ...

    def on_death(self, agent: Agent) -> None:
        """Release any per-organism state. Default implementations may no-op."""
        ...
