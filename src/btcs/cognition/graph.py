"""The cognitive cycle as an explicit LangGraph state machine.

The project plan asks for a cyclic Sense -> Integrate -> Reason -> Act graph, and
this is it. What the graph buys, concretely, is two things: the stages become
inspectable and individually testable rather than being buried inside one
method, and LangGraph's checkpointer lets a long run be paused and resumed at a
stage boundary.

What it deliberately does *not* buy is a separate API call per node. Each stage
shares nearly all of its context with the others, so one request per node would
multiply the bill roughly fourfold for very little extra behaviour (OPEN-6).
Sense and Integrate are local work; Reason makes the single call; Act returns
the decision for the runner to dispatch.

The graph is optional. `cognition.use_graph` selects it; with the flag off the
brain is called directly and the behaviour is identical. That is on purpose --
if the graph ever changes an outcome, one of the two paths is wrong, and there
is a test that runs both and compares.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig

from btcs.brains.base import Brain, TurnBridge
from btcs.config import Config
from btcs.world.agents import Agent
from btcs.world.types import Decision, Percept


class CycleState(TypedDict, total=False):
    """What flows around the loop.

    Kept to plain data so a checkpointed state can be written to disk and read
    back without needing live simulation objects.
    """

    tick: int
    agent_id: str
    percept: Percept
    memory: dict[str, str]
    decision: Decision
    stages: list[str]


class CognitiveCycle:
    """Sense -> Integrate -> Reason -> Act, wired as a LangGraph state machine."""

    __slots__ = ("_config", "_brain", "_graph", "_checkpointer")

    def __init__(self, config: Config, brain: Brain) -> None:
        self._config = config
        self._brain = brain
        self._checkpointer = None
        self._graph = self._build()

    def _build(self) -> Any:
        from langgraph.graph import END, StateGraph

        graph = StateGraph(CycleState)
        graph.add_node("sense", self._sense)
        graph.add_node("integrate", self._integrate)
        graph.add_node("reason", self._reason)
        graph.add_node("act", self._act)

        graph.set_entry_point("sense")
        graph.add_edge("sense", "integrate")
        graph.add_edge("integrate", "reason")
        graph.add_edge("reason", "act")
        # The cycle closes at the tick boundary rather than inside the graph:
        # one pass is one decision, and the runner drives the repetition. A
        # self-edge here would let an organism take several turns per tick.
        graph.add_edge("act", END)

        try:
            from langgraph.checkpoint.memory import MemorySaver

            self._checkpointer = MemorySaver()
            return graph.compile(checkpointer=self._checkpointer)
        except ImportError:  # pragma: no cover - checkpointing is optional
            return graph.compile()

    # -- stages ------------------------------------------------------------
    #
    # The bridge and agent are passed through the invocation config rather than
    # the state, because they are live objects and the state must stay
    # serialisable for checkpointing.

    def _sense(self, state: CycleState, config: RunnableConfig) -> CycleState:
        context = _context(config)
        bridge: TurnBridge = context["bridge"]
        percept: Percept = context["percept"]
        return {"percept": percept, "agent_id": bridge.service.resolve(bridge.token).id,
                "stages": [*state.get("stages", []), "sense"]}

    def _integrate(self, state: CycleState, config: RunnableConfig) -> CycleState:
        bridge: TurnBridge = _context(config)["bridge"]
        return {"memory": bridge.memory(),
                "stages": [*state.get("stages", []), "integrate"]}

    def _reason(self, state: CycleState, config: RunnableConfig) -> CycleState:
        """The only stage that can cost money."""
        context = _context(config)
        agent: Agent = context["agent"]
        bridge: TurnBridge = context["bridge"]
        decision = self._brain.decide(state["percept"], agent, bridge)
        return {"decision": decision, "stages": [*state.get("stages", []), "reason"]}

    def _act(self, state: CycleState, config: RunnableConfig) -> CycleState:
        # The decision is handed back rather than executed. Dispatch belongs to
        # the runner, so the service's one-act-per-turn gate stays the single
        # place an action can happen.
        return {"stages": [*state.get("stages", []), "act"]}

    # -- driving -----------------------------------------------------------

    def run(self, percept: Percept, agent: Agent, bridge: TurnBridge, tick: int) -> Decision:
        """One pass of the cycle. Returns the decision for the runner to dispatch."""
        result = self._graph.invoke(
            {"tick": tick, "stages": []},
            config={
                "configurable": {
                    "thread_id": f"{agent.id}",
                    "percept": percept,
                    "agent": agent,
                    "bridge": bridge,
                }
            },
        )
        return result["decision"]


def _context(config: RunnableConfig | None) -> dict[str, Any]:
    if not config or "configurable" not in config:
        raise RuntimeError("cognitive cycle invoked without its live context")
    return config["configurable"]
