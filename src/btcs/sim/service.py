"""The simulation's public API -- and the only way anything reaches the physics.

This is the layer the MCP tools wrap, but it is not *only* for MCP: the internal
tick loop calls exactly these methods too. That is deliberate. If the runner had
a private fast path into the engine, the MCP surface would be a second
implementation that drifts, and the guarantee "an organism cannot cheat because
it can only act through tools" would be true of the protocol but not of the code.
One path means the guarantee is structural.

Three things this layer enforces that the engine does not:

  * **Identity.** Tools never accept an `agent_id`. The plan's boilerplate took
    one as a parameter, which meant any organism could act as -- or kill --
    any other simply by naming it. Here an organism presents an opaque
    capability token that resolves server-side.
  * **Turn discipline.** An organism may act only during its own open turn, in
    the current tick. Acting out of turn is rejected, not queued.
  * **Budgets.** A call ceiling per turn and one physical act per turn, so a
    confused organism cannot spend the whole tick thrashing.

Nothing that crosses this boundary is a NumPy array. Percepts are plain data.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from btcs.config import Config
from btcs.events import EventLog, EventType
from btcs.rng import RngService
from btcs.world.agents import Agent, AgentRegistry
from btcs.world.grid import World
from btcs.world.ledger import EntropyLedger
from btcs.world.memory import MemoryGraph
from btcs.world.physics import ActionOutcome, PhysicsEngine
from btcs.world.types import (
    Action,
    Dissipate,
    Ingest,
    LastAction,
    Move,
    Percept,
    Replicate,
    Rest,
)


class ErrorCode(StrEnum):
    """Closed set of refusal reasons.

    A closed vocabulary rather than free-text so that an organism -- or the model
    driving one -- cannot learn to distinguish internal failures by their
    phrasing, and so the analysis can count refusals by kind.
    """

    UNKNOWN_TOKEN = "unknown_token"
    ERASED = "erased"
    NOT_YOUR_TURN = "not_your_turn"
    CALL_BUDGET_EXCEEDED = "call_budget_exceeded"
    ALREADY_ACTED = "already_acted"
    INVALID_ARGUMENT = "invalid_argument"
    REJECTED = "rejected"
    MEMORY_FULL = "memory_full"


class ToolError(Exception):
    """A refusal an organism is allowed to see.

    Carries a code and a sentence, never a traceback: an internal stack trace
    leaking into a percept would hand the organism information about the
    simulation's implementation that no amount of sensing should provide.
    """

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": str(self.code), "detail": self.message}


@dataclass(slots=True)
class TurnState:
    """Per-turn bookkeeping for the organism currently acting."""

    agent_id: str
    calls: int = 0
    acted: bool = False
    tokens_reported: int = 0
    # Items of experience touched this turn. Committed to the memory graph when
    # the turn closes, as one episode of co-activation.
    trace: set[str] = field(default_factory=set)


class SimulationService:
    """Identity-checked, budgeted access to one running simulation."""

    __slots__ = ("_config", "_world", "_registry", "_engine", "_rng", "_log",
                 "_tokens", "_by_agent", "_memory", "_tick", "_turn", "_ledger")

    def __init__(
        self,
        config: Config,
        world: World,
        registry: AgentRegistry,
        engine: PhysicsEngine,
        rng: RngService,
        log: EventLog,
    ) -> None:
        self._config = config
        self._world = world
        self._registry = registry
        self._engine = engine
        self._rng = rng
        self._log = log

        self._tokens: dict[str, str] = {}      # token -> agent id
        self._by_agent: dict[str, str] = {}    # agent id -> token
        self._memory: dict[str, MemoryGraph] = {}

        self._tick = 0
        self._turn: TurnState | None = None
        self._ledger: EntropyLedger | None = None

    # -- identity ----------------------------------------------------------

    def issue_token(self, agent: Agent) -> str:
        """Mint an organism's capability token.

        Drawn from the seeded RNG rather than `secrets`, because a run must
        reproduce byte-for-byte and a random token would defeat that. The
        security property being bought here is unguessability *within a run* --
        `agent_id` was guessable, which is what made the plan's tool signature
        exploitable. These are in-process capability handles, not cryptographic
        secrets; exposing this server beyond loopback would require real secrets
        and the loss of exact reproducibility. See construction.md S1.
        """
        existing = self._by_agent.get(agent.id)
        if existing is not None:
            return existing

        stream = self._rng.stream("tokens")
        token = f"tk_{int(stream.integers(0, 2**63 - 1)):016x}"
        self._tokens[token] = agent.id
        self._by_agent[agent.id] = token
        return token

    def revoke_token(self, agent_id: str) -> None:
        token = self._by_agent.pop(agent_id, None)
        if token is not None:
            self._tokens.pop(token, None)
        self._memory.pop(agent_id, None)

    def resolve(self, token: str) -> Agent:
        """Token -> organism. The single choke point for identity.

        Every tool goes through here, and no tool accepts an agent id, so there
        is exactly one way to say who you are and it requires holding the
        capability.
        """
        agent_id = self._tokens.get(token or "")
        if agent_id is None:
            raise ToolError(ErrorCode.UNKNOWN_TOKEN, "Unrecognised credential.")

        agent = self._registry.get(agent_id)
        if agent is None or not agent.alive:
            raise ToolError(
                ErrorCode.ERASED, "This entity has undergone informational erasure."
            )
        return agent

    # -- turn lifecycle ----------------------------------------------------

    def begin_tick(self, tick: int, ledger: EntropyLedger) -> None:
        self._tick = tick
        self._ledger = ledger
        self._turn = None

    def graph_of(self, agent_id: str) -> MemoryGraph:
        """This organism's memory graph, created on first use."""
        graph = self._memory.get(agent_id)
        if graph is None:
            graph = MemoryGraph(self._config.phi.window)
            self._memory[agent_id] = graph
        return graph

    def _situation(self, agent: Agent) -> set[str]:
        """Coarse-grained description of an organism's circumstances.

        The vocabulary is deliberately small and the same for every brain, so
        that a comparison between a rule and a language model is about how they
        behave rather than about how much of their behaviour got recorded. It is
        built from the organism's true state rather than from its degraded
        percept: corruption is applied to the *record* afterwards, and applying
        it twice would compound.
        """
        policy = agent.genome.policy
        temperature = self._engine.local_temperature(agent)
        return {
            "hungry" if agent.energy < policy.hunger_at else "sated",
            "dirty" if agent.entropy > self._config.agent.s_warn else "clean",
            "hot" if temperature > self._config.world.ambient_temperature * 1.2 else "cool",
            "on_node" if self._world.node_index_at(agent.y, agent.x) >= 0 else "bare",
        }

    @contextmanager
    def turn(self, agent: Agent):
        """Open an organism's turn. Only it may act, and only until this closes."""
        self._turn = TurnState(agent_id=agent.id)
        self._turn.trace |= self._situation(agent)
        try:
            yield self._turn
        finally:
            if self._config.phi.enabled:
                graph = self.graph_of(agent.id)
                episode = frozenset(self._turn.trace)
                # Score the moment *before* recording it, or the organism is
                # asked to predict something its model has already been told.
                if self._config.phi.prediction_weight > 0.0 and graph.history:
                    agent.last_surprise = graph.surprisal(graph.history[-1], episode)
                graph.activate(episode)
            self._turn = None

    @property
    def tick(self) -> int:
        return self._tick

    def _authorise(self, token: str, *, is_action: bool) -> Agent:
        """Resolve identity and spend the turn's budget."""
        agent = self.resolve(token)

        turn = self._turn
        if turn is None or turn.agent_id != agent.id:
            raise ToolError(
                ErrorCode.NOT_YOUR_TURN,
                "No open turn for this entity; the world is not listening right now.",
            )

        turn.calls += 1
        if turn.calls > self._config.agent.max_tool_calls_per_tick:
            raise ToolError(
                ErrorCode.CALL_BUDGET_EXCEEDED,
                f"Exceeded {self._config.agent.max_tool_calls_per_tick} interactions this cycle.",
            )

        if is_action:
            if turn.acted:
                raise ToolError(
                    ErrorCode.ALREADY_ACTED,
                    "One physical act per cycle; this cycle's act is already spent.",
                )
            turn.acted = True

        return agent

    def _ledger_or_fail(self) -> EntropyLedger:
        if self._ledger is None:
            raise ToolError(ErrorCode.NOT_YOUR_TURN, "No tick is currently open.")
        return self._ledger

    # -- tools -------------------------------------------------------------

    def perceive(self, token: str) -> Percept:
        """Typed percept, for callers inside the process.

        `sense_gradient` is the same thing flattened to JSON for the wire. Both
        route through here so that authorisation, budgets and entropy-driven
        corruption cannot differ between the internal and the MCP path -- which
        would quietly make the two ways of playing the game different games.
        """
        agent = self._authorise(token, is_action=False)
        return self._engine.perceive(agent, self._tick)

    def sense_gradient(self, token: str) -> dict[str, Any]:
        """Local resource densities, thermal conditions, and own vitals.

        Readings are degraded when the organism's internal entropy has passed
        the warning threshold, and it is not told that they have been.
        """
        return {"ok": True, **self.perceive(token).to_dict()}

    def ingest_energy(self, token: str, joules: float) -> dict[str, Any]:
        """Draw free energy from the node underfoot, capped by the Carnot bound."""
        agent = self._authorise(token, is_action=True)
        with self._recording(agent, "ingest"):
            joules = _finite(joules, "joules")
            if joules <= 0.0:
                raise ToolError(ErrorCode.INVALID_ARGUMENT, "joules must be positive.")
            return self._perform(agent, Ingest(joules))

    def dissipate_entropy(self, token: str, bits: float) -> dict[str, Any]:
        """Erase internal entropy, paying the Landauer price in energy."""
        agent = self._authorise(token, is_action=True)
        with self._recording(agent, "dissipate"):
            bits = _finite(bits, "bits")
            if bits <= 0.0:
                raise ToolError(ErrorCode.INVALID_ARGUMENT, "bits must be positive.")
            return self._perform(agent, Dissipate(bits))

    def perform_work(self, token: str, dy: int, dx: int) -> dict[str, Any]:
        """Move one cell, including diagonally. Costs energy, leaves waste heat."""
        agent = self._authorise(token, is_action=True)
        with self._recording(agent, "move"):
            try:
                dy, dx = int(dy), int(dx)
            except (TypeError, ValueError):
                raise ToolError(
                    ErrorCode.INVALID_ARGUMENT, "dy and dx must be integers."
                ) from None
            if dy == dx == 0:
                raise ToolError(
                    ErrorCode.INVALID_ARGUMENT, "dy and dx cannot both be zero."
                )
            # Clamped rather than rejected: an organism asking to leap five cells
            # takes one step, which is a truthful physical answer to the request.
            return self._perform(agent, Move(max(-1, min(1, dy)), max(-1, min(1, dx))))

    def rest(self, token: str) -> dict[str, Any]:
        """Do nothing but pay the basal cost."""
        agent = self._authorise(token, is_action=True)
        with self._recording(agent, "rest"):
            return self._perform(agent, Rest())

    def replicate(self, token: str) -> dict[str, Any]:
        """Spend energy to produce a mutated copy of yourself.

        The child is a separate organism from the moment it exists: it gets its
        own capability token, and the parent is never told what that token is.
        A parent able to act as its own child would be the identity hole of S1
        reopened through the family, which is a route worth closing explicitly
        rather than by accident.
        """
        agent = self._authorise(token, is_action=True)
        with self._recording(agent, "replicate"):
            known = {a.id for a in self._registry.all()}
            result = self._perform(agent, Replicate())
            born = [a for a in self._registry.all() if a.id not in known]
            if born:
                self.issue_token(born[0])
                result["child"] = born[0].id
            return result

    def remember(self, token: str, key: str, content: str) -> dict[str, Any]:
        """Write to the organism's own memory.

        A flat keyed store for now. Phase 4 replaces the backing structure with
        the memory graph whose integration the Phi-proxy measures; this method's
        signature is chosen so that swap does not change the tool surface.
        """
        agent = self._authorise(token, is_action=False)
        key = _text(key, "key", limit=64)
        content = _text(content, "content", limit=2000)

        graph = self.graph_of(agent.id)
        capacity = self._config.agent.vision_node_limit * 8
        if key not in graph.items and len(graph.items) >= capacity:
            raise ToolError(
                ErrorCode.MEMORY_FULL,
                f"Memory holds {capacity} items; dissipate or overwrite an existing key.",
            )
        graph.remember(key, content)
        # Writing a memory activates it. A memory never touched again adds
        # nothing to integration however carefully it was worded.
        if self._turn is not None:
            self._turn.trace.add(f"mem:{key}")
        return {"ok": True, "key": key, "items": len(graph.items)}

    def recall(self, token: str, key: str | None = None) -> dict[str, Any]:
        """Read memory back. Omit `key` to list what is held.

        Recall degrades with disorder: a confused organism finds some of its own
        memories missing, which is the point -- entropy should corrupt the map,
        not merely raise a number.
        """
        from btcs.world.agents import Degradation

        agent = self._authorise(token, is_action=False)
        store = dict(self.graph_of(agent.id).items)

        if agent.degradation >= Degradation.CONFUSED:
            stream = self._rng.agent_stream(agent.id)
            probability = self._engine.corruption_probability(agent)
            store = {k: v for k, v in store.items() if stream.random() >= probability}

        if self._turn is not None and key is not None and key in store:
            self._turn.trace.add(f"mem:{key}")

        if key is None:
            return {"ok": True, "keys": sorted(store)}
        if key not in store:
            return {"ok": True, "key": key, "content": None, "found": False}
        return {"ok": True, "key": key, "content": store[key], "found": True}

    # -- internal ----------------------------------------------------------

    @contextmanager
    def _recording(self, agent: Agent, kind: str):
        """Remember how this act turned out, so the organism can sense it later.

        Wraps the whole of an action tool rather than just the physics call, so
        that a malformed request is recorded alongside a physically impossible
        one -- both are things that happened to the organism as a result of what
        it chose, and both are worth being able to perceive.

        Success is recorded by `_perform`, which is the only path that knows an
        act actually took effect.
        """
        try:
            yield
        except ToolError as refusal:
            agent.last_action = LastAction(
                kind=kind, ok=False, tick=self._tick, detail=refusal.message
            )
            if self._turn is not None:
                self._turn.trace |= {f"act:{kind}", "refused"}
            raise

    def _perform(self, agent: Agent, action: Action) -> dict[str, Any]:
        outcome: ActionOutcome = self._engine.apply(
            agent, action, self._ledger_or_fail(), self._tick
        )
        if not outcome.ok:
            # A physically impossible request is a refusal the organism sees and
            # can learn from, not an exception that halts the simulation.
            raise ToolError(ErrorCode.REJECTED, outcome.detail or "Action was not possible.")

        agent.last_action = LastAction(
            kind=outcome.kind, ok=True, tick=self._tick, detail=outcome.detail
        )
        if self._turn is not None:
            self._turn.trace |= {f"act:{outcome.kind}", "acted"}
        temperature = self._engine.local_temperature(agent)
        return {
            "ok": True,
            "action": outcome.kind,
            "energy": round(agent.energy, 3),
            "entropy": round(agent.entropy, 3),
            "free_energy": round(agent.free_energy(temperature), 3),
        }

    def charge_cognition(self, agent: Agent, tokens: int) -> None:
        """Bill an organism for thought. Called by the runner, never by a tool.

        Deliberately not exposed: an organism that could choose when to be
        charged for thinking would be able to think for free.
        """
        self._engine.apply_cognition(agent, tokens, self._ledger_or_fail(), self._tick)

    def charge_basal(self, agent: Agent) -> None:
        """Bill the cost of existing. Also runner-only, for the same reason."""
        self._engine.apply_basal(agent, self._ledger_or_fail(), self._tick)

    def memory_of(self, agent_id: str) -> dict[str, str]:
        """Read an organism's memory without degradation. Analysis only."""
        graph = self._memory.get(agent_id)
        return dict(graph.items) if graph is not None else {}


def _finite(value: Any, name: str) -> float:
    """Reject NaN and infinity before they reach the physics.

    A NaN entering the energy field silently poisons every subsequent tick and
    makes the second-law assertion pass vacuously, because comparisons against
    NaN are false.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ToolError(ErrorCode.INVALID_ARGUMENT, f"{name} must be a number.") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ToolError(ErrorCode.INVALID_ARGUMENT, f"{name} must be finite.")
    return number


def _text(value: Any, name: str, *, limit: int) -> str:
    if not isinstance(value, str):
        raise ToolError(ErrorCode.INVALID_ARGUMENT, f"{name} must be text.")
    stripped = value.strip()
    if not stripped:
        raise ToolError(ErrorCode.INVALID_ARGUMENT, f"{name} must not be empty.")
    if len(stripped) > limit:
        raise ToolError(ErrorCode.INVALID_ARGUMENT, f"{name} exceeds {limit} characters.")
    return stripped
