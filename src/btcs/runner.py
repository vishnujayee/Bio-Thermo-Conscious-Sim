"""The tick loop.

Assembles a world, a population, an engine and a brain, then runs them forward
while auditing the books every tick. Nothing here decides anything about
physics; it sequences the phases and enforces the invariants.

The two audits at the end of each tick are the reason this file exists:

  * the **second law** -- dS_total must be non-negative, or the run is
    physically meaningless from that point on;
  * the **ledger consistency check** -- the entropy the ledger says organisms
    gained must equal the entropy they actually gained. This is the stronger of
    the two, because it catches the failure the second-law check cannot: a
    future code path that changes an organism's entropy without filing an
    entry would keep dS_total comfortably positive while quietly making the
    accounting fiction.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from btcs.brains.base import Brain, TurnBridge
from btcs.brains.mock import MockBrain
from btcs.cognition.budget import BudgetExceeded
from btcs.config import Config
from btcs.events import EventLog, EventType
from btcs.rng import RngService
from btcs.sim.scenario import ACTIONS, apply_overrides, build as build_scenario
from btcs.sim.service import ErrorCode, SimulationService, ToolError
from btcs.world.agents import AgentRegistry, found_population
from btcs.world.grid import World
from btcs.world.ledger import EntropyLedger, SecondLawViolation
from btcs.world.physics import PhysicsEngine
from btcs.world.types import Dissipate, Ingest, Move, Replicate, Rest, describe


class LedgerInconsistency(AssertionError):
    """The ledger's account of agent entropy disagrees with the organisms themselves.

    Always an engine bug: some code path mutated `agent.entropy` without filing
    a ledger entry, which silently invalidates the second-law audit.
    """


@dataclass(slots=True)
class TickReport:
    tick: int
    population: int
    total_agent_entropy: float
    total_agent_energy: float
    mean_phi: float
    ds_agents: float
    ds_env: float
    ds_total: float
    cumulative_entropy: float
    deaths: int
    world: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    config: Config
    ticks_completed: int
    final_population: int
    extinct: bool
    cumulative_entropy: float
    log_path: Path
    history: list[TickReport]
    wall_seconds: float
    # Set when the run stopped for a reason other than reaching max_ticks or
    # going extinct -- currently only an exhausted spend ceiling.
    halted_reason: str | None = None
    spend_usd: float = 0.0

    def summary(self) -> str:
        outcome = "EXTINCT" if self.extinct else f"{self.final_population} alive"
        return (
            f"{self.ticks_completed} ticks | {outcome} | "
            f"cumulative dS = {self.cumulative_entropy:,.1f} bits | "
            + (f"spend ${self.spend_usd:.4f} | " if self.spend_usd else "")
            + (f"HALTED: {self.halted_reason} | " if self.halted_reason else "")
            + f"{self.wall_seconds:.2f}s | log: {self.log_path}"
        )


class SimulationRunner:
    """Drives a simulation from founding population to extinction or tick limit."""

    __slots__ = ("config", "rng", "log", "world", "registry", "engine", "service",
                 "brain", "cycle", "scenario",
                 "_cumulative_entropy", "_history", "_owns_log")

    def __init__(
        self,
        config: Config,
        *,
        log: EventLog | None = None,
        brain: Brain | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.rng = RngService(config.run.seed)

        self._owns_log = log is None
        self.log = log if log is not None else EventLog(self._default_log_path(config))

        self.world = World(config, self.rng)
        self.registry = AgentRegistry()
        self.engine = PhysicsEngine(config, self.world, self.registry, self.rng, self.log)
        # Everything reaches the physics through the service -- the tick loop
        # included. There is no private fast path, so the MCP tool surface and
        # the internal loop cannot drift into being different games.
        self.service = SimulationService(
            config, self.world, self.registry, self.engine, self.rng, self.log
        )
        self.brain = brain if brain is not None else self._build_brain(config)
        # Built once, not per turn: compiling a LangGraph state machine for every
        # organism on every tick would cost more than the graph saves.
        self.cycle = self._build_cycle(config)

        self.scenario = build_scenario(config)

        self._cumulative_entropy = 0.0
        self._history: list[TickReport] = []

    @staticmethod
    def _default_log_path(config: Config) -> Path:
        return (
            Path(config.run.output_dir)
            / f"{config.run.name}_seed{config.run.seed}"
            / "events.jsonl"
        )

    def _build_brain(self, config: Config) -> Brain:
        if config.cognition.brain == "mock":
            return MockBrain(config, self.rng)
        if config.cognition.brain == "local":
            # A model served on this machine: no credential, no bill, but real
            # wall-clock time. Imported lazily like the others.
            from btcs.brains.local import LocalBrain

            return LocalBrain(config, self.rng, self.log)
        if config.cognition.brain == "llm":
            # Imported here rather than at module scope so the project stays
            # importable, and the mock brain runnable, without the anthropic SDK.
            from btcs.brains.llm import LLMBrain

            return LLMBrain(config, self.rng, self.log)
        raise ValueError(f"unknown brain: {config.cognition.brain!r}")

    def _build_cycle(self, config: Config):
        """The explicit cognitive graph, when configured. Imported lazily."""
        if not config.cognition.use_graph:
            return None
        from btcs.cognition.graph import CognitiveCycle

        return CognitiveCycle(config, self.brain)

    # -- lifecycle ---------------------------------------------------------

    def run(self, on_tick: Callable[[TickReport], None] | None = None) -> RunResult:
        """Run to the tick limit or to extinction.

        `on_tick` is called after each completed tick, once its invariants have
        passed -- progress reporting should never see a state the audit has not
        yet blessed.
        """
        started = time.perf_counter()
        self._found()

        tick = 0
        extinct = False
        halted: str | None = None
        for tick in range(1, self.config.run.max_ticks + 1):
            try:
                report = self.step(tick)
            except BudgetExceeded as exhausted:
                # The money ran out mid-tick. That tick is abandoned rather than
                # half-counted: its ledger never reached the audit, so admitting
                # it to the history would put an unverified tick in the results.
                halted = "budget_exceeded"
                self.log.emit(tick, EventType.BUDGET_EXCEEDED, None,
                              spent_usd=round(exhausted.spent, 6),
                              ceiling_usd=exhausted.ceiling)
                self.log.flush()
                tick -= 1
                break

            self._history.append(report)
            if on_tick is not None:
                on_tick(report)
            if report.population == 0:
                extinct = True
                if self.config.run.stop_on_extinction:
                    break

        result = RunResult(
            config=self.config,
            ticks_completed=tick,
            final_population=self.registry.population,
            extinct=extinct,
            cumulative_entropy=self._cumulative_entropy,
            log_path=self.log.path,
            history=self._history,
            wall_seconds=time.perf_counter() - started,
            halted_reason=halted,
            spend_usd=self.spend_usd,
        )

        self.log.emit(
            tick, EventType.RUN_FINISHED, None,
            ticks=result.ticks_completed, population=result.final_population,
            extinct=extinct, cumulative_entropy=round(self._cumulative_entropy, 6),
            halted_reason=halted, spend_usd=round(self.spend_usd, 6),
        )
        self.log.flush()
        if self._owns_log:
            self.log.close()
        return result

    def _found(self) -> None:
        self.log.emit(
            0, EventType.RUN_STARTED, None,
            name=self.config.run.name, seed=self.config.run.seed,
            brain=self.brain.name, config=self.config.to_dict(),
        )
        for agent in found_population(self.config, self.registry, self.rng, self.world):
            self.log.emit(
                0, EventType.AGENT_BORN, agent.id,
                generation=agent.generation, y=agent.y, x=agent.x,
                energy=agent.energy, entropy=agent.entropy, phi=agent.phi,
                genome=agent.genome.to_dict(),
            )
        self.log.flush()

    # -- one tick ----------------------------------------------------------

    def _apply_scenario(self, tick: int) -> None:
        """Change the world, before anything in it gets a turn this tick.

        Config changes are rebound across the world, the engine and the service
        **in one place**, because `World` and `PhysicsEngine` must hold the same
        `Config` instance -- two of them means the engine charges one price while
        the world applies another, and the ledger stops balancing in a way that
        is genuinely hard to trace. The engine's constructor asserts it; this is
        the only code that is allowed to move the ground under it.
        """
        for event in self.scenario.due(tick):
            detail = ""
            if event.set:
                self.config = apply_overrides(self.config, event.set)
                self.config.validate()
                self.world._config = self.config
                self.engine._config = self.config
                self.service._config = self.config
                assert self.world.config is self.engine._config
            if event.do:
                detail = ACTIONS[event.do](self.world, self.rng.stream("scenario"))

            self.log.emit(
                tick, EventType.SCENARIO_OVERRIDE, None,
                scenario=self.scenario.name, note=event.note,
                changed=event.set or None, action=event.do, detail=detail or None,
            )

    def step(self, tick: int) -> TickReport:
        physics = self.config.physics
        ledger = EntropyLedger(k_boltzmann=physics.k_boltzmann)

        entropy_before = self.registry.total_entropy()
        self.log.emit(tick, EventType.TICK_STARTED, None,
                      population=self.registry.population)

        self._apply_scenario(tick)

        self.world.step(tick, ledger, self.log)

        self.service.begin_tick(tick, ledger)
        for agent in self._turn_order():
            if not agent.alive:
                continue
            self._take_turn(agent, tick, ledger)

        deaths = self._reap(ledger, tick)

        self._audit(tick, ledger, entropy_before)
        self._cumulative_entropy += ledger.total

        report = TickReport(
            tick=tick,
            population=self.registry.population,
            total_agent_entropy=self.registry.total_entropy(),
            total_agent_energy=self.registry.total_energy(),
            mean_phi=self.registry.mean_phi(),
            ds_agents=ledger.agent_bits,
            ds_env=ledger.env_bits,
            ds_total=ledger.total,
            cumulative_entropy=self._cumulative_entropy,
            deaths=deaths,
            world=self.world.snapshot(),
        )

        self.log.emit(tick, EventType.ENTROPY_LEDGER, None,
                      **{k: round(v, 6) for k, v in ledger.summary().items()},
                      cumulative=round(self._cumulative_entropy, 6))
        self.log.emit(tick, EventType.TICK_FINISHED, None,
                      population=report.population, deaths=deaths,
                      agent_entropy=round(report.total_agent_entropy, 4),
                      agent_energy=round(report.total_agent_energy, 4),
                      **{k: round(v, 4) if isinstance(v, float) else v
                         for k, v in report.world.items()})
        self.log.flush()
        return report

    @property
    def spend_usd(self) -> float:
        """Real money spent so far. Zero for any run on the mock brain."""
        budget = getattr(self.brain, "budget", None)
        return float(getattr(budget, "spent_usd", 0.0))

    def _take_turn(self, agent, tick: int, ledger: EntropyLedger) -> None:
        """One organism's turn, played entirely through the service.

        The organism perceives and acts using its own capability token, exactly
        as it would over MCP. Nothing here can name an agent by id.
        """
        token = self.service.issue_token(agent)
        # Brains that make API calls want to know which tick their log lines
        # belong to; ones that do not simply lack the method.
        if hasattr(self.brain, "set_tick"):
            self.brain.set_tick(tick)

        with self.service.turn(agent):
            percept = self.service.perceive(token)
            bridge = TurnBridge(self.service, token)
            decision = (
                self.cycle.run(percept, agent, bridge, tick)
                if self.cycle is not None
                else self.brain.decide(percept, agent, bridge)
            )

            # Order matters: thinking is charged before acting, so an organism
            # cannot spend energy it only has because of the action it chose.
            self.service.charge_cognition(agent, decision.tokens)

            try:
                self._dispatch(token, decision.action)
            except ToolError as refusal:
                # A physically impossible request costs the turn but not the run.
                self.log.emit(tick, EventType.BASAL, agent.id,
                              rejected=describe(decision.action),
                              reason=refusal.message, code=str(refusal.code))

            self.service.charge_basal(agent)

        # Charged after the turn closes, because the surprise is scored against
        # the episode the turn just produced. Inside the bracketed tick, so the
        # ledger audit sees it.
        self.engine.charge_surprise(agent, ledger, tick)
        self.engine.resolve_degradation(agent, tick)
        self._integrate(agent, tick)
        agent.ticks_survived += 1

    def _integrate(self, agent, tick: int) -> None:
        """Corrupt the record, then re-measure what is left of its integration.

        Order matters and is the whole of the death spiral. Disorder eats
        episodes out of the organism's experience; the measure then reads the
        damaged graph, so Phi falls; a lower Phi shields less, so the next tick
        eats more. Nothing here implements a spiral -- it is what happens when
        the damage and the measure act on the same structure.
        """
        if not self.config.phi.enabled:
            return

        graph = self.service.graph_of(agent.id)
        from btcs.world.agents import Degradation

        if agent.degradation >= Degradation.CONFUSED:
            graph.corrupt(
                self.rng.agent_stream(agent.id),
                self.engine.corruption_probability(agent),
            )
        self.engine.update_phi(agent, graph, tick)

    def _dispatch(self, token: str, action) -> dict:
        """Route a brain's chosen action to the matching tool.

        This is the same set of calls an LLM-driven organism will make over MCP
        in Phase 3; the mock brain simply chooses them deterministically.
        """
        match action:
            case Rest():
                return self.service.rest(token)
            case Move(dy=dy, dx=dx):
                return self.service.perform_work(token, dy, dx)
            case Ingest(joules=joules):
                return self.service.ingest_energy(token, joules)
            case Dissipate(bits=bits):
                return self.service.dissipate_entropy(token, bits)
            case Replicate():
                return self.service.replicate(token)
        raise ToolError(ErrorCode.INVALID_ARGUMENT, f"unhandled action: {action!r}")

    def _turn_order(self):
        """Randomised but reproducible acting order.

        Sorted order would give low-numbered organisms a permanent first-mover
        advantage on contested nodes, and that advantage would show up in the
        results as if it were selection.
        """
        agents = self.registry.living()
        if len(agents) <= 1:
            return agents
        order = self.rng.stream("turn_order").permutation(len(agents))
        return [agents[i] for i in order]

    def _reap(self, ledger: EntropyLedger, tick: int) -> int:
        deaths = 0
        for agent in self.registry.living():
            if self.engine.check_death(agent, ledger, tick):
                self.brain.on_death(agent)
                # Revoke first: a token outliving its organism would be a
                # credential for a dead entity.
                self.service.revoke_token(agent.id)
                self.registry.remove(agent.id)
                deaths += 1
        return deaths

    def _audit(self, tick: int, ledger: EntropyLedger, entropy_before: float) -> None:
        """Both invariants, checked before the tick is allowed to count."""
        physics = self.config.physics
        if not physics.assert_second_law:
            return

        tolerance = physics.second_law_tolerance

        measured = self.registry.total_entropy() - entropy_before
        drift = abs(measured - ledger.agent_bits)
        # Scale the tolerance with the magnitude involved: these are sums over a
        # whole population, so a fixed absolute epsilon would false-positive on
        # large runs purely through float accumulation.
        allowed = max(tolerance, 1e-9 * max(1.0, abs(measured), abs(ledger.agent_bits)))
        if drift > allowed:
            raise LedgerInconsistency(
                f"tick {tick}: organisms' entropy changed by {measured:.9g} bits but the "
                f"ledger recorded {ledger.agent_bits:.9g} (drift {drift:.3g}). "
                "Some code path is mutating agent.entropy without filing an entry."
            )

        if not ledger.is_valid(tolerance):
            self.log.emit(tick, EventType.SECOND_LAW_VIOLATION, None, **ledger.summary())
            self.log.flush()
            raise SecondLawViolation(tick, ledger)


def run_simulation(config: Config, **kwargs: Any) -> RunResult:
    """Convenience entry point."""
    return SimulationRunner(config, **kwargs).run()
