"""Reading a run back: metrics, lineages, replay.

Every result in this project so far -- the selection differentials, the Phi
sweeps, the calibration decisions -- was computed by a throwaway script reading
the JSONL log. That was the right way to find them and the wrong way to keep
them: a number nobody can recompute is a number nobody can check.

So this module is the analysis, in the repository, tested.

**The log is the authoritative record of a run**, not the terminal output and not
anything held in memory. Everything here reads it and nothing here needs the
simulation to still be running -- which is what makes a result from six months
ago as checkable as one from this morning.

Replay works because two things are true at once: every position change is
logged (birth gives a starting cell, every move gives the new one), and the
world's initial layout is a deterministic function of the seed, which the log
carries. So the whole history can be reconstructed exactly, without ever having
stored a grid.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from btcs.events import read_events


@dataclass(slots=True)
class Organism:
    """One life, reconstructed from the events it generated."""

    id: str
    generation: int
    parent: str | None = None
    born: int = 0
    died: int | None = None
    cause: str | None = None
    lifespan: int = 0
    offspring: int = 0
    total_tokens: int = 0
    strategy: str = ""
    policy: dict[str, float] = field(default_factory=dict)
    phi: list[float] = field(default_factory=list)
    surprise: list[float] = field(default_factory=list)

    @property
    def bred(self) -> bool:
        return self.offspring > 0

    @property
    def mean_phi(self) -> float:
        return statistics.mean(self.phi) if self.phi else 0.0


@dataclass(slots=True)
class Tick:
    """One tick's worth of world-level numbers."""

    tick: int
    population: int
    deaths: int
    agent_energy: float
    agent_entropy: float
    extractable_energy: float
    field_energy: float
    mean_temperature: float
    max_temperature: float
    nodes: int
    ds_total: float = 0.0
    ds_agents: float = 0.0
    ds_env: float = 0.0
    cumulative: float = 0.0


class Run:
    """A finished run, loaded from its event log."""

    __slots__ = ("path", "name", "seed", "brain", "config", "ticks", "organisms",
                 "actions", "refusals", "deaths_by_cause", "scenario",
                 "scenario_events", "extinct", "spend_usd", "_moves")

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.name = "unknown"
        self.seed = 0
        self.brain = "unknown"
        self.config: dict[str, Any] = {}
        self.ticks: list[Tick] = []
        self.organisms: dict[str, Organism] = {}
        self.actions: Counter[str] = Counter()
        self.refusals: Counter[str] = Counter()
        self.deaths_by_cause: Counter[str] = Counter()
        self.scenario = "none"
        self.scenario_events: list[dict[str, Any]] = []
        self.extinct = False
        self.spend_usd = 0.0
        # agent id -> [(tick, y, x), ...], in order. The raw material for replay.
        self._moves: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        self._load()

    # -- loading -----------------------------------------------------------

    def _load(self) -> None:
        ledgers: dict[int, dict[str, float]] = {}
        finished: dict[int, dict[str, Any]] = {}

        for event in read_events(self.path):
            kind = event["type"]
            tick = event.get("tick", 0)

            match kind:
                case "run_started":
                    self.name = event.get("name", self.name)
                    self.seed = event.get("seed", 0)
                    self.brain = event.get("brain", "unknown")
                    self.config = event.get("config") or {}
                    self.scenario = (
                        self.config.get("scenario", {}).get("name", "none")
                        if isinstance(self.config.get("scenario"), dict) else "none"
                    )
                case "run_finished":
                    self.extinct = bool(event.get("extinct"))
                    self.spend_usd = float(event.get("spend_usd") or 0.0)
                case "tick_finished":
                    finished[tick] = event
                case "entropy_ledger":
                    ledgers[tick] = event
                case "scenario_override":
                    self.scenario_events.append(event)
                case "agent_born":
                    self._born(event, tick)
                case "agent_died":
                    self._died(event, tick)
                case "replication":
                    parent = self.organisms.get(event["agent"])
                    if parent is not None:
                        parent.offspring += 1
                case "work":
                    self._moves[event["agent"]].append(
                        (tick, int(event["y"]), int(event["x"]))
                    )
                    self.actions["move"] += 1
                case "ingest" | "dissipate":
                    self.actions[kind] += 1
                case "phi_changed":
                    organism = self.organisms.get(event["agent"])
                    if organism is not None:
                        organism.phi.append(float(event["phi"]))
                case "surprise":
                    organism = self.organisms.get(event["agent"])
                    if organism is not None:
                        organism.surprise.append(float(event["surprise"]))
                case "basal" if event.get("code"):
                    self.refusals[str(event.get("reason", "?"))[:60]] += 1

        self._assemble(finished, ledgers)

    def _born(self, event: dict[str, Any], tick: int) -> None:
        genome = event.get("genome") or {}
        organism = Organism(
            id=event["agent"],
            generation=int(event.get("generation", 1)),
            parent=event.get("parent"),
            born=tick,
            strategy=genome.get("strategy", ""),
            policy=dict(genome.get("policy") or {}),
        )
        self.organisms[organism.id] = organism
        self._moves[organism.id].append((tick, int(event["y"]), int(event["x"])))

    def _died(self, event: dict[str, Any], tick: int) -> None:
        organism = self.organisms.get(event["agent"])
        if organism is None:
            return
        organism.died = tick
        organism.cause = event.get("cause")
        organism.lifespan = int(event.get("ticks_survived", 0))
        organism.total_tokens = int(event.get("total_tokens", 0))
        self.deaths_by_cause[str(event.get("cause", "?"))] += 1

    def _assemble(self, finished, ledgers) -> None:
        for tick in sorted(finished):
            snapshot = finished[tick]
            ledger = ledgers.get(tick, {})
            self.ticks.append(Tick(
                tick=tick,
                population=int(snapshot.get("population", 0)),
                deaths=int(snapshot.get("deaths", 0)),
                agent_energy=float(snapshot.get("agent_energy", 0.0)),
                agent_entropy=float(snapshot.get("agent_entropy", 0.0)),
                extractable_energy=float(snapshot.get("extractable_energy", 0.0)),
                field_energy=float(snapshot.get("field_energy", 0.0)),
                mean_temperature=float(snapshot.get("mean_temperature", 0.0)),
                max_temperature=float(snapshot.get("max_temperature", 0.0)),
                nodes=int(snapshot.get("nodes", 0)),
                ds_total=float(ledger.get("dS_total", 0.0)),
                ds_agents=float(ledger.get("dS_agents", 0.0)),
                ds_env=float(ledger.get("dS_env", 0.0)),
                cumulative=float(ledger.get("cumulative", 0.0)),
            ))

    # -- summary -----------------------------------------------------------

    @property
    def duration(self) -> int:
        return self.ticks[-1].tick if self.ticks else 0

    @property
    def final_population(self) -> int:
        return self.ticks[-1].population if self.ticks else 0

    @property
    def mean_population(self) -> float:
        return statistics.mean(t.population for t in self.ticks) if self.ticks else 0.0

    @property
    def peak_population(self) -> int:
        return max((t.population for t in self.ticks), default=0)

    @property
    def births(self) -> int:
        return sum(o.offspring for o in self.organisms.values())

    @property
    def generations(self) -> int:
        return max((o.generation for o in self.organisms.values()), default=0)

    @property
    def founders(self) -> list[Organism]:
        return [o for o in self.organisms.values() if o.parent is None]

    def lifespans(self) -> list[int]:
        return [o.lifespan for o in self.organisms.values() if o.died is not None]

    def mean_lifespan(self) -> float:
        lives = self.lifespans()
        return statistics.mean(lives) if lives else 0.0

    # -- the instrument ----------------------------------------------------

    def selection_differential(self, trait: str) -> tuple[float, float, int, int]:
        """Compare a trait between organisms that reproduced and those that did not.

        The instrument behind every evolutionary claim this project makes, and
        behind the one it retracted. Returns (bred, never bred, n_bred, n_never).

        It says which organisms *did* reproduce, which is not the same as showing
        a population mean that moves and stays moved. Do not let it become a
        claim about adaptation on its own.
        """
        bred, never = [], []
        for organism in self.organisms.values():
            value = self._trait(organism, trait)
            if value is None:
                continue
            (bred if organism.bred else never).append(value)
        return (
            statistics.mean(bred) if bred else math.nan,
            statistics.mean(never) if never else math.nan,
            len(bred), len(never),
        )

    @staticmethod
    def _trait(organism: Organism, trait: str) -> float | None:
        if trait == "phi":
            return organism.mean_phi if organism.phi else None
        if trait == "surprise":
            return statistics.mean(organism.surprise) if organism.surprise else None
        if trait == "lifespan":
            return float(organism.lifespan) if organism.died is not None else None
        return organism.policy.get(trait)

    def trait_by_generation(self, trait: str) -> dict[int, float]:
        """Mean of a heritable trait per generation. Drift and selection, together."""
        buckets: dict[int, list[float]] = defaultdict(list)
        for organism in self.organisms.values():
            value = self._trait(organism, trait)
            if value is not None:
                buckets[organism.generation].append(value)
        return {gen: statistics.mean(values) for gen, values in sorted(buckets.items())}

    def lineage(self, root: str | None = None) -> dict[str, list[str]]:
        """Parent -> children, for drawing a family tree."""
        children: dict[str, list[str]] = defaultdict(list)
        for organism in self.organisms.values():
            if organism.parent:
                children[organism.parent].append(organism.id)
        if root is None:
            return dict(children)
        reachable, stack = {}, [root]
        while stack:
            current = stack.pop()
            kids = sorted(children.get(current, []))
            reachable[current] = kids
            stack.extend(kids)
        return reachable

    def descendants(self, agent_id: str) -> int:
        seen, stack = set(), [agent_id]
        tree = self.lineage()
        while stack:
            for child in tree.get(stack.pop(), []):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return len(seen)

    # -- replay ------------------------------------------------------------

    def positions_at(self, tick: int) -> dict[str, tuple[int, int]]:
        """Where every living organism stood at the end of a tick.

        Reconstructed rather than stored: birth records a starting cell and every
        move records the new one, so the last entry at or before `tick` is the
        position. Nothing needs to have been snapshotted.
        """
        placed: dict[str, tuple[int, int]] = {}
        for agent_id, moves in self._moves.items():
            organism = self.organisms.get(agent_id)
            if organism is None or organism.born > tick:
                continue
            if organism.died is not None and organism.died < tick:
                continue
            latest = [m for m in moves if m[0] <= tick]
            if latest:
                _, y, x = latest[-1]
                placed[agent_id] = (y, x)
        return placed

    def replay_world(self, tick: int):
        """Rebuild the world as it stood, by re-running its deterministic parts.

        The initial node layout is a function of the seed, which the log carries,
        so it never had to be written down. Node births and deaths after that are
        logged and applied in order.
        """
        from btcs.config import Config
        from btcs.rng import RngService
        from btcs.world.grid import World

        config = Config.from_dict(self.config)
        world = World(config, RngService(config.run.seed))

        spawned, removed = [], []
        for event in read_events(self.path):
            if event.get("tick", 0) > tick:
                break
            if event["type"] == "node_spawned":
                spawned.append((event["y"], event["x"], event["energy"],
                                event["temperature"]))
            elif event["type"] == "node_exhausted":
                removed.append((event["y"], event["x"]))
        return world, spawned, removed


def load(path: str | Path) -> Run:
    return Run(path)


def compare(runs: list[Run], trait: str = "phi") -> str:
    """Line-per-run comparison of the selection differential on one trait."""
    lines = [f"{'run':>22} {'ticks':>6} {'pop':>5} {'births':>7} {'gens':>5} "
             f"{'bred':>8} {'never':>8} {'diff':>8}"]
    for run in runs:
        bred, never, _, _ = run.selection_differential(trait)
        lines.append(
            f"{run.name[:22]:>22} {run.duration:>6} {run.final_population:>5} "
            f"{run.births:>7} {run.generations:>5} {bred:>8.3f} {never:>8.3f} "
            f"{bred - never:>+8.3f}"
        )
    return "\n".join(lines)
