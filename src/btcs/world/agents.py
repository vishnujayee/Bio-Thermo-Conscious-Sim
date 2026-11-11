"""Organisms and the registry that holds them.

An organism is deliberately thin: it is a bundle of thermodynamic state plus a
genome. It has no methods that change its own energy or entropy -- every such
mutation goes through `PhysicsEngine`, which is the only component that can file
entropy entries. If an organism could adjust its own books, the second-law
assertion would be checking nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from btcs.config import Config, PolicyParams
from btcs.world.types import LastAction

LN2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class Genome:
    """Heritable strategy.

    Two representations of the same thing. `strategy` is prose the LLM brain
    reads; `policy` is numbers the mock brain reads. Both mutate at replication,
    which lets selection be validated end-to-end on the free brain before any
    money is spent on the expensive one.

    Executable policy code is Phase 5b and is deliberately absent here.
    """

    strategy: str
    policy: PolicyParams = field(default_factory=PolicyParams)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "policy": {
                "dissipate_at": self.policy.dissipate_at,
                "dissipate_fraction": self.policy.dissipate_fraction,
                "hunger_at": self.policy.hunger_at,
                "greed": self.policy.greed,
                "heat_aversion": self.policy.heat_aversion,
                "explore_bias": self.policy.explore_bias,
                "replicate_at": self.policy.replicate_at,
            },
        }


class Degradation:
    """Rungs of the ladder from lucid to erased (construction.md 3.3)."""

    LUCID = 0
    NOISY = 1      # S > s_warn      -- perception acquires noise
    CONFUSED = 2   # S > s_confuse   -- information is lost outright
    DEMENTED = 3   # S > s_dement    -- context fills with junk, cognition costs more

    NAMES = {LUCID: "lucid", NOISY: "noisy", CONFUSED: "confused", DEMENTED: "demented"}


@dataclass(slots=True)
class Agent:
    """One digital organism."""

    id: str
    y: int
    x: int
    energy: float
    entropy: float
    phi: float
    genome: Genome
    generation: int = 1
    parent_id: str | None = None
    birth_tick: int = 0
    alive: bool = True
    degradation: int = Degradation.LUCID

    # Lifetime counters, used for analysis and for selection diagnostics.
    total_tokens: int = 0
    total_ingested: float = 0.0
    total_dissipated_bits: float = 0.0
    ticks_survived: int = 0
    offspring: int = 0
    # Bits of surprise the last moment carried, given this organism's memory.
    last_surprise: float = 0.0
    # The outcome of the most recent physical act. Set by the service, which is
    # the one place every act -- accepted or refused -- passes through.
    last_action: LastAction | None = None
    death_tick: int | None = None
    death_cause: str | None = None

    def free_energy(self, temperature: float) -> float:
        """Helmholtz free energy F = E - T*S.

        Entropy is stored in bits, so it converts to thermodynamic units with a
        factor of `k_B * ln2`. This single expression is what couples heat to
        mortality: the same organism that survives comfortably in a cold cell is
        already dead in a hot one, because temperature multiplies its accumulated
        disorder.
        """
        return self.energy - temperature * self.entropy * LN2

    def is_viable(self, temperature: float) -> bool:
        return self.free_energy(temperature) > 0.0

    def vitals(self) -> dict[str, Any]:
        return {
            "energy": self.energy,
            "entropy": self.entropy,
            "phi": self.phi,
            "generation": self.generation,
            "degradation": Degradation.NAMES[self.degradation],
            "y": self.y,
            "x": self.x,
        }


class AgentRegistry:
    """Owns the living population and assigns identities.

    Iteration order is always sorted by id, so nothing downstream can
    accidentally depend on dictionary insertion order and break reproducibility.
    """

    __slots__ = ("_agents", "_serial")

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._serial = 0

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._agents

    def __getitem__(self, agent_id: str) -> Agent:
        return self._agents[agent_id]

    def get(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def next_id(self, generation: int) -> str:
        self._serial += 1
        return f"A{self._serial:04d}g{generation}"

    def add(self, agent: Agent) -> Agent:
        if agent.id in self._agents:
            raise ValueError(f"duplicate agent id: {agent.id}")
        self._agents[agent.id] = agent
        return agent

    def remove(self, agent_id: str) -> Agent:
        return self._agents.pop(agent_id)

    def all(self) -> list[Agent]:
        return [self._agents[key] for key in sorted(self._agents)]

    def living(self) -> list[Agent]:
        return [agent for agent in self.all() if agent.alive]

    @property
    def population(self) -> int:
        return sum(1 for agent in self._agents.values() if agent.alive)

    # -- aggregates -------------------------------------------------------

    def total_entropy(self) -> float:
        return sum(agent.entropy for agent in self._agents.values() if agent.alive)

    def total_energy(self) -> float:
        return sum(agent.energy for agent in self._agents.values() if agent.alive)

    def mean_phi(self) -> float:
        living = self.living()
        return sum(agent.phi for agent in living) / len(living) if living else 0.0

    def max_generation(self) -> int:
        living = self.living()
        return max((agent.generation for agent in living), default=0)


def found_population(config: Config, registry: AgentRegistry, rng: Any, world: Any) -> list[Agent]:
    """Seed the world with the founding generation.

    All founders share the ancestor genome; divergence is the job of mutation at
    replication, not of initialisation. Starting them identical means any later
    spread in the population is attributable to selection rather than to a lucky
    draw at t=0.
    """
    stream = rng.stream("founding")
    agent_cfg = config.agent
    genome = Genome(strategy=config.selection.ancestor_strategy, policy=agent_cfg.policy)

    founders: list[Agent] = []
    for _ in range(agent_cfg.initial_population):
        agent = Agent(
            id=registry.next_id(1),
            y=int(stream.integers(0, world.height)),
            x=int(stream.integers(0, world.width)),
            energy=agent_cfg.initial_energy,
            entropy=agent_cfg.initial_entropy,
            phi=agent_cfg.initial_phi,
            genome=replace(genome),
            generation=1,
            parent_id=None,
            birth_tick=0,
        )
        founders.append(registry.add(agent))
    return founders


# -- mutation -------------------------------------------------------------


def mutate(genome: Genome, stream: Any, config: Config) -> Genome:
    """Produce a mutated copy of a genome. Both halves drift.

    Prose mutates by **point mutations over words**: each word may be dropped,
    duplicated, or swapped with its neighbour. Nothing is ever written in.

    That is a deliberate choice on two counts. Scientifically, it is drift with
    no author -- if a lineage's strategy improves, no one wrote the improvement,
    which is the only way an evolutionary claim here can mean anything. (The
    alternative, asking a model to "rewrite this strategy slightly", would make
    every adaptation partly the model's design and partly ours.)

    And for safety, it closes S2 by construction: mutation is a closed operation
    over the words the ancestor already had, so no new token -- and therefore no
    injected instruction -- can appear in a genome that did not start with it.
    """
    return Genome(
        strategy=_mutate_text(genome.strategy, stream, config.selection),
        policy=_mutate_policy(genome.policy, stream, config.selection.mutation_rate),
    )


def _mutate_text(text: str, stream: Any, selection: Any) -> str:
    words = text.split()
    if not words:
        return text

    rate = selection.mutation_rate
    mutated: list[str] = []
    index = 0
    while index < len(words):
        word = words[index]
        if stream.random() < rate:
            roll = stream.random()
            if roll < 0.34:
                index += 1               # deletion
                continue
            if roll < 0.67:
                mutated.append(word)     # duplication
                mutated.append(word)
                index += 1
                continue
            if index + 1 < len(words):   # transposition
                mutated.append(words[index + 1])
                mutated.append(word)
                index += 2
                continue
        mutated.append(word)
        index += 1

    # A lineage that deletes its way to silence keeps the parent's text: an
    # empty genome is not a strategy, and the prompt would be malformed.
    if not mutated:
        return text

    drifted = " ".join(mutated)
    if len(drifted) > selection.genome_max_chars:
        # Truncate on a word boundary rather than mid-word, so the text stays
        # readable prose. Duplication is what makes this reachable at all.
        drifted = drifted[: selection.genome_max_chars].rsplit(" ", 1)[0]
    return drifted or text


def _mutate_policy(policy: PolicyParams, stream: Any, rate: float) -> PolicyParams:
    """Multiplicative gaussian drift, clamped to ranges the engine can use.

    Clamping is not a value judgement about which strategies are allowed -- it
    keeps a mutant from asking the engine for something meaningless, such as a
    negative appetite, which would be a crash rather than a phenotype.
    """
    def drift(value: float, low: float, high: float) -> float:
        return float(min(high, max(low, value * (1.0 + stream.normal(0.0, rate)))))

    return PolicyParams(
        dissipate_at=drift(policy.dissipate_at, 1.0, 500.0),
        dissipate_fraction=drift(policy.dissipate_fraction, 0.01, 1.0),
        hunger_at=drift(policy.hunger_at, 1.0, 1000.0),
        greed=drift(policy.greed, 0.0, 10.0),
        heat_aversion=drift(policy.heat_aversion, 0.0, 10.0),
        explore_bias=drift(policy.explore_bias, 0.0, 1.0),
        replicate_at=drift(policy.replicate_at, 1.0, 10_000.0),
    )
