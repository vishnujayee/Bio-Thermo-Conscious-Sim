"""Deterministic rule-based brain. Costs nothing, decides instantly.

This exists so that the entire simulation -- physics, selection, death, the
whole apparatus -- can be built and validated before a single API call is made.
It is not a placeholder to be thrown away: it stays permanently as the control
condition for every experiment, because "did the LLM actually do better than a
handful of thresholds?" is the question the project ultimately has to answer.

The policy reads its thresholds from the organism's genome, so selection acts on
this brain exactly as it will act on the LLM one. A wasteful parameter set must
demonstrably lose to a frugal one here before we spend money finding out whether
a wasteful *prompt* does.
"""

from __future__ import annotations

from btcs.config import Config
from btcs.rng import RngService
from btcs.world.agents import Agent
from btcs.world.types import (
    Decision,
    Dissipate,
    Ingest,
    Move,
    Percept,
    Replicate,
    Rest,
)

# Token counts a real model would plausibly spend on the same percept. They are
# invented, but they are invented *consistently*: cost scales with how much the
# organism is looking at, so a greedy wide-sensing strategy is charged more
# entropy than a narrow one, exactly as it would be through the API.
_TOKENS_BASE = 40
_TOKENS_PER_NODE = 12
_TOKENS_DELIBERATION = 25


class MockBrain:
    """A frugal, legible survival policy.

    Priority order, highest first:

      1. Erase entropy when it has passed the genome's threshold and the
         organism can afford the Landauer bill.
      2. Eat, if standing on something edible.
      3. Move toward the best visible node, discounted by distance and by how
         hot the destination is.
      4. Explore.

    The heat discount in step 3 is what makes this policy non-trivial. A rich
    node in a hot cell is a poor meal, because the Carnot bound is computed
    against the local temperature -- so the obvious greedy strategy of always
    running at the biggest number is measurably worse than this one.
    """

    __slots__ = ("_config", "_rng")

    def __init__(self, config: Config, rng: RngService) -> None:
        self._config = config
        self._rng = rng

    @property
    def name(self) -> str:
        return "mock"

    def on_death(self, agent: Agent) -> None:  # noqa: D102 - nothing to release
        return None

    def decide(self, percept: Percept, agent: Agent, bridge=None) -> Decision:
        # `bridge` is the deliberation handle an LLM brain uses mid-turn. This
        # policy decides from the percept alone, so it is accepted and ignored.
        policy = agent.genome.policy
        tokens = _TOKENS_BASE + _TOKENS_PER_NODE * len(percept.nodes)

        # 1. Buy back order, if it is worth buying and affordable.
        if percept.entropy > policy.dissipate_at:
            excess = percept.entropy - policy.dissipate_at
            bits = min(
                excess * policy.dissipate_fraction,
                self._config.agent.max_dissipate_bits,
            )
            if bits > 0.0:
                price = bits * percept.local_temperature * 0.693 / (
                    self._config.physics.erase_efficiency
                )
                # Only erase if it leaves enough to keep foraging afterwards.
                if price < percept.energy * 0.5:
                    return Decision(
                        Dissipate(bits),
                        tokens + _TOKENS_DELIBERATION,
                        note="entropy above threshold",
                    )

        # 2. Spend the surplus on a child, if this lineage thinks it is rich
        # enough. Placed above eating deliberately: an organism that is already
        # wealthy gains little from one more meal and everything from an heir,
        # and where exactly that line falls is the heritable trait that Phase 5
        # exists to put under selection.
        #
        # The threshold is the lineage's own, floored at what the world will
        # actually accept: a mutant whose `replicate_at` has drifted below the
        # world's minimum would otherwise spend every turn asking for a child it
        # cannot afford, and starve while being refused. Knowing whether you are
        # rich enough to breed is not privileged information.
        breeding_energy = max(
            policy.replicate_at, self._config.selection.replication_energy_threshold
        )
        if percept.energy >= breeding_energy:
            return Decision(Replicate(), tokens, note="surplus to a child")

        # 3. Eat what is underfoot.
        if percept.on_node:
            return Decision(
                Ingest(self._config.physics.ingest_rate_limit),
                tokens,
                note="harvesting",
            )

        # 4. Move toward the most attractive node.
        target = self._best_node(percept, policy)
        if target is not None:
            step = _unit_step(target.dy, target.dx)
            if step != (0, 0):
                return Decision(
                    Move(*step), tokens, note=f"approaching node {target.index}"
                )

        # 5. Nothing worth pursuing. Wander, or rest and conserve.
        stream = self._rng.agent_stream(agent.id)
        if stream.random() < policy.explore_bias:
            dy = int(stream.integers(-1, 2))
            dx = int(stream.integers(-1, 2))
            if (dy, dx) != (0, 0):
                return Decision(Move(dy, dx), tokens, note="exploring")

        return Decision(Rest(), tokens // 2, note="conserving")

    def _best_node(self, percept: Percept, policy):
        """Score visible nodes by what they are actually worth reaching.

        Extractable energy already carries the Carnot factor, so this only has
        to add the cost of getting there and a penalty for destinations hotter
        than where the organism currently stands.
        """
        best = None
        best_score = 0.0
        hunger = 1.0 if percept.energy < policy.hunger_at else 0.5

        for node in percept.nodes:
            if node.extractable <= 0.0:
                continue
            distance = max(1.0, node.distance)
            heat_penalty = 1.0 + policy.heat_aversion * max(
                0.0, node.temperature - percept.local_temperature
            ) / max(percept.local_temperature, 0.05)
            score = (node.extractable * policy.greed * hunger) / (distance * heat_penalty)
            if score > best_score:
                best, best_score = node, score
        return best


def _unit_step(dy: int, dx: int) -> tuple[int, int]:
    """Reduce an offset to a single step, diagonals included."""
    return (0 if dy == 0 else (1 if dy > 0 else -1),
            0 if dx == 0 else (1 if dx > 0 else -1))
