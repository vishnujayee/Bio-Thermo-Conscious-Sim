"""What a live run would cost, without making a single API call.

The estimator is itself a brain. It wraps the mock policy, builds the *real*
prompts for every organism on every tick, measures them, and models the cache --
so the figure it produces comes from the actual prompt assembly path rather than
from a guess about it. If the prompt gets longer, the estimate moves.

Token counts are approximated locally rather than through `count_tokens`,
because the point of this tool is to answer "what will this cost" on a machine
with no credentials at all. The approximation is deliberately pessimistic: it is
better to be told a run costs more than it does than to discover the opposite
afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from btcs.brains.mock import MockBrain
from btcs.cognition.pricing import Usage, cost_usd
from btcs.cognition.prompt import TOOLS, prefix_fingerprint, system_blocks, user_message
from btcs.config import Config
from btcs.rng import RngService
from btcs.world.agents import Agent
from btcs.world.types import Decision, Percept

# English prose runs a little under four characters per token. Rounding the
# divisor down biases every estimate upward, which is the safe direction.
CHARS_PER_TOKEN = 3.5

# What a model typically emits for a single tool call plus a sentence of
# reasoning. Adaptive thinking makes this vary a lot; this is a working figure,
# not a measurement, and it is the estimate's largest source of error.
ASSUMED_OUTPUT_TOKENS = 180


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / CHARS_PER_TOKEN) + 1)


@dataclass(slots=True)
class Estimate:
    """Projected cost of a run, broken down enough to argue with."""

    ticks: int = 0
    turns: int = 0
    usage_by_model: dict[str, Usage] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)

    @property
    def total_usd(self) -> float:
        return sum(self.cost_by_model.values())

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for usage in self.usage_by_model.values():
            total = total + usage
        return total

    def add(self, model: str, usage: Usage) -> None:
        self.usage_by_model[model] = self.usage_by_model.get(model, Usage()) + usage
        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + cost_usd(usage, model)

    def report(self) -> str:
        lines = [
            f"Projected cost: ${self.total_usd:.2f}",
            f"  {self.turns:,} turns over {self.ticks:,} ticks "
            f"(one request each; deliberation would add more)",
        ]
        total = self.total_usage
        lines.append(
            f"  tokens: {total.total_input:,} in "
            f"({total.cache_read_input_tokens:,} from cache, "
            f"{total.cache_hit_rate:.0%} hit rate), {total.output_tokens:,} out"
        )
        for model in sorted(self.cost_by_model, key=lambda m: -self.cost_by_model[m]):
            usage = self.usage_by_model[model]
            share = self.cost_by_model[model]
            lines.append(
                f"  {model:<20} ${share:>8.2f}  "
                f"({usage.total_input:,} in / {usage.output_tokens:,} out)"
            )
        if self.turns:
            lines.append(f"  cost per turn: ${self.total_usd / self.turns:.5f}")
        return "\n".join(lines)


class EstimatingBrain:
    """Decides like the mock brain, but prices every turn as if it were real.

    Cache modelling: the first time a given prefix fingerprint is seen it is
    charged as a cache *write*; every later turn on that prefix is charged as a
    *read*. That is what makes replication show up as a saving -- clones share a
    genome, so they share a fingerprint, so a growing lineage reads a cache it
    only paid to write once.
    """

    __slots__ = ("_config", "_mock", "estimate", "_seen_prefixes")

    def __init__(self, config: Config, rng: RngService) -> None:
        self._config = config
        self._mock = MockBrain(config, rng)
        self.estimate = Estimate()
        self._seen_prefixes: set[str] = set()

    @property
    def name(self) -> str:
        return "estimate"

    def on_death(self, agent: Agent) -> None:
        return None

    def set_tick(self, tick: int) -> None:
        self.estimate.ticks = max(self.estimate.ticks, tick)

    def decide(self, percept: Percept, agent: Agent, bridge: Any = None) -> Decision:
        tier = self._config.cognition.tier_for(agent.phi)
        memory = bridge.memory() if bridge is not None else {}

        prefix_text = "".join(b["text"] for b in system_blocks(agent, self._config))
        prefix_text += "".join(
            t["name"] + t["description"] + str(t["input_schema"]) for t in TOOLS
        )
        volatile = user_message(percept, memory, self._config)["content"][0]["text"]

        prefix_tokens = approx_tokens(prefix_text)
        volatile_tokens = approx_tokens(volatile)

        fingerprint = prefix_fingerprint(agent, self._config)
        first_time = fingerprint not in self._seen_prefixes
        self._seen_prefixes.add(fingerprint)

        usage = Usage(
            input_tokens=volatile_tokens,
            cache_creation_input_tokens=prefix_tokens if first_time else 0,
            cache_read_input_tokens=0 if first_time else prefix_tokens,
            output_tokens=ASSUMED_OUTPUT_TOKENS,
        )
        self.estimate.add(tier.model, usage)
        self.estimate.turns += 1

        # Decide exactly as the mock brain would, so the simulated population
        # behaves the same way it will in the real run being priced.
        return self._mock.decide(percept, agent, bridge)


def estimate_run(config: Config) -> tuple[Estimate, Any]:
    """Run the simulation with the estimating brain. Costs nothing."""
    from btcs.events import EventLog
    from btcs.runner import SimulationRunner

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as scratch:
        log_path = Path(scratch) / "estimate.jsonl"
        with EventLog(log_path) as log:
            rng = RngService(config.run.seed)
            brain = EstimatingBrain(config, rng)
            runner = SimulationRunner(config, log=log, brain=brain)
            result = runner.run()
    brain.estimate.ticks = result.ticks_completed
    return brain.estimate, result
