"""Spend tracking and the kill switch.

An unattended simulation making API calls in a loop is the failure mode that
costs real money, so the ceiling is enforced in the one place every request must
pass through, and crossing it stops the run rather than logging a warning.

The ledger here is deliberately shaped like the entropy ledger: per-model and
per-organism attribution, so "where did the money go" is answerable from the run
log without re-deriving it from raw usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from btcs.cognition.pricing import ModelPrice, Usage, cost_usd


class BudgetExceeded(RuntimeError):
    """The run's spend ceiling was crossed.

    Raised rather than logged: a budget that can be exceeded quietly is not a
    budget. The runner catches it, checkpoints, and stops cleanly.
    """

    def __init__(self, spent: float, ceiling: float) -> None:
        super().__init__(
            f"budget exhausted: ${spent:.4f} spent against a ${ceiling:.2f} ceiling"
        )
        self.spent = spent
        self.ceiling = ceiling


@dataclass(slots=True)
class BudgetGuard:
    """Accumulates real spend and refuses to let a run exceed its ceiling."""

    ceiling_usd: float
    halt_on_exceed: bool = True
    price_overrides: dict[str, ModelPrice] | None = None

    spent_usd: float = 0.0
    requests: int = 0
    usage: Usage = field(default_factory=Usage)
    by_model: dict[str, float] = field(default_factory=dict)
    by_agent: dict[str, float] = field(default_factory=dict)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.spent_usd)

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.ceiling_usd

    def would_exceed(self, estimate_usd: float) -> bool:
        """Check before spending, for callers that can estimate a request."""
        return self.spent_usd + estimate_usd > self.ceiling_usd

    def record(self, usage: Usage, model: str, agent_id: str | None = None) -> float:
        """Bill one request. Returns its cost; raises once the ceiling is past.

        The charge lands before the check, so the recorded total always reflects
        money actually spent -- the request has already happened by the time we
        get here, and pretending otherwise would under-report the bill.
        """
        cost = cost_usd(usage, model, self.price_overrides)

        self.spent_usd += cost
        self.requests += 1
        self.usage = self.usage + usage
        self.by_model[model] = self.by_model.get(model, 0.0) + cost
        if agent_id is not None:
            self.by_agent[agent_id] = self.by_agent.get(agent_id, 0.0) + cost

        if self.halt_on_exceed and self.spent_usd >= self.ceiling_usd:
            raise BudgetExceeded(self.spent_usd, self.ceiling_usd)
        return cost

    def summary(self) -> dict[str, object]:
        return {
            "spent_usd": round(self.spent_usd, 6),
            "ceiling_usd": self.ceiling_usd,
            "remaining_usd": round(self.remaining_usd, 6),
            "requests": self.requests,
            "cost_per_request": round(self.spent_usd / self.requests, 6) if self.requests else 0.0,
            "cache_hit_rate": round(self.usage.cache_hit_rate, 4),
            **self.usage.to_dict(),
            "by_model": {k: round(v, 6) for k, v in sorted(self.by_model.items())},
        }

    def report(self) -> str:
        """One-line human summary, for the CLI."""
        return (
            f"${self.spent_usd:.4f} / ${self.ceiling_usd:.2f} over {self.requests} requests "
            f"({self.usage.total_input:,} in / {self.usage.output_tokens:,} out, "
            f"cache hits {self.usage.cache_hit_rate:.0%})"
        )
