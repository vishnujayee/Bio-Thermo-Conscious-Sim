"""What a thought costs, in dollars and in bits.

Two conversions live here, and they are deliberately in the same file because
they are the same idea measured twice. The token counts an API bills for are the
token counts law L3 turns into internal entropy. The invoice and the metabolism
are readings from one meter.

Prices are USD per million tokens, verified 2026-09. They are reference data
rather than physics, so they live in code with a date on them rather than in the
run config -- but `price_of` accepts an override map so an experiment can model
a different pricing regime without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass

from btcs.config import CognitionConfig


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens."""

    input: float
    output: float

    @property
    def cache_write(self) -> float:
        """Writing to cache costs about 1.25x fresh input."""
        return self.input * 1.25

    @property
    def cache_read(self) -> float:
        """Reading from cache costs about a tenth of fresh input."""
        return self.input * 0.1


# Verified 2026-09. Anthropic first-party API rates.
MODEL_PRICES: dict[str, ModelPrice] = {
    "claude-opus-5": ModelPrice(input=5.00, output=25.00),
    "claude-sonnet-5": ModelPrice(input=2.00, output=10.00),
    "claude-haiku-4-5": ModelPrice(input=1.00, output=5.00),
    "claude-opus-4-8": ModelPrice(input=5.00, output=25.00),
    "claude-sonnet-4-6": ModelPrice(input=3.00, output=15.00),
}

# Charged for a model we have no price for, so an unknown id shows up as
# conspicuously expensive rather than silently free.
UNKNOWN_MODEL_PRICE = ModelPrice(input=15.00, output=75.00)


def price_of(model: str, overrides: dict[str, ModelPrice] | None = None) -> ModelPrice:
    if overrides and model in overrides:
        return overrides[model]
    return MODEL_PRICES.get(model, UNKNOWN_MODEL_PRICE)


@dataclass(frozen=True, slots=True)
class Usage:
    """One request's token counts, as the API reports them.

    Kept as a value object rather than passing the SDK's usage around, so that
    the replay path can reconstruct usage from a log without needing the SDK.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def from_response(cls, usage: object) -> "Usage":
        """Read an SDK usage object defensively.

        Fields have been added to `usage` over time; missing ones read as zero
        rather than raising, because an accounting crash mid-run would cost more
        than a slightly under-reported cache statistic.
        """
        return cls(
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_creation_input_tokens=int(
                getattr(usage, "cache_creation_input_tokens", 0) or 0
            ),
            cache_read_input_tokens=int(
                getattr(usage, "cache_read_input_tokens", 0) or 0
            ),
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            self.cache_read_input_tokens + other.cache_read_input_tokens,
        )

    @property
    def total_input(self) -> int:
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    @property
    def cache_hit_rate(self) -> float:
        """Share of input served from cache. The number to watch in Phase 3.

        If this sits near zero across repeated ticks, a silent invalidator is at
        work -- a timestamp in the system prompt, an unsorted dict, a tool list
        that varies -- and the run is paying full price for a prefix it was
        supposed to be reusing.
        """
        return self.cache_read_input_tokens / self.total_input if self.total_input else 0.0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


def cost_usd(
    usage: Usage, model: str, overrides: dict[str, ModelPrice] | None = None
) -> float:
    """Dollars for one request."""
    price = price_of(model, overrides)
    return (
        usage.input_tokens * price.input
        + usage.cache_creation_input_tokens * price.cache_write
        + usage.cache_read_input_tokens * price.cache_read
        + usage.output_tokens * price.output
    ) / 1_000_000.0


def entropy_tokens(usage: Usage, cognition: CognitionConfig) -> float:
    """Weighted token count that law L3 converts into internal entropy.

    The weights are the whole of OPEN-10. Under the defaults a cache read is
    worth a tenth of a fresh token, mirroring its price -- which means an
    organism whose prompt prefix stays stable literally thinks more cheaply, in
    entropy as well as in dollars, and cache-stable lineages should become
    fitter. Set `cache_read_entropy_weight` to 1.0 to switch that selective
    pressure off and find out whether it was doing any work.
    """
    return (
        usage.input_tokens * cognition.input_entropy_weight
        + usage.cache_creation_input_tokens * cognition.cache_write_entropy_weight
        + usage.cache_read_input_tokens * cognition.cache_read_entropy_weight
        + usage.output_tokens * cognition.output_entropy_weight
    )
