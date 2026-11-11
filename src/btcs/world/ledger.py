"""Double-entry entropy bookkeeping.

The second law is the project's central correctness property, and the only way
to keep it honest is to make it *arithmetic* rather than an aspiration. Every
process that moves energy or entropy files an entry here; at the end of each
tick the runner asserts that the books balance non-negatively.

Units
-----
Entropy is carried in **bits** everywhere. Thermodynamic entropy is Q/T in
energy-per-temperature units, so the conversion is

    S_bits = Q / (k_B * T * ln2)

which is exactly the Landauer relation read backwards: dumping `n * k_B * T *
ln2` joules of heat into a reservoir at temperature `T` raises its entropy by
`n` bits.

Why the sign conventions matter
-------------------------------
`agent_bits` is the change in entropy *inside* organisms, and it may legally be
negative -- that is what dissipation is for. `env_bits` is entropy delivered to
the environment, and in a correct implementation it always covers the deficit.
If `total` ever goes negative, some process is creating order for free, and the
run is physically meaningless from that tick onward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

LN2 = math.log(2.0)


@dataclass(slots=True)
class EntropyLedger:
    """Tick-scoped record of entropy movement.

    A fresh ledger is opened each tick so that a violation can be attributed to
    the tick that caused it, rather than being buried in a running total.
    """

    k_boltzmann: float = 1.0

    agent_bits: float = 0.0
    env_bits: float = 0.0

    # Per-process attribution, so a violation can be traced to the law that
    # produced it instead of just reported as a number.
    by_process: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """dS_total for the tick. Must be >= 0 (within tolerance)."""
        return self.agent_bits + self.env_bits

    # -- entries ----------------------------------------------------------

    def agent_entropy(self, bits: float, process: str) -> None:
        """Record a change in an organism's internal entropy, in bits.

        Negative values are legitimate only when some process has simultaneously
        paid for them by heating the environment (see `heat_to_environment`).
        """
        self.agent_bits += bits
        self._attribute(process, bits)

    def environment_entropy(self, bits: float, process: str) -> None:
        """Record entropy generated directly in the environment, in bits."""
        self.env_bits += bits
        self._attribute(process, bits)

    def heat_to_environment(self, joules: float, temperature: float, process: str) -> float:
        """Dump `joules` of heat into a reservoir at `temperature`.

        This is the workhorse: almost every energy cost in the simulation ends
        up here, because energy an organism spends does not vanish -- it becomes
        heat, and heat carries entropy. Returns the bits generated.
        """
        if joules < 0.0:
            raise ValueError(f"heat must be non-negative, got {joules!r}")
        bits = joules / (self.k_boltzmann * temperature * LN2)
        self.env_bits += bits
        self._attribute(process, bits)
        return bits

    def heat_transfer(
        self, joules: float, from_temperature: float, to_temperature: float, process: str
    ) -> float:
        """Move `joules` between two reservoirs, recording the net entropy.

        Used by thermal diffusion and by the coupling to the external sink. The
        result is non-negative exactly when heat flows from hot to cold, so this
        method is itself a check on the field update being physical.
        """
        bits = (
            joules / (self.k_boltzmann * to_temperature * LN2)
            - joules / (self.k_boltzmann * from_temperature * LN2)
        )
        self.env_bits += bits
        self._attribute(process, bits)
        return bits

    def _attribute(self, process: str, bits: float) -> None:
        self.by_process[process] = self.by_process.get(process, 0.0) + bits

    # -- inspection -------------------------------------------------------

    def is_valid(self, tolerance: float) -> bool:
        return self.total >= -abs(tolerance)

    def worst_process(self) -> tuple[str, float] | None:
        """The most entropy-negative process, for diagnosing a violation."""
        if not self.by_process:
            return None
        name = min(self.by_process, key=lambda key: self.by_process[key])
        return name, self.by_process[name]

    def summary(self) -> dict[str, float]:
        return {
            "dS_agents": self.agent_bits,
            "dS_env": self.env_bits,
            "dS_total": self.total,
            **{f"dS_{name}": bits for name, bits in sorted(self.by_process.items())},
        }


class SecondLawViolation(AssertionError):
    """Raised when a tick produces a net entropy decrease.

    This is always a bug in the engine, never a legitimate simulation outcome.
    It is an exception rather than a logged warning because every number
    computed after a violation is physically meaningless.
    """

    def __init__(self, tick: int, ledger: EntropyLedger) -> None:
        worst = ledger.worst_process()
        culprit = f"; largest negative contributor: {worst[0]}={worst[1]:.6g} bits" if worst else ""
        super().__init__(
            f"second law violated at tick {tick}: "
            f"dS_total={ledger.total:.6g} bits "
            f"(agents={ledger.agent_bits:.6g}, env={ledger.env_bits:.6g}){culprit}"
        )
        self.tick = tick
        self.ledger = ledger
