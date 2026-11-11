"""The contract between a brain and the physics.

A brain -- deterministic or LLM-driven -- receives a `Percept` and returns a
`Decision`. Nothing else crosses that line. Keeping the interface this narrow is
what lets the mock brain and the LLM brain be swapped without the engine
noticing, and it is the same shape the MCP tool surface will expose in Phase 2.

Note what a Percept does *not* contain: any array, any other organism's private
state, any absolute node identity beyond a local index. An organism perceives
its neighbourhood, not the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from btcs.world.grid import NodeView


@dataclass(frozen=True, slots=True)
class LastAction:
    """What came of the organism's previous act.

    Without this an organism is blind to the consequences of its own behaviour:
    it is refused, told nothing, and repeats the mistake next tick. The first
    real-model run failed nineteen of twenty-three acts that way, every one of
    them the same act for the same reason.

    This is a *sense*, not a rule. Nothing here tells an organism what to do; it
    tells it what happened, which is the minimum required for any within-lifetime
    learning to be possible at all. A brain remains free to ignore it.
    """

    kind: str
    ok: bool
    tick: int
    detail: str = ""

    def summary(self) -> str:
        outcome = "succeeded" if self.ok else f"failed ({self.detail})"
        return f"{self.kind} {outcome}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ok": self.ok, "tick": self.tick,
                "detail": self.detail}


@dataclass(frozen=True, slots=True)
class Percept:
    """What one organism senses this tick.

    Values may be degraded relative to the truth: an organism whose internal
    entropy has passed `s_warn` receives noisy readings, and past `s_confuse`
    some nodes are simply missing from `nodes`. It is not told which. That
    asymmetry is the point -- accumulated disorder should corrupt the map, not
    announce itself.
    """

    tick: int
    agent_id: str
    energy: float
    entropy: float
    phi: float
    free_energy: float
    local_temperature: float
    ambient_temperature: float
    degradation: int
    nodes: tuple[NodeView, ...]
    on_node: bool
    # What came of the previous act, or None if there was none -- or if the
    # organism is now too disordered to remember it.
    last_action: LastAction | None = None
    # True when the readings above have been corrupted. The engine knows; the
    # organism is not told.
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "energy": round(self.energy, 3),
            "entropy": round(self.entropy, 3),
            "phi": round(self.phi, 3),
            "free_energy": round(self.free_energy, 3),
            "local_temperature": round(self.local_temperature, 4),
            "on_node": self.on_node,
            "last_action": self.last_action.to_dict() if self.last_action else None,
            "nodes": [
                {
                    "dy": n.dy,
                    "dx": n.dx,
                    "energy": round(n.energy, 2),
                    "temperature": round(n.temperature, 3),
                    "extractable": round(n.extractable, 2),
                }
                for n in self.nodes
            ],
        }


# -- actions --------------------------------------------------------------
#
# A closed set. Replication arrives in Phase 5; executable genomes in Phase 5b.


@dataclass(frozen=True, slots=True)
class Rest:
    """Do nothing but pay the basal cost. Cheap, and sometimes correct."""


@dataclass(frozen=True, slots=True)
class Move:
    """One step, including diagonals. Costs energy and dumps waste heat."""

    dy: int
    dx: int


@dataclass(frozen=True, slots=True)
class Ingest:
    """Draw stock from the node underfoot, capped by the Carnot bound."""

    joules: float


@dataclass(frozen=True, slots=True)
class Dissipate:
    """Erase internal entropy, paying the Landauer price in energy."""

    bits: float


@dataclass(frozen=True, slots=True)
class Replicate:
    """Spend energy to make a mutated copy of yourself.

    The only act whose payoff never accrues to the actor. Everything else an
    organism can do is an investment in its own survival; this one is a transfer
    out of it, and whether it is worth making is exactly the question the
    simulation exists to ask.
    """


Action = Rest | Move | Ingest | Dissipate | Replicate


@dataclass(frozen=True, slots=True)
class Decision:
    """A brain's output: what to do, and what it cost to decide.

    `tokens` is the load-bearing field. It is the bridge between cognition and
    thermodynamics -- the engine converts it directly into internal entropy via
    law L3, so an expensive thought is a physically hot one. The mock brain
    reports a deterministic count; the LLM brain will report real API usage.
    """

    action: Action
    tokens: int
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def describe(action: Action) -> str:
    """Short label for logs and dashboards."""
    match action:
        case Rest():
            return "rest"
        case Move(dy=dy, dx=dx):
            return f"move({dy:+d},{dx:+d})"
        case Ingest(joules=joules):
            return f"ingest({joules:.1f})"
        case Dissipate(bits=bits):
            return f"dissipate({bits:.1f})"
        case Replicate():
            return "replicate"
    return "unknown"
