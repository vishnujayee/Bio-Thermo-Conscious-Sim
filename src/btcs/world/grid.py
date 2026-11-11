"""The 2D physical environment.

A toroidal grid carrying a temperature field and a population of free-energy
nodes. The world is an *open* system: energy enters when nodes respawn and
leaves through radiative coupling to an external sink at ambient temperature.
Between those two boundaries, everything runs downhill.

The design decision that matters most here is that a node has two independent
properties -- an energy *stock* and a *temperature* -- and both decay. Under the
Carnot bound (physics.L2) an organism can only extract work in proportion to
`1 - T_ambient/T_node`, so a node that has cooled to ambient is worthless even
while its stock is untouched. That is what the project plan means by free energy
degrading into ambient thermal noise: the energy is still there, it has simply
stopped being *free*.

Topology is a torus (wrap-around) rather than a bounded box. A box would make
corners systematically safer than the middle, and organisms would evolve to
exploit the geometry rather than the physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from btcs.config import Config
from btcs.events import EventLog, EventType
from btcs.rng import RngService
from btcs.world.ledger import EntropyLedger


@dataclass(frozen=True, slots=True)
class NodeView:
    """One free-energy node as an organism perceives it."""

    index: int
    dy: int
    dx: int
    distance: float
    energy: float
    temperature: float
    # Fraction of this node's energy that is actually extractable as work,
    # after the Carnot bound and engine inefficiency. Zero at thermal equilibrium.
    efficiency: float

    @property
    def extractable(self) -> float:
        return self.energy * self.efficiency


class World:
    """Temperature field plus free-energy nodes on a torus."""

    __slots__ = ("_config", "_rng", "temperature", "node_y", "node_x",
                 "node_energy", "node_temperature", "_next_node_serial")

    def __init__(self, config: Config, rng: RngService) -> None:
        self._config = config
        self._rng = rng

        world = config.world
        self.temperature = np.full(
            (world.height, world.width), world.ambient_temperature, dtype=np.float64
        )

        self.node_y = np.zeros(0, dtype=np.int64)
        self.node_x = np.zeros(0, dtype=np.int64)
        self.node_energy = np.zeros(0, dtype=np.float64)
        self.node_temperature = np.zeros(0, dtype=np.float64)
        self._next_node_serial = 0

        for _ in range(world.node_count):
            self._spawn_node()

    # -- geometry ---------------------------------------------------------

    @property
    def config(self) -> Config:
        """The config this world was built from.

        Exposed so `PhysicsEngine` can verify the two share one instance -- see
        the check in its constructor.
        """
        return self._config

    @property
    def height(self) -> int:
        return self._config.world.height

    @property
    def width(self) -> int:
        return self._config.world.width

    @property
    def node_count(self) -> int:
        return int(self.node_energy.size)

    def wrap(self, y: int, x: int) -> tuple[int, int]:
        return int(y) % self.height, int(x) % self.width

    def toroidal_delta(self, from_y: int, from_x: int, to_y: int, to_x: int) -> tuple[int, int]:
        """Shortest signed offset on a torus, so 'nearest' respects wrap-around."""
        dy = (to_y - from_y + self.height // 2) % self.height - self.height // 2
        dx = (to_x - from_x + self.width // 2) % self.width - self.width // 2
        return int(dy), int(dx)

    def temperature_at(self, y: int, x: int) -> float:
        return float(self.temperature[y % self.height, x % self.width])

    # -- nodes ------------------------------------------------------------

    def _spawn_node(self) -> int:
        stream = self._rng.stream("world.nodes")
        world = self._config.world

        y = int(stream.integers(0, self.height))
        x = int(stream.integers(0, self.width))
        energy = float(
            max(1.0, stream.normal(world.node_energy_mean, world.node_energy_sigma))
        )
        temperature = float(
            max(
                world.ambient_temperature * 1.05,
                stream.normal(world.node_temperature_mean, world.node_temperature_sigma),
            )
        )

        self.node_y = np.append(self.node_y, y)
        self.node_x = np.append(self.node_x, x)
        self.node_energy = np.append(self.node_energy, energy)
        self.node_temperature = np.append(self.node_temperature, temperature)
        self._next_node_serial += 1
        return self.node_count - 1

    def carnot_efficiency(self, node_temperature: float, cold_temperature: float) -> float:
        """Ceiling on work extractable from a node, times engine inefficiency.

        `1 - T_cold/T_hot` is the Carnot bound; a real engine never reaches it,
        hence the additional `ingest_efficiency` factor. Clamped at zero so a
        node at or below the cold reservoir yields nothing rather than going
        negative and paying organisms to eat.

        The cold reservoir is the **local cell**, not global ambient. This is
        not a detail: an engine can only reject waste heat into the reservoir it
        actually touches, and using ambient here would let an organism standing
        in a hot cell extract more work than the second law permits -- the
        entropy ledger goes negative within a few ticks if you get this wrong.

        Using the local cell also produces the crowding dynamics the project
        wants for free: waste heat from work and cognition warms the neighbourhood,
        which raises T_cold, which shrinks everyone's extraction efficiency. A
        crowd degrades its own metabolism by being a crowd.
        """
        physics = self._config.physics
        cold = max(physics.min_temperature, cold_temperature)
        if not physics.carnot_cap_enabled:
            return physics.ingest_efficiency
        if node_temperature <= cold:
            return 0.0
        return (1.0 - cold / node_temperature) * physics.ingest_efficiency

    def nodes_near(self, y: int, x: int, radius: int, limit: int) -> list[NodeView]:
        """Nodes within a Chebyshev radius, richest-first.

        Ranked by extractable energy rather than raw stock, because a cold node
        full of energy is not food.
        """
        if self.node_count == 0:
            return []

        half_h, half_w = self.height // 2, self.width // 2
        dy = (self.node_y - y + half_h) % self.height - half_h
        dx = (self.node_x - x + half_w) % self.width - half_w
        within = np.flatnonzero((np.abs(dy) <= radius) & (np.abs(dx) <= radius))
        if within.size == 0:
            return []

        # Efficiency is previewed against the temperature at the *node's* cell,
        # since that is where the organism would stand to eat it.
        views = [
            NodeView(
                index=int(i),
                dy=int(dy[i]),
                dx=int(dx[i]),
                distance=float(max(abs(dy[i]), abs(dx[i]))),
                energy=float(self.node_energy[i]),
                temperature=float(self.node_temperature[i]),
                efficiency=self.carnot_efficiency(
                    float(self.node_temperature[i]),
                    float(self.temperature[self.node_y[i], self.node_x[i]]),
                ),
            )
            for i in within
        ]
        # Deterministic ordering: richest first, ties broken by index so the
        # list never depends on numpy's internal sort stability.
        views.sort(key=lambda v: (-v.extractable, v.index))
        return views[:limit]

    def node_index_at(self, y: int, x: int) -> int:
        """Index of the richest node on this exact cell, or -1."""
        if self.node_count == 0:
            return -1
        here = np.flatnonzero((self.node_y == y) & (self.node_x == x))
        if here.size == 0:
            return -1
        return int(here[np.argmax(self.node_energy[here])])

    def draw_from_node(self, index: int, joules: float) -> float:
        """Remove up to `joules` of stock from a node. Returns what was available."""
        available = float(self.node_energy[index])
        drawn = min(available, max(0.0, joules))
        self.node_energy[index] = available - drawn
        return drawn

    # -- heat -------------------------------------------------------------

    def deposit_heat(self, y: int, x: int, joules: float, ledger: EntropyLedger, process: str) -> None:
        """Add waste heat to one cell and bill the entropy it carries."""
        if joules <= 0.0:
            return
        y, x = self.wrap(y, x)
        temperature = max(self._config.physics.min_temperature, float(self.temperature[y, x]))
        ledger.heat_to_environment(joules, temperature, process)
        self.temperature[y, x] += joules / self._config.world.cell_heat_capacity

    # -- per-tick evolution -----------------------------------------------

    def step(self, tick: int, ledger: EntropyLedger, log: EventLog) -> None:
        """Advance the environment one tick.

        Order is deliberate: nodes decay, then their leaked heat enters the
        field, then the field diffuses, then the field radiates to the sink.
        Each stage files its own entropy entry.
        """
        self._decay_nodes(tick, ledger, log)
        self._diffuse(ledger)
        self._radiate_to_sink(ledger)
        self._respawn(tick, log)
        self._remove_exhausted(tick, log)

    def _decay_nodes(self, tick: int, ledger: EntropyLedger, log: EventLog) -> None:
        """Nodes cool toward ambient and leak stock into the field.

        Cooling is the master clock of scarcity: it destroys the *gradient*,
        which is what organisms actually eat.
        """
        if self.node_count == 0:
            return
        world = self._config.world

        gap = world.ambient_temperature - self.node_temperature
        self.node_temperature += gap * world.node_cooling_rate

        leaked = self.node_energy * world.node_leak_rate
        self.node_energy -= leaked

        # Leaked stock becomes ambient heat at the node's own cell.
        floor = self._config.physics.min_temperature
        for i in np.flatnonzero(leaked > 0.0):
            y, x = int(self.node_y[i]), int(self.node_x[i])
            joules = float(leaked[i])
            temperature = max(floor, float(self.temperature[y, x]))
            ledger.heat_to_environment(joules, temperature, "node_leak")
            self.temperature[y, x] += joules / world.cell_heat_capacity

    def _diffuse(self, ledger: EntropyLedger) -> None:
        """Five-point explicit diffusion on the torus.

        Conserves total field energy exactly (the discrete Laplacian sums to
        zero) and always raises entropy, because heat only moves down-gradient.
        The entropy is computed cell-wise as `C * dT / T`, which is the correct
        first-order accounting for a small step.
        """
        alpha = self._config.world.thermal_diffusion
        if alpha <= 0.0:
            return

        field = self.temperature
        laplacian = (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
            - 4.0 * field
        )
        delta = alpha * laplacian

        floor = self._config.physics.min_temperature
        capacity = self._config.world.cell_heat_capacity
        safe = np.maximum(field, floor)
        # Sum of C*dT/T over all cells. Positive whenever heat flows hot -> cold.
        bits = float(np.sum(capacity * delta / safe)) / (
            ledger.k_boltzmann * np.log(2.0)
        )
        ledger.environment_entropy(bits, "diffusion")

        self.temperature = np.maximum(field + delta, floor)

    def _radiate_to_sink(self, ledger: EntropyLedger) -> None:
        """Radiate toward ambient. This is the world's boundary with the outside.

        Heat leaving a cell at T for a sink at T_ambient generates entropy
        whenever T != T_ambient, which is what makes the system dissipative
        rather than merely closed.
        """
        world = self._config.world
        coupling = world.ambient_coupling
        if coupling <= 0.0:
            return

        floor = self._config.physics.min_temperature
        field = self.temperature
        delta = (world.ambient_temperature - field) * coupling
        joules = -delta * world.cell_heat_capacity  # positive where heat leaves

        safe = np.maximum(field, floor)
        ambient = max(floor, world.ambient_temperature)
        bits = float(
            np.sum(joules / ambient - joules / safe)
        ) / (ledger.k_boltzmann * np.log(2.0))
        ledger.environment_entropy(bits, "radiation")

        self.temperature = np.maximum(field + delta, floor)

    def _respawn(self, tick: int, log: EventLog) -> None:
        """Fresh free energy enters the world. Without this it reaches heat death."""
        rate = self._config.world.node_respawn_rate
        if rate <= 0.0:
            return
        if self._rng.stream("world.respawn").random() < rate:
            index = self._spawn_node()
            log.emit(
                tick,
                EventType.NODE_SPAWNED,
                None,
                y=int(self.node_y[index]),
                x=int(self.node_x[index]),
                energy=float(self.node_energy[index]),
                temperature=float(self.node_temperature[index]),
            )

    def _remove_exhausted(self, tick: int, log: EventLog) -> None:
        if self.node_count == 0:
            return
        threshold = self._config.world.node_exhaustion_threshold
        spent = np.flatnonzero(self.node_energy < threshold)
        if spent.size == 0:
            return

        for i in spent:
            log.emit(tick, EventType.NODE_EXHAUSTED, None,
                     y=int(self.node_y[i]), x=int(self.node_x[i]))

        keep = np.setdiff1d(np.arange(self.node_count), spent, assume_unique=True)
        self.node_y = self.node_y[keep]
        self.node_x = self.node_x[keep]
        self.node_energy = self.node_energy[keep]
        self.node_temperature = self.node_temperature[keep]

    # -- diagnostics ------------------------------------------------------

    def total_field_energy(self) -> float:
        return float(np.sum(self.temperature) * self._config.world.cell_heat_capacity)

    def total_node_energy(self) -> float:
        return float(np.sum(self.node_energy))

    def total_extractable_energy(self) -> float:
        """Free energy actually available as work -- the honest measure of scarcity."""
        if self.node_count == 0:
            return 0.0
        efficiencies = np.array(
            [
                self.carnot_efficiency(
                    float(self.node_temperature[i]),
                    float(self.temperature[self.node_y[i], self.node_x[i]]),
                )
                for i in range(self.node_count)
            ]
        )
        return float(np.sum(self.node_energy * efficiencies))

    def snapshot(self) -> dict[str, float]:
        return {
            "nodes": self.node_count,
            "node_energy": self.total_node_energy(),
            "extractable_energy": self.total_extractable_energy(),
            "field_energy": self.total_field_energy(),
            "mean_temperature": float(np.mean(self.temperature)),
            "max_temperature": float(np.max(self.temperature)),
        }
