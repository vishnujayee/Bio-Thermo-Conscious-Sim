"""Macro-scenario controls: making the world change under the organisms.

Every run so far has been in a world whose rules were fixed at tick zero. That
turned out to matter more than it sounds. Phase 4 measured that integration does
not pay for itself, and the strongest remaining explanation is that **a good map
is worth little where the right answer never changes**: error correction protects
a model of an environment that was not going to surprise anyone anyway.

So this module is not scaffolding for a dashboard. It is the apparatus for the
experiment that could overturn the headline result, which is why it was built
before the panels.

Two kinds of intervention, deliberately kept apart:

  * **`set`** changes a *rule* -- a config value. Ambient temperature rises, food
    stops respawning. The world's physics is different from that tick on.
  * **`do`** changes the *state* -- an event that happens to the world as it is.
    A meteor, a die-off, a windfall.

The distinction is worth keeping because they answer different questions. `set`
asks what organisms adapted to one world do in another; `do` asks what they do
after a shock in the same world.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from btcs.config import Config, ConfigError


@dataclass(frozen=True, slots=True)
class ScenarioEvent:
    """One intervention, at one tick."""

    at: int
    # Dotted config paths to new values, e.g. {"world.ambient_temperature": 1.6}.
    set: dict[str, Any] = field(default_factory=dict)
    # Name of a world-state action from ACTIONS below.
    do: str | None = None
    # Free text, carried into the event log so a run explains itself.
    note: str = ""


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    events: tuple[ScenarioEvent, ...] = ()

    def due(self, tick: int) -> list[ScenarioEvent]:
        return [event for event in self.events if event.at == tick]

    @property
    def last_tick(self) -> int:
        return max((event.at for event in self.events), default=0)


# -- world-state actions ---------------------------------------------------
#
# Each takes the world and a seeded stream and changes what is there, without
# changing any rule. Registered by name so a config file can only ask for one
# that exists -- an unknown name fails at load rather than at tick 300.


def _halve_nodes(world: Any, stream: Any) -> str:
    world.node_energy *= 0.5
    return f"halved the stored energy of {len(world.node_energy)} nodes"


def _windfall(world: Any, stream: Any) -> str:
    world.node_energy *= 2.0
    return f"doubled the stored energy of {len(world.node_energy)} nodes"


def _chill(world: Any, stream: Any) -> str:
    """Strip the thermal field back to ambient: every gradient, gone at once."""
    world.node_temperature[:] = world.config.world.ambient_temperature
    return "every node dropped to ambient temperature"


def _scatter(world: Any, stream: Any) -> str:
    """Move every node somewhere else. The map an organism holds is now wrong."""
    count = len(world.node_y)
    if count == 0:
        return "nothing to scatter"
    world.node_y[:] = stream.integers(0, world.height, size=count)
    world.node_x[:] = stream.integers(0, world.width, size=count)
    return f"relocated {count} nodes"


ACTIONS: dict[str, Callable[[Any, Any], str]] = {
    "halve_nodes": _halve_nodes,
    "windfall": _windfall,
    "chill": _chill,
    "scatter": _scatter,
}


# -- presets ---------------------------------------------------------------


def _seasons(period: int, ticks: int) -> list[ScenarioEvent]:
    """Alternating plenty and scarcity, forever.

    The scenario built for OPEN-19. Nodes move and the climate swings, so the
    right answer this season is the wrong one next season -- which is the only
    condition under which holding an accurate model of the world could be worth
    paying for. If integration never pays here, it does not pay.
    """
    events: list[ScenarioEvent] = []
    for index, at in enumerate(range(period, ticks + 1, period)):
        winter = index % 2 == 0
        events.append(
            ScenarioEvent(
                at=at,
                set={
                    # Milder than the first attempt. At 1.5 / 0.2 the population
                    # fell to one survivor, which measures extinction rather than
                    # adaptation -- a scenario has to be survivable to be an
                    # experiment.
                    "world.ambient_temperature": 1.3 if winter else 1.0,
                    "world.node_respawn_rate": 0.35 if winter else 0.85,
                },
                # Only winter moves the food. Scattering every season leaves no
                # interval long enough for a map to be worth holding, which
                # would rig the experiment against integration rather than test it.
                do="scatter" if winter else None,
                note="winter" if winter else "summer",
            )
        )
    return events


def preset(name: str, ticks: int) -> Scenario:
    """Build a named scenario sized to the length of the run.

    The severities were measured, not chosen. A scenario that reliably wipes the
    population out measures extinction rather than adaptation, and one nothing
    notices measures nothing at all -- so each of these was tuned until survival
    was in doubt but not settled.
    """
    quarter = max(1, ticks // 4)

    match name:
        case "none":
            return Scenario("none")
        case "heat_wave":
            return Scenario("heat_wave", (
                ScenarioEvent(quarter, {"world.ambient_temperature": 1.25},
                              note="the climate warms; every gradient shrinks"),
                ScenarioEvent(quarter * 3, {"world.ambient_temperature": 1.0},
                              note="the climate returns"),
            ))
        case "drought":
            return Scenario("drought", (
                ScenarioEvent(quarter, {"world.node_respawn_rate": 0.08},
                              note="new food becomes rare"),
                ScenarioEvent(quarter * 3, {"world.node_respawn_rate": 0.6},
                              note="the rains return"),
            ))
        case "gradient_collapse":
            return Scenario("gradient_collapse", (
                ScenarioEvent(quarter * 2, do="chill",
                              note="every gradient vanishes at once"),
            ))
        case "bottleneck":
            return Scenario("bottleneck", (
                ScenarioEvent(quarter, do="halve_nodes", note="a lean year"),
                ScenarioEvent(quarter * 2, do="halve_nodes", note="a leaner one"),
                ScenarioEvent(quarter * 3, do="windfall", note="recovery"),
            ))
        case "seasons":
            return Scenario("seasons", tuple(_seasons(max(20, ticks // 10), ticks)))

    raise ConfigError(
        f"unknown scenario {name!r}; known: {', '.join(sorted(PRESETS))}"
    )


PRESETS = frozenset({"none", "heat_wave", "drought", "gradient_collapse",
                     "bottleneck", "seasons"})


# -- applying an override --------------------------------------------------


def apply_overrides(config: Config, changes: dict[str, Any]) -> Config:
    """Return a new Config with dotted paths replaced.

    Frozen dataclasses all the way down, so this rebuilds rather than mutates.
    The caller is responsible for rebinding the result everywhere at once --
    `World` and `PhysicsEngine` must share one instance, or the engine charges
    one price while the world applies another and the ledger quietly stops
    balancing.
    """
    sections: dict[str, dict[str, Any]] = {}
    for path, value in changes.items():
        section, _, name = path.partition(".")
        if not name:
            raise ConfigError(
                f"scenario override {path!r} must be a dotted path like "
                "'world.ambient_temperature'"
            )
        if not hasattr(config, section):
            raise ConfigError(f"scenario override names unknown section {section!r}")
        target = getattr(config, section)
        if not any(f.name == name for f in dataclasses.fields(target)):
            raise ConfigError(
                f"scenario override names unknown field {name!r} in [{section}]"
            )
        if isinstance(value, float | int) and not math.isfinite(value):
            raise ConfigError(f"scenario override {path!r} must be finite")
        sections.setdefault(section, {})[name] = value

    updated = {
        section: dataclasses.replace(getattr(config, section), **fields)
        for section, fields in sections.items()
    }
    return dataclasses.replace(config, **updated)


def build(config: Config) -> Scenario:
    """The scenario a config asks for: a preset, explicit events, or nothing."""
    scenario = config.scenario
    if scenario.events:
        return Scenario(scenario.name or "custom", tuple(scenario.events))
    return preset(scenario.name, config.run.max_ticks)
