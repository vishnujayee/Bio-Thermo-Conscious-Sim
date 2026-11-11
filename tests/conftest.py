"""Shared fixtures.

The `small_config` fixture deliberately shrinks the world rather than reusing
the shipped defaults: tests should be fast, and a 48x48 grid with 90 nodes
spends most of its time on physics that the assertions never look at.
"""

from __future__ import annotations

import dataclasses

import pytest

from btcs.config import Config
from btcs.events import EventLog
from btcs.rng import RngService
from btcs.world.agents import Agent, AgentRegistry, Genome
from btcs.world.grid import World
from btcs.world.ledger import EntropyLedger
from btcs.world.physics import PhysicsEngine


@pytest.fixture
def small_config() -> Config:
    base = Config()
    return dataclasses.replace(
        base,
        run=dataclasses.replace(base.run, max_ticks=60, seed=42),
        world=dataclasses.replace(base.world, width=16, height=16, node_count=10),
        agent=dataclasses.replace(base.agent, initial_population=6),
    )


@pytest.fixture
def log(tmp_path) -> EventLog:
    with EventLog(tmp_path / "events.jsonl") as handle:
        yield handle


@pytest.fixture
def rng() -> RngService:
    return RngService(42)


@pytest.fixture
def world(small_config, rng) -> World:
    return World(small_config, rng)


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry()


@pytest.fixture
def engine(small_config, world, registry, rng, log) -> PhysicsEngine:
    return PhysicsEngine(small_config, world, registry, rng, log)


@pytest.fixture
def ledger(small_config) -> EntropyLedger:
    return EntropyLedger(k_boltzmann=small_config.physics.k_boltzmann)


@pytest.fixture
def make_agent(small_config, registry):
    """Factory for an organism placed at a chosen spot with chosen vitals."""

    def _make(*, y=0, x=0, energy=None, entropy=None, phi=None, **kwargs) -> Agent:
        cfg = small_config.agent
        agent = Agent(
            id=registry.next_id(1),
            y=y,
            x=x,
            energy=cfg.initial_energy if energy is None else energy,
            entropy=cfg.initial_entropy if entropy is None else entropy,
            phi=cfg.initial_phi if phi is None else phi,
            genome=Genome(strategy="test", policy=cfg.policy),
            **kwargs,
        )
        return registry.add(agent)

    return _make
