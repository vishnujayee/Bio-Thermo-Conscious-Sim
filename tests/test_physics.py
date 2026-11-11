"""The five laws, checked individually.

Each law gets tests for its formula, for the entropy it files, and for the edge
cases where an organism cannot afford what it asked for. The integration-level
guarantees (dS_total over a whole run, reproducibility) live in test_runner.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from btcs.world.agents import Degradation
from btcs.world.types import Dissipate, Ingest, Move, Rest

LN2 = math.log(2.0)


class TestLandauer:
    """L1 -- erasing n bits costs n*k_B*T*ln2/eta joules."""

    def test_cost_matches_the_formula(self, engine, small_config):
        physics = small_config.physics
        bits, temperature = 10.0, 2.0
        expected = bits * physics.k_boltzmann * temperature * LN2 / physics.erase_efficiency
        assert engine.landauer_cost(bits, temperature) == pytest.approx(expected)

    def test_cost_is_linear_in_bits(self, engine):
        assert engine.landauer_cost(20.0, 1.0) == pytest.approx(
            2.0 * engine.landauer_cost(10.0, 1.0)
        )

    def test_erasing_is_more_expensive_when_hot(self, engine):
        """The trap the whole selection story depends on: heat makes forgetting dear."""
        assert engine.landauer_cost(10.0, 4.0) > engine.landauer_cost(10.0, 1.0)

    def test_dissipation_nets_positive_entropy(self, engine, ledger, make_agent, world):
        """The crucial property. Erasure lowers an organism's entropy, and must
        raise the world's by strictly more, because eta_erase < 1."""
        agent = make_agent(energy=500.0, entropy=50.0)
        engine.apply(agent, Dissipate(20.0), ledger, tick=1)

        assert ledger.agent_bits < 0
        assert ledger.total > 0

    def test_environment_gains_bits_over_eta(self, small_config, engine, ledger, make_agent):
        """dS_env should be exactly n/eta_erase bits for n bits erased."""
        agent = make_agent(energy=1000.0, entropy=60.0)
        engine.apply(agent, Dissipate(20.0), ledger, tick=1)

        expected = 20.0 / small_config.physics.erase_efficiency
        assert ledger.env_bits == pytest.approx(expected, rel=1e-6)

    def test_reversible_limit_breaks_even(self, small_config, registry, rng, log,
                                          make_agent):
        """At eta_erase = 1 the books balance exactly -- the Landauer bound itself."""
        import dataclasses

        from btcs.world.grid import World
        from btcs.world.ledger import EntropyLedger
        from btcs.world.physics import PhysicsEngine

        config = dataclasses.replace(
            small_config,
            physics=dataclasses.replace(small_config.physics, erase_efficiency=1.0),
        )
        world = World(config, rng)
        engine = PhysicsEngine(config, world, registry, rng, log)
        ledger = EntropyLedger()
        agent = make_agent(energy=1000.0, entropy=60.0)
        engine.apply(agent, Dissipate(20.0), ledger, tick=1)

        assert ledger.total == pytest.approx(0.0, abs=1e-9)

    def test_erases_only_what_it_can_afford(self, engine, ledger, make_agent):
        """A poor organism stays confused -- it cannot buy its way back to order."""
        agent = make_agent(energy=2.0, entropy=60.0)
        before = agent.entropy
        engine.apply(agent, Dissipate(30.0), ledger, tick=1)

        assert agent.entropy > before - 30.0
        assert agent.energy >= 0.0
        assert ledger.total >= 0.0

    def test_cannot_erase_more_than_it_has(self, engine, ledger, make_agent):
        agent = make_agent(energy=1000.0, entropy=5.0)
        engine.apply(agent, Dissipate(50.0), ledger, tick=1)
        assert agent.entropy == pytest.approx(0.0, abs=1e-9)

    def test_capped_by_max_dissipate_bits(self, small_config, engine, ledger, make_agent):
        agent = make_agent(energy=5000.0, entropy=500.0)
        engine.apply(agent, Dissipate(999.0), ledger, tick=1)
        assert agent.total_dissipated_bits <= small_config.agent.max_dissipate_bits + 1e-9

    def test_nothing_to_erase_is_rejected(self, engine, ledger, make_agent):
        agent = make_agent(entropy=0.0)
        outcome = engine.apply(agent, Dissipate(10.0), ledger, tick=1)
        assert not outcome.ok


class TestCarnot:
    """L2 -- ingestion is bounded by the gradient to the local cell."""

    def test_efficiency_is_zero_at_thermal_equilibrium(self, world, small_config):
        ambient = small_config.world.ambient_temperature
        assert world.carnot_efficiency(ambient, ambient) == 0.0

    def test_efficiency_is_zero_below_equilibrium(self, world):
        assert world.carnot_efficiency(0.5, 1.0) == 0.0

    def test_efficiency_follows_the_bound(self, world, small_config):
        physics = small_config.physics
        expected = (1.0 - 1.0 / 4.0) * physics.ingest_efficiency
        assert world.carnot_efficiency(4.0, 1.0) == pytest.approx(expected)

    def test_efficiency_never_exceeds_the_carnot_bound(self, world):
        for hot in (1.5, 2.0, 4.0, 10.0):
            assert world.carnot_efficiency(hot, 1.0) <= 1.0 - 1.0 / hot

    def test_a_hot_cell_degrades_extraction(self, world):
        """The crowding mechanic: waste heat raises T_cold and shrinks the yield."""
        cool = world.carnot_efficiency(4.0, 1.0)
        warm = world.carnot_efficiency(4.0, 2.5)
        assert warm < cool

    def test_ingestion_nets_non_negative_entropy(self, engine, ledger, make_agent, world):
        """Carnot is a theorem here, not an assertion: the books produce it."""
        world.node_y = np.array([3])
        world.node_x = np.array([3])
        world.node_energy = np.array([500.0])
        world.node_temperature = np.array([6.0])

        agent = make_agent(y=3, x=3, energy=100.0)
        engine.apply(agent, Ingest(50.0), ledger, tick=1)
        assert ledger.total >= -1e-9

    def test_ingestion_at_ideal_efficiency_breaks_even(
        self, small_config, registry, rng, log, make_agent
    ):
        """At the Carnot bound exactly, the books balance to zero.

        Which is the Carnot theorem itself, recovered from bookkeeping alone.
        """
        import dataclasses

        from btcs.world.grid import World
        from btcs.world.ledger import EntropyLedger
        from btcs.world.physics import PhysicsEngine

        config = dataclasses.replace(
            small_config,
            physics=dataclasses.replace(small_config.physics, ingest_efficiency=1.0),
        )
        # World and engine must be built from the same config instance.
        world = World(config, rng)
        world.node_y = np.array([3])
        world.node_x = np.array([3])
        world.node_energy = np.array([500.0])
        world.node_temperature = np.array([6.0])

        engine = PhysicsEngine(config, world, registry, rng, log)
        ledger = EntropyLedger()
        agent = make_agent(y=3, x=3, energy=100.0)
        engine.apply(agent, Ingest(50.0), ledger, tick=1)

        assert ledger.total == pytest.approx(0.0, abs=1e-6)

    def test_eating_a_cold_node_is_rejected(self, engine, ledger, make_agent, world,
                                            small_config):
        world.node_y = np.array([2])
        world.node_x = np.array([2])
        world.node_energy = np.array([500.0])
        world.node_temperature = np.array([small_config.world.ambient_temperature])

        agent = make_agent(y=2, x=2)
        outcome = engine.apply(agent, Ingest(50.0), ledger, tick=1)
        assert not outcome.ok
        assert "equilibrium" in outcome.detail

    def test_ingestion_without_a_node_is_rejected(self, engine, ledger, make_agent, world):
        world.node_y = np.zeros(0, dtype=np.int64)
        world.node_x = np.zeros(0, dtype=np.int64)
        world.node_energy = np.zeros(0)
        world.node_temperature = np.zeros(0)

        outcome = engine.apply(make_agent(), Ingest(10.0), ledger, tick=1)
        assert not outcome.ok

    def test_ingestion_is_rate_limited(self, small_config, engine, ledger, make_agent, world):
        world.node_y = np.array([1])
        world.node_x = np.array([1])
        world.node_energy = np.array([10_000.0])
        world.node_temperature = np.array([8.0])

        agent = make_agent(y=1, x=1, energy=10.0)
        engine.apply(agent, Ingest(99_999.0), ledger, tick=1)
        drawn = 10_000.0 - float(world.node_energy[0])
        assert drawn <= small_config.physics.ingest_rate_limit + 1e-9


class TestCognition:
    """L3 -- tokens become internal entropy, superlinearly in Phi."""

    def test_tokens_generate_entropy(self, engine, ledger, make_agent):
        agent = make_agent(entropy=0.0)
        engine.apply_cognition(agent, 100, ledger, tick=1)
        assert agent.entropy > 0.0
        assert ledger.agent_bits > 0.0

    def test_entropy_is_linear_in_tokens(self, engine, ledger, make_agent):
        a = make_agent(entropy=0.0)
        b = make_agent(entropy=0.0)
        engine.apply_cognition(a, 100, ledger, tick=1)
        engine.apply_cognition(b, 200, ledger, tick=1)
        assert b.entropy == pytest.approx(2.0 * a.entropy)

    def test_higher_phi_costs_superlinearly_more(self, engine, ledger, make_agent):
        """The price of mind. Doubling Phi must more than double the entropy."""
        low = make_agent(entropy=0.0, phi=1.0)
        high = make_agent(entropy=0.0, phi=2.0)
        engine.apply_cognition(low, 100, ledger, tick=1)
        engine.apply_cognition(high, 100, ledger, tick=1)

        ratio = high.entropy / low.entropy
        expected = (3.0 / 2.0) ** engine._config.physics.phi_entropy_exponent
        assert ratio == pytest.approx(expected)
        assert ratio > 1.0

    def test_thinking_costs_energy_as_well(self, engine, ledger, make_agent):
        agent = make_agent(energy=100.0)
        engine.apply_cognition(agent, 100, ledger, tick=1)
        assert agent.energy < 100.0

    def test_dementia_adds_a_token_surcharge(self, small_config, engine, ledger, make_agent):
        """The death spiral in one assertion: disorder makes thinking pricier."""
        lucid = make_agent(entropy=0.0)
        demented = make_agent(entropy=0.0)
        demented.degradation = Degradation.DEMENTED

        engine.apply_cognition(lucid, 100, ledger, tick=1)
        engine.apply_cognition(demented, 100, ledger, tick=1)

        assert demented.total_tokens == lucid.total_tokens + (
            small_config.agent.dementia_injection_tokens
        )
        assert demented.entropy > lucid.entropy


class TestWork:
    """L5 -- moving costs energy and leaves waste heat behind."""

    def test_move_changes_position(self, engine, ledger, make_agent):
        agent = make_agent(y=5, x=5)
        engine.apply(agent, Move(1, 0), ledger, tick=1)
        assert (agent.y, agent.x) == (6, 5)

    def test_move_costs_energy(self, small_config, engine, ledger, make_agent):
        agent = make_agent(y=5, x=5, energy=100.0)
        engine.apply(agent, Move(0, 1), ledger, tick=1)
        assert agent.energy == pytest.approx(100.0 - small_config.physics.work_cost_per_step)

    def test_move_wraps_around_the_torus(self, engine, ledger, make_agent, world):
        agent = make_agent(y=world.height - 1, x=0)
        engine.apply(agent, Move(1, 0), ledger, tick=1)
        assert agent.y == 0

    def test_move_deposits_waste_heat(self, engine, ledger, make_agent, world):
        before = world.temperature.sum()
        agent = make_agent(y=5, x=5, energy=100.0)
        engine.apply(agent, Move(1, 1), ledger, tick=1)
        assert world.temperature.sum() > before

    def test_move_generates_entropy(self, engine, ledger, make_agent):
        engine.apply(make_agent(energy=100.0), Move(1, 0), ledger, tick=1)
        assert ledger.total > 0.0

    def test_move_is_rejected_when_too_poor(self, engine, ledger, make_agent):
        agent = make_agent(energy=0.1)
        outcome = engine.apply(agent, Move(1, 0), ledger, tick=1)
        assert not outcome.ok
        assert agent.energy == pytest.approx(0.1)

    def test_zero_displacement_is_rejected(self, engine, ledger, make_agent):
        assert not engine.apply(make_agent(), Move(0, 0), ledger, tick=1).ok

    def test_rest_is_free(self, engine, ledger, make_agent):
        agent = make_agent(energy=100.0)
        outcome = engine.apply(agent, Rest(), ledger, tick=1)
        assert outcome.ok
        assert agent.energy == pytest.approx(100.0)


class TestFreeEnergyAndDeath:
    """F = E - T*S*ln2, and what happens when it runs out."""

    def test_free_energy_formula(self, make_agent):
        agent = make_agent(energy=100.0, entropy=10.0)
        assert agent.free_energy(2.0) == pytest.approx(100.0 - 2.0 * 10.0 * LN2)

    def test_heat_makes_the_same_organism_lethal(self, make_agent):
        """The property that couples temperature to mortality.

        One organism, unchanged, is comfortably alive in a cold cell and already
        dead in a hot one. Nothing about it changed except where it stands.
        """
        agent = make_agent(energy=50.0, entropy=60.0)
        assert agent.is_viable(temperature=1.0)
        assert not agent.is_viable(temperature=4.0)

    def test_death_when_free_energy_is_gone(self, engine, ledger, make_agent):
        agent = make_agent(energy=0.0, entropy=10.0)
        assert engine.check_death(agent, ledger, tick=5)
        assert not agent.alive
        assert agent.death_tick == 5

    def test_survivor_is_untouched(self, engine, ledger, make_agent):
        agent = make_agent(energy=500.0, entropy=1.0)
        assert not engine.check_death(agent, ledger, tick=5)
        assert agent.alive

    def test_death_releases_entropy_to_the_world(self, engine, ledger, make_agent):
        """Books stay balanced across erasure: what leaves the population arrives
        in the environment, one bit for one bit."""
        agent = make_agent(energy=0.0, entropy=25.0)
        engine.check_death(agent, ledger, tick=5)

        assert ledger.agent_bits == pytest.approx(-25.0)
        assert ledger.env_bits >= 25.0
        assert ledger.total >= -1e-9

    def test_death_cause_is_recorded(self, engine, ledger, make_agent):
        starved = make_agent(energy=0.0, entropy=1.0)
        engine.check_death(starved, ledger, tick=5)
        assert starved.death_cause == "energy_exhausted"

        cooked = make_agent(energy=1.0, entropy=500.0)
        engine.check_death(cooked, ledger, tick=5)
        assert cooked.death_cause == "entropy_overwhelmed"


class TestDegradationLadder:
    """Rungs from lucid to demented, driven purely by accumulated entropy."""

    @pytest.mark.parametrize(
        ("entropy", "expected"),
        [
            (0.0, Degradation.LUCID),
            (39.0, Degradation.LUCID),
            (41.0, Degradation.NOISY),
            (61.0, Degradation.CONFUSED),
            (81.0, Degradation.DEMENTED),
            (999.0, Degradation.DEMENTED),
        ],
    )
    def test_rung_follows_entropy(self, engine, make_agent, entropy, expected):
        agent = make_agent(entropy=entropy)
        engine.resolve_degradation(agent, tick=1)
        assert agent.degradation == expected

    def test_recovery_is_possible(self, engine, make_agent):
        """Dissipation must be able to walk an organism back down the ladder,
        or the mechanic is a trapdoor rather than a gradient."""
        agent = make_agent(entropy=85.0)
        engine.resolve_degradation(agent, tick=1)
        assert agent.degradation == Degradation.DEMENTED

        agent.entropy = 10.0
        engine.resolve_degradation(agent, tick=2)
        assert agent.degradation == Degradation.LUCID


class TestPerception:
    def test_lucid_perception_is_truthful(self, engine, make_agent):
        agent = make_agent(energy=123.0, entropy=5.0)
        percept = engine.perceive(agent, tick=1)
        assert percept.energy == pytest.approx(123.0)
        assert percept.entropy == pytest.approx(5.0)
        assert not percept.degraded

    def test_degraded_perception_is_corrupted(self, engine, make_agent):
        agent = make_agent(energy=123.0, entropy=50.0)
        engine.resolve_degradation(agent, tick=1)
        percept = engine.perceive(agent, tick=1)

        assert percept.degraded
        assert percept.energy != pytest.approx(123.0)

    def test_percept_never_leaks_the_grid(self, engine, make_agent):
        """An organism perceives its neighbourhood, never the world's arrays."""
        percept = engine.perceive(make_agent(), tick=1)
        for value in vars(percept).values() if hasattr(percept, "__dict__") else []:
            assert not isinstance(value, np.ndarray)
        assert isinstance(percept.nodes, tuple)

    def test_vision_is_capped(self, small_config, engine, make_agent):
        percept = engine.perceive(make_agent(y=8, x=8), tick=1)
        assert len(percept.nodes) <= small_config.agent.vision_node_limit
