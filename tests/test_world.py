"""The environment: fields, nodes, diffusion, and the boundary with the sink."""

from __future__ import annotations

import numpy as np
import pytest

from btcs.world.ledger import EntropyLedger


class TestGeometry:
    def test_grid_starts_at_ambient(self, world, small_config):
        assert np.allclose(world.temperature, small_config.world.ambient_temperature)

    def test_wrap_is_toroidal(self, world):
        assert world.wrap(world.height, world.width) == (0, 0)
        assert world.wrap(-1, -1) == (world.height - 1, world.width - 1)

    def test_toroidal_delta_takes_the_short_way(self, world):
        """Distance across the seam must be short, or the torus is decorative."""
        dy, dx = world.toroidal_delta(0, 0, world.height - 1, 0)
        assert dy == -1
        assert dx == 0

    def test_toroidal_delta_is_antisymmetric(self, world):
        # Deliberately not exactly half the grid apart: at the antipode, +w/2 and
        # -w/2 are equally short and the tie-break makes antisymmetry fail
        # legitimately.
        forward = world.toroidal_delta(2, 3, 9, 7)
        backward = world.toroidal_delta(9, 7, 2, 3)
        assert forward == (-backward[0], -backward[1])


class TestNodes:
    def test_initial_node_count(self, world, small_config):
        assert world.node_count == small_config.world.node_count

    def test_nodes_start_hotter_than_ambient(self, world, small_config):
        """A node at ambient carries no extractable free energy at all."""
        assert (world.node_temperature > small_config.world.ambient_temperature).all()

    def test_nodes_start_with_positive_energy(self, world):
        assert (world.node_energy > 0).all()

    def test_drawing_reduces_stock(self, world):
        before = float(world.node_energy[0])
        drawn = world.draw_from_node(0, 10.0)
        assert drawn == pytest.approx(10.0)
        assert float(world.node_energy[0]) == pytest.approx(before - 10.0)

    def test_drawing_cannot_overdraw(self, world):
        world.node_energy[0] = 5.0
        assert world.draw_from_node(0, 100.0) == pytest.approx(5.0)
        assert float(world.node_energy[0]) == pytest.approx(0.0)

    def test_node_index_at_finds_nothing_on_empty_ground(self, world):
        world.node_y = np.zeros(0, dtype=np.int64)
        world.node_x = np.zeros(0, dtype=np.int64)
        world.node_energy = np.zeros(0)
        world.node_temperature = np.zeros(0)
        assert world.node_index_at(0, 0) == -1

    def test_nodes_near_respects_radius(self, world):
        found = world.nodes_near(0, 0, radius=1, limit=99)
        for node in found:
            assert abs(node.dy) <= 1 and abs(node.dx) <= 1

    def test_nodes_near_is_ranked_by_extractable_not_stock(self, world):
        """A cold node full of energy is not food, and must not rank as such."""
        world.node_y = np.array([0, 1])
        world.node_x = np.array([0, 1])
        world.node_energy = np.array([1000.0, 100.0])   # first is richer...
        world.node_temperature = np.array([1.01, 8.0])  # ...but nearly at ambient

        found = world.nodes_near(0, 0, radius=5, limit=5)
        assert found[0].temperature == pytest.approx(8.0)

    def test_nodes_near_honours_the_limit(self, world):
        assert len(world.nodes_near(8, 8, radius=99, limit=3)) <= 3

    def test_exhausted_nodes_are_removed(self, world, small_config, log):
        world.node_energy[:] = 0.0
        world._remove_exhausted(tick=1, log=log)
        assert world.node_count == 0


class TestThermalField:
    def test_diffusion_conserves_field_energy(self, world, small_config):
        """The discrete Laplacian sums to zero, so diffusion moves heat without
        creating or destroying it. If this fails, every energy figure is wrong."""
        world.temperature[5, 5] += 50.0
        before = world.temperature.sum()

        world._diffuse(EntropyLedger())
        assert world.temperature.sum() == pytest.approx(before, rel=1e-12)

    def test_diffusion_smooths_a_hotspot(self, world):
        world.temperature[5, 5] += 50.0
        peak = world.temperature[5, 5]
        world._diffuse(EntropyLedger())
        assert world.temperature[5, 5] < peak
        assert world.temperature[5, 6] > world._config.world.ambient_temperature

    def test_diffusion_generates_entropy(self, world):
        """Heat flowing hot to cold must always raise the world's entropy."""
        world.temperature[5, 5] += 50.0
        ledger = EntropyLedger()
        world._diffuse(ledger)
        assert ledger.env_bits > 0.0

    def test_uniform_field_diffuses_to_nothing(self, world):
        ledger = EntropyLedger()
        before = world.temperature.copy()
        world._diffuse(ledger)
        assert np.allclose(world.temperature, before)
        assert ledger.env_bits == pytest.approx(0.0, abs=1e-12)

    def test_radiation_cools_toward_ambient(self, world, small_config):
        world.temperature[3, 3] = 10.0
        world._radiate_to_sink(EntropyLedger())
        assert world.temperature[3, 3] < 10.0
        assert world.temperature[3, 3] > small_config.world.ambient_temperature

    def test_radiation_generates_entropy(self, world):
        world.temperature[3, 3] = 10.0
        ledger = EntropyLedger()
        world._radiate_to_sink(ledger)
        assert ledger.env_bits > 0.0

    def test_radiation_removes_energy_from_the_world(self, world):
        """This is the open-system boundary: energy genuinely leaves."""
        world.temperature[3, 3] = 10.0
        before = world.total_field_energy()
        world._radiate_to_sink(EntropyLedger())
        assert world.total_field_energy() < before

    def test_deposit_heat_warms_one_cell_and_bills_it(self, world):
        ledger = EntropyLedger()
        before = float(world.temperature[2, 2])
        world.deposit_heat(2, 2, 100.0, ledger, "test")

        assert float(world.temperature[2, 2]) > before
        assert ledger.env_bits > 0.0

    def test_deposit_heat_ignores_nothing(self, world):
        ledger = EntropyLedger()
        world.deposit_heat(2, 2, 0.0, ledger, "test")
        assert ledger.env_bits == 0.0

    def test_temperature_never_falls_below_the_floor(self, world, small_config, log):
        for tick in range(20):
            world.step(tick, EntropyLedger(), log)
        assert (world.temperature >= small_config.physics.min_temperature).all()


class TestWorldStep:
    def test_nodes_cool_toward_ambient(self, world, small_config, log):
        import dataclasses

        # Respawn off: a new node appearing mid-step changes the array length.
        world._config = dataclasses.replace(
            small_config,
            world=dataclasses.replace(small_config.world, node_respawn_rate=0.0),
        )
        before = world.node_temperature.copy()
        world.step(1, EntropyLedger(), log)
        assert (world.node_temperature <= before + 1e-12).all()

    def test_extractable_energy_decays_without_agents(self, world, small_config, log):
        """The world runs down on its own. Nobody has to eat for this to happen."""
        import dataclasses

        world._config = dataclasses.replace(
            small_config,
            world=dataclasses.replace(small_config.world, node_respawn_rate=0.0,
                                      node_cooling_rate=0.05),
        )
        before = world.total_extractable_energy()
        for tick in range(50):
            world.step(tick, EntropyLedger(), log)
        assert world.total_extractable_energy() < before

    def test_step_only_ever_adds_entropy(self, world, log):
        world.temperature[4, 4] += 30.0
        for tick in range(25):
            ledger = EntropyLedger()
            world.step(tick, ledger, log)
            assert ledger.total >= -1e-9

    def test_energy_is_never_created_without_respawn(self, world, small_config, log):
        """With the inflow switched off, total energy must be non-increasing.

        This is the check that would catch a sign error in diffusion or in the
        sink coupling -- either of which would otherwise show up much later as
        an inexplicably immortal population.
        """
        import dataclasses

        world._config = dataclasses.replace(
            small_config,
            world=dataclasses.replace(small_config.world, node_respawn_rate=0.0),
        )
        total = lambda: world.total_field_energy() + world.total_node_energy()

        previous = total()
        for tick in range(40):
            world.step(tick, EntropyLedger(), log)
            current = total()
            assert current <= previous + 1e-6
            previous = current

    def test_snapshot_reports_the_expected_keys(self, world):
        snapshot = world.snapshot()
        assert {"nodes", "node_energy", "extractable_energy",
                "field_energy", "mean_temperature"} <= set(snapshot)
