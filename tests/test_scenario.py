"""Phase 6: making the world change under the organisms.

Scenarios are not dashboard scaffolding. Phase 4 found that integration does not
pay for itself, and the strongest remaining explanation was that a good map is
worth little where the right answer never changes -- so this module is the
apparatus for the experiment that could have overturned the headline result.
(It did not. See `docs/phi.md`.)

The load-bearing test here is `test_world_and_engine_never_diverge`. A config
override that reaches the world but not the engine means the engine charges one
price while the world applies another, and the entropy ledger stops balancing in
a way that is genuinely hard to trace.
"""

from __future__ import annotations

import dataclasses

import pytest

from btcs.config import Config, ConfigError, ScenarioEventConfig
from btcs.events import EventLog, EventType, read_events
from btcs.runner import SimulationRunner
from btcs.sim.scenario import ACTIONS, PRESETS, apply_overrides, build, preset


class TestOverrides:
    def test_a_dotted_path_replaces_one_value(self, small_config):
        updated = apply_overrides(small_config, {"world.ambient_temperature": 2.0})
        assert updated.world.ambient_temperature == 2.0

    def test_everything_else_is_left_alone(self, small_config):
        updated = apply_overrides(small_config, {"world.ambient_temperature": 2.0})
        assert updated.physics == small_config.physics
        assert updated.world.width == small_config.world.width

    def test_several_sections_at_once(self, small_config):
        updated = apply_overrides(small_config, {
            "world.node_respawn_rate": 0.1,
            "physics.ingest_efficiency": 0.5,
        })
        assert updated.world.node_respawn_rate == 0.1
        assert updated.physics.ingest_efficiency == 0.5

    def test_the_original_is_untouched(self, small_config):
        before = small_config.world.ambient_temperature
        apply_overrides(small_config, {"world.ambient_temperature": 9.0})
        assert small_config.world.ambient_temperature == before

    @pytest.mark.parametrize("path", ["ambient_temperature", "wrold.x", "world.nonsense"])
    def test_a_bad_path_is_refused(self, small_config, path):
        with pytest.raises(ConfigError):
            apply_overrides(small_config, {path: 1.0})

    def test_a_non_finite_value_is_refused(self, small_config):
        """A NaN in the temperature field poisons every later tick and makes the
        second-law assertion pass vacuously, because comparisons against NaN are
        false."""
        with pytest.raises(ConfigError, match="finite"):
            apply_overrides(small_config, {"world.ambient_temperature": float("nan")})


class TestPresets:
    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_every_preset_builds(self, name):
        scenario = preset(name, ticks=400)
        assert scenario.name == name

    def test_an_unknown_preset_is_refused(self):
        with pytest.raises(ConfigError, match="unknown scenario"):
            preset("apocalypse", ticks=100)

    def test_presets_scale_to_the_length_of_the_run(self):
        short = preset("heat_wave", ticks=100)
        long = preset("heat_wave", ticks=1000)
        assert short.last_tick < long.last_tick

    def test_every_event_does_something(self):
        for name in sorted(PRESETS):
            for event in preset(name, ticks=400).events:
                assert event.set or event.do

    def test_every_named_action_exists(self):
        for name in sorted(PRESETS):
            for event in preset(name, ticks=400).events:
                assert event.do is None or event.do in ACTIONS

    def test_seasons_alternates(self):
        """The scenario built for OPEN-19: the right answer this season has to
        become the wrong one next season, or a map is never worth holding."""
        notes = [e.note for e in preset("seasons", ticks=400).events]
        assert "winter" in notes and "summer" in notes
        assert notes[0] != notes[1]


class TestConfigIntegration:
    def test_an_unknown_scenario_name_fails_at_load(self, small_config):
        """Not at tick 300. A run that dies four minutes in because of a typo
        has wasted four minutes."""
        config = dataclasses.replace(
            small_config,
            scenario=dataclasses.replace(small_config.scenario, name="apocalypse"),
        )
        with pytest.raises(ConfigError, match="scenario.name"):
            config.validate()

    def test_an_unknown_action_fails_at_load(self, small_config):
        config = dataclasses.replace(
            small_config,
            scenario=dataclasses.replace(
                small_config.scenario,
                events=[ScenarioEventConfig(at=5, do="summon_meteor")],
            ),
        )
        with pytest.raises(ConfigError, match="unknown"):
            config.validate()

    def test_an_event_that_does_nothing_fails_at_load(self, small_config):
        config = dataclasses.replace(
            small_config,
            scenario=dataclasses.replace(
                small_config.scenario, events=[ScenarioEventConfig(at=5)]
            ),
        )
        with pytest.raises(ConfigError, match="does nothing"):
            config.validate()

    def test_explicit_events_beat_the_preset_name(self, small_config):
        config = dataclasses.replace(
            small_config,
            scenario=dataclasses.replace(
                small_config.scenario,
                name="mine",
                events=[ScenarioEventConfig(at=3, set={"world.node_respawn_rate": 0.1})],
            ),
        )
        scenario = build(config)
        assert scenario.name == "mine"
        assert len(scenario.events) == 1


class TestInTheRunner:
    def _run(self, small_config, tmp_path, events=None, name="none", ticks=30):
        config = dataclasses.replace(
            small_config,
            run=dataclasses.replace(small_config.run, max_ticks=ticks),
            scenario=dataclasses.replace(
                small_config.scenario, name=name, events=events or []
            ),
        )
        path = tmp_path / "e.jsonl"
        with EventLog(path) as log:
            runner = SimulationRunner(config, log=log)
            runner.run()
        return runner, list(read_events(path))

    def test_an_override_reaches_the_world(self, small_config, tmp_path):
        runner, _ = self._run(
            small_config, tmp_path,
            events=[ScenarioEventConfig(at=5, set={"world.ambient_temperature": 2.5})],
        )
        assert runner.config.world.ambient_temperature == 2.5

    def test_world_and_engine_never_diverge(self, small_config, tmp_path):
        """The invariant that makes an override safe at all.

        Two `Config` instances means the engine charges one price while the world
        applies another, and the ledger stops balancing in a way that is very
        hard to trace back to here.
        """
        runner, _ = self._run(
            small_config, tmp_path,
            events=[ScenarioEventConfig(at=4, set={"physics.ingest_efficiency": 0.4})],
        )
        assert runner.world.config is runner.config
        assert runner.engine._config is runner.config
        assert runner.service._config is runner.config

    def test_the_intervention_is_logged(self, small_config, tmp_path):
        _, events = self._run(
            small_config, tmp_path,
            events=[ScenarioEventConfig(at=6, do="halve_nodes", note="a lean year")],
        )
        overrides = [e for e in events if e["type"] == EventType.SCENARIO_OVERRIDE]
        assert len(overrides) == 1
        assert overrides[0]["tick"] == 6
        assert overrides[0]["note"] == "a lean year"

    def test_a_world_action_changes_the_world(self, small_config, tmp_path):
        # Respawn off: a node that appears after the event is born hot, and the
        # assertion would be about the respawn rule rather than about `chill`.
        quiet = dataclasses.replace(
            small_config,
            world=dataclasses.replace(small_config.world, node_respawn_rate=0.0),
        )
        config = dataclasses.replace(
            quiet,
            run=dataclasses.replace(quiet.run, max_ticks=3),
            scenario=dataclasses.replace(
                quiet.scenario, events=[ScenarioEventConfig(at=3, do="chill")]
            ),
        )
        with EventLog(tmp_path / "chill.jsonl") as log:
            runner = SimulationRunner(config, log=log)
            runner._found()
            runner.step(1)
            runner.step(3)
        ambient = runner.config.world.ambient_temperature
        assert all(t <= ambient + 1e-9 for t in runner.world.node_temperature)

    def test_scatter_moves_the_food(self, small_config, tmp_path):
        """What makes an organism's map wrong, which is the point of seasons."""
        # Respawn off, or the node arrays change length mid-test and the
        # comparison is about a new node rather than about the scatter.
        quiet = dataclasses.replace(
            small_config,
            world=dataclasses.replace(small_config.world, node_respawn_rate=0.0),
        )
        config = dataclasses.replace(
            quiet,
            run=dataclasses.replace(quiet.run, max_ticks=6),
            scenario=dataclasses.replace(
                quiet.scenario,
                events=[ScenarioEventConfig(at=5, do="scatter")],
            ),
        )
        with EventLog(tmp_path / "a.jsonl") as log:
            runner = SimulationRunner(config, log=log)
            runner._found()
            for tick in range(1, 5):
                runner.step(tick)
            before = runner.world.node_y.copy()
            runner.step(5)
        assert not (runner.world.node_y == before).all()

    def test_the_invariants_survive_every_preset(self, small_config, tmp_path):
        """Both audits run every tick. A scenario that broke the second law would
        make every result after it meaningless."""
        for name in sorted(PRESETS):
            runner, _ = self._run(small_config, tmp_path / name, name=name, ticks=60)
            assert runner.config.world.width == small_config.world.width

    def test_a_scenario_run_stays_reproducible(self, small_config, tmp_path):
        """World actions draw from the seeded RNG, so they must not be the thing
        that breaks replay."""
        first = self._run(small_config, tmp_path / "a", name="seasons", ticks=60)[1]
        second = self._run(small_config, tmp_path / "b", name="seasons", ticks=60)[1]
        assert first == second

    def test_no_scenario_means_no_interventions(self, small_config, tmp_path):
        _, events = self._run(small_config, tmp_path, name="none")
        assert not [e for e in events if e["type"] == EventType.SCENARIO_OVERRIDE]


class TestPrediction:
    """OPEN-18: being wrong about your own world costs entropy (Still et al. 2012)."""

    def test_a_predictable_world_is_barely_surprising(self):
        from btcs.world.memory import MemoryGraph

        graph = MemoryGraph(24)
        for index in range(24):
            graph.activate(frozenset({"a", "b"} if index % 2 else {"c", "d"}))
        expected = graph.surprisal(frozenset({"a", "b"}), frozenset({"c", "d"}))
        confounded = graph.surprisal(frozenset({"a", "b"}), frozenset({"a", "b"}))
        assert expected < confounded

    def test_an_unpredictable_world_is_surprising(self):
        import random

        from btcs.world.memory import MemoryGraph

        rng = random.Random(0)
        ordered = MemoryGraph(24)
        noisy = MemoryGraph(24)
        for index in range(24):
            ordered.activate(frozenset({"a", "b"} if index % 2 else {"c", "d"}))
            noisy.activate(frozenset(k for k in "abcd" if rng.random() < 0.5))
        following = frozenset({"c", "d"})
        assert ordered.surprisal(frozenset({"a", "b"}), following) < noisy.surprisal(
            frozenset({"a", "b"}), following
        )

    def test_a_memory_with_no_history_is_never_surprised(self):
        from btcs.world.memory import MemoryGraph

        assert MemoryGraph(24).surprisal(frozenset(), frozenset({"a"})) == 0.0

    def test_surprise_is_charged_as_entropy(self, small_config, tmp_path):
        config = dataclasses.replace(
            small_config,
            run=dataclasses.replace(small_config.run, max_ticks=40),
            phi=dataclasses.replace(small_config.phi, prediction_weight=1.0),
        )
        path = tmp_path / "p.jsonl"
        with EventLog(path) as log:
            SimulationRunner(config, log=log).run()
        charges = [e for e in read_events(path) if e["type"] == EventType.SURPRISE]
        assert charges
        assert all(e["entropy_bits"] > 0 for e in charges)

    def test_it_is_off_by_default(self, small_config, tmp_path):
        """The mechanism was measured and not supported, so it does not ship on.
        See docs/phi.md."""
        assert Config().phi.prediction_weight == 0.0
        path = tmp_path / "off.jsonl"
        with EventLog(path) as log:
            SimulationRunner(
                dataclasses.replace(
                    small_config,
                    run=dataclasses.replace(small_config.run, max_ticks=20),
                ),
                log=log,
            ).run()
        assert not [e for e in read_events(path) if e["type"] == EventType.SURPRISE]

    def test_charging_surprise_keeps_the_books(self, small_config, tmp_path):
        """The ledger audit runs every tick; a charge that skipped an entry would
        trip it. This asserts the run completes, which is the audit passing."""
        config = dataclasses.replace(
            small_config,
            run=dataclasses.replace(small_config.run, max_ticks=60),
            phi=dataclasses.replace(small_config.phi, prediction_weight=2.0),
        )
        with EventLog(tmp_path / "b.jsonl") as log:
            result = SimulationRunner(config, log=log).run()
        assert result.cumulative_entropy > 0.0
