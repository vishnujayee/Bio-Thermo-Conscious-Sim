"""End-to-end guarantees.

These are the Phase 1 exit criteria from construction.md, expressed as tests:

  * a full run completes with dS_total >= 0 on every single tick;
  * the ledger's account of organism entropy matches the organisms themselves;
  * two runs with the same seed produce byte-identical logs.

If any of these fail, nothing the simulation reports afterwards can be trusted.
"""

from __future__ import annotations

import dataclasses

import pytest

from btcs.config import Config
from btcs.events import EventLog, EventType, read_events
from btcs.runner import LedgerInconsistency, SimulationRunner
from btcs.world.ledger import SecondLawViolation


def _run(config: Config, path) -> "RunResult":  # noqa: F821
    with EventLog(path) as log:
        return SimulationRunner(config, log=log).run()


class TestSecondLaw:
    def test_holds_on_every_tick_of_a_full_run(self, small_config, tmp_path):
        """The headline invariant. Not sampled -- every tick, no exceptions."""
        config = dataclasses.replace(
            small_config, run=dataclasses.replace(small_config.run, max_ticks=200)
        )
        result = _run(config, tmp_path / "events.jsonl")

        assert result.ticks_completed > 0
        for report in result.history:
            assert report.ds_total >= -config.physics.second_law_tolerance

    def test_cumulative_entropy_only_rises(self, small_config, tmp_path):
        result = _run(small_config, tmp_path / "events.jsonl")
        curve = [report.cumulative_entropy for report in result.history]
        assert curve == sorted(curve)

    def test_holds_across_many_seeds(self, small_config, tmp_path):
        """A single seed could get lucky; a violation is usually configuration
        dependent, so sweep."""
        for seed in (1, 7, 99, 1234, 20260905):
            config = dataclasses.replace(
                small_config,
                run=dataclasses.replace(small_config.run, seed=seed, max_ticks=80),
            )
            result = _run(config, tmp_path / f"seed{seed}.jsonl")
            assert all(r.ds_total >= -1e-9 for r in result.history)

    def test_holds_under_a_crowded_world(self, small_config, tmp_path):
        """Crowding is where the Carnot cold reservoir matters most: many agents
        heating shared cells is exactly what breaks a naive implementation."""
        config = dataclasses.replace(
            small_config,
            world=dataclasses.replace(small_config.world, width=6, height=6, node_count=4),
            agent=dataclasses.replace(small_config.agent, initial_population=25),
            run=dataclasses.replace(small_config.run, max_ticks=120),
        )
        result = _run(config, tmp_path / "crowded.jsonl")
        assert all(r.ds_total >= -1e-9 for r in result.history)

    def test_violation_is_raised_not_swallowed(self, small_config, tmp_path):
        """Turning off the Landauer price makes forgetting free, which breaks the
        second law. The engine must refuse to continue rather than log a warning."""
        config = dataclasses.replace(
            small_config,
            physics=dataclasses.replace(
                small_config.physics, landauer_enabled=False, basal_entropy_gain=0.0
            ),
            agent=dataclasses.replace(
                small_config.agent,
                initial_entropy=200.0,
                policy=dataclasses.replace(small_config.agent.policy, dissipate_at=1.0),
            ),
            run=dataclasses.replace(small_config.run, max_ticks=40),
        )
        with pytest.raises(SecondLawViolation):
            _run(config, tmp_path / "broken.jsonl")


class TestLedgerConsistency:
    def test_ledger_matches_the_organisms(self, small_config, tmp_path):
        """The stronger invariant. dS_total staying positive proves nothing if a
        code path is mutating agent.entropy without filing an entry -- this is
        what catches that, and it runs on every tick of every test run."""
        config = dataclasses.replace(
            small_config, run=dataclasses.replace(small_config.run, max_ticks=150)
        )
        _run(config, tmp_path / "events.jsonl")  # raises LedgerInconsistency if broken

    def test_inconsistency_is_detected(self, small_config, tmp_path):
        """Prove the check has teeth.

        A subclass that changes entropy without filing an entry is exactly the
        bug this audit exists to catch, and it is the kind of change a future
        phase could introduce by accident. Note that dS_total stays comfortably
        positive throughout -- only the consistency check notices.
        """
        from btcs.world.physics import PhysicsEngine

        class LeakyEngine(PhysicsEngine):
            def apply_basal(self, agent, ledger, tick):
                super().apply_basal(agent, ledger, tick)
                agent.entropy += 5.0  # never filed with the ledger

        from btcs.sim.service import SimulationService

        runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
        runner.engine = LeakyEngine(
            runner.config, runner.world, runner.registry, runner.rng, runner.log
        )
        # The service holds its own engine reference, and the runner reaches the
        # physics only through it, so the swap has to happen in both places.
        runner.service = SimulationService(
            runner.config, runner.world, runner.registry, runner.engine,
            runner.rng, runner.log,
        )
        runner._found()

        with pytest.raises(LedgerInconsistency, match="without filing"):
            runner.step(1)


class TestReproducibility:
    def test_same_seed_gives_byte_identical_logs(self, small_config, tmp_path):
        """The Phase 1 exit criterion, and the foundation of every experiment."""
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        _run(small_config, a)
        _run(small_config, b)
        assert a.read_bytes() == b.read_bytes()

    def test_different_seed_gives_a_different_run(self, small_config, tmp_path):
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        _run(small_config, a)
        _run(dataclasses.replace(
            small_config, run=dataclasses.replace(small_config.run, seed=999)
        ), b)
        assert a.read_bytes() != b.read_bytes()

    def test_results_agree_across_repeats(self, small_config, tmp_path):
        first = _run(small_config, tmp_path / "a.jsonl")
        second = _run(small_config, tmp_path / "b.jsonl")

        assert first.ticks_completed == second.ticks_completed
        assert first.final_population == second.final_population
        assert first.cumulative_entropy == pytest.approx(second.cumulative_entropy)


class TestRunMechanics:
    def test_population_is_founded(self, small_config, tmp_path):
        with EventLog(tmp_path / "e.jsonl") as log:
            runner = SimulationRunner(small_config, log=log)
            runner._found()
            assert runner.registry.population == small_config.agent.initial_population

    def test_founders_are_genetically_identical(self, small_config, tmp_path):
        """Any later spread must be attributable to selection, not to a lucky
        draw at t=0."""
        with EventLog(tmp_path / "e.jsonl") as log:
            runner = SimulationRunner(small_config, log=log)
            runner._found()
            genomes = {a.genome.strategy for a in runner.registry.living()}
            assert len(genomes) == 1

    def test_run_stops_on_extinction(self, small_config, tmp_path):
        config = dataclasses.replace(
            small_config,
            agent=dataclasses.replace(small_config.agent, initial_energy=1.0),
            run=dataclasses.replace(small_config.run, max_ticks=500),
        )
        result = _run(config, tmp_path / "e.jsonl")
        assert result.extinct
        assert result.ticks_completed < 500

    def test_log_records_the_full_config(self, small_config, tmp_path):
        """A run must be self-describing, or it cannot be reproduced later."""
        path = tmp_path / "e.jsonl"
        _run(small_config, path)
        started = next(e for e in read_events(path) if e["type"] == "run_started")
        assert started["config"]["physics"]["erase_efficiency"] == (
            small_config.physics.erase_efficiency
        )
        assert started["seed"] == small_config.run.seed

    def test_log_contains_a_ledger_entry_per_tick(self, small_config, tmp_path):
        path = tmp_path / "e.jsonl"
        result = _run(small_config, path)
        ledgers = [e for e in read_events(path) if e["type"] == EventType.ENTROPY_LEDGER]
        assert len(ledgers) == result.ticks_completed

    def test_deaths_are_logged_with_a_cause(self, small_config, tmp_path):
        config = dataclasses.replace(
            small_config,
            agent=dataclasses.replace(small_config.agent, initial_energy=5.0),
        )
        path = tmp_path / "e.jsonl"
        _run(config, path)
        deaths = [e for e in read_events(path) if e["type"] == EventType.AGENT_DIED]
        assert deaths
        assert all(d["cause"] in {"energy_exhausted", "entropy_overwhelmed"} for d in deaths)

    def test_turn_order_varies_between_ticks(self, small_config, tmp_path):
        """Fixed order would give low-numbered organisms a permanent first-mover
        advantage that would read as selection in the results."""
        with EventLog(tmp_path / "e.jsonl") as log:
            runner = SimulationRunner(small_config, log=log)
            runner._found()
            orders = {tuple(a.id for a in runner._turn_order()) for _ in range(8)}
        assert len(orders) > 1

    def test_on_tick_callback_fires(self, small_config, tmp_path):
        seen: list[int] = []
        with EventLog(tmp_path / "e.jsonl") as log:
            result = SimulationRunner(small_config, log=log).run(
                on_tick=lambda report: seen.append(report.tick)
            )
        assert seen == list(range(1, result.ticks_completed + 1))

    def test_unknown_brain_is_rejected(self, small_config, tmp_path):
        config = dataclasses.replace(
            small_config,
            cognition=dataclasses.replace(small_config.cognition, brain="mock"),
        )
        with EventLog(tmp_path / "e.jsonl") as log:
            runner = SimulationRunner(config, log=log)

            # Config validation catches a bad name first; this reaches past it to
            # the runner's own guard, which is the last line of defence if a
            # brain name is ever added to the config's allow-list without a
            # matching construction branch.
            object.__setattr__(config.cognition, "brain", "telepathy")
            with pytest.raises(ValueError, match="unknown brain"):
                runner._build_brain(config)

    def test_mock_run_spends_nothing(self, small_config, tmp_path):
        """The zero-cost guarantee, asserted rather than assumed."""
        result = _run(small_config, tmp_path / "e.jsonl")
        assert result.spend_usd == 0.0
        assert result.halted_reason is None


class TestShippedDefaults:
    def test_the_default_world_sustains_a_population(self, tmp_path):
        """Guards the Phase 1 tuning.

        If someone changes a physics constant and the population collapses
        immediately, selection has no range to act on and every downstream
        experiment is measuring noise. This test is what makes that a failure
        rather than a surprise three phases later.
        """
        config = Config()
        config = dataclasses.replace(
            config,
            run=dataclasses.replace(config.run, max_ticks=250),
            agent=dataclasses.replace(config.agent, initial_population=16),
        )
        result = _run(config, tmp_path / "default.jsonl")

        assert not result.extinct, "default config drives the population extinct"
        assert result.final_population >= 5
        assert result.ticks_completed == 250
