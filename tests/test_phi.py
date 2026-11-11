"""Phase 4: the memory graph and the Phi-proxy over it.

The measure has to earn trust before any result computed from it means anything,
so most of this file is about what Phi says on inputs whose answer we already
know. The two that matter most:

  * **monotony scores zero.** A set of items always active together carries no
    information about each other. An organism that does the same thing in every
    situation has a memory that is uniform, not integrated, and a measure that
    called that "highly conscious" would be measuring the wrong thing.
  * **independent halves score zero.** Two subsystems that vary without
    reference to each other are fragments, whatever their size.

Everything else -- tiers, hysteresis, the shield -- is plumbing around those two.
"""

from __future__ import annotations

import dataclasses
import random

import pytest

from btcs.config import Config
from btcs.events import EventLog
from btcs.rng import RngService
from btcs.runner import SimulationRunner
from btcs.world.agents import Degradation
from btcs.world.ledger import EntropyLedger
from btcs.world.memory import MemoryGraph


def graph_of(episodes, window=40) -> MemoryGraph:
    graph = MemoryGraph(window)
    for episode in episodes:
        graph.activate(episode)
    return graph


def phi_of(episodes, window=40) -> float:
    return graph_of(episodes, window).phi()


class TestTheMeasure:
    def test_an_empty_memory_is_not_integrated(self):
        assert phi_of([]) == 0.0

    def test_a_single_item_is_not_integrated(self):
        """One element cannot be integrated with anything. Honestly zero rather
        than undefined."""
        assert phi_of([{"a"}] * 10) == 0.0

    def test_monotony_scores_zero(self):
        """The case a naive measure gets backwards.

        Items always active together are perfectly redundant: knowing one tells
        you nothing you did not already know. An organism repeating itself has a
        uniform memory, not an integrated one.
        """
        assert phi_of([{"a", "b", "c"}] * 20) == 0.0

    def test_independent_halves_score_near_zero(self):
        """Fragmentation. Two subsystems varying without reference to each other."""
        rng = random.Random(0)
        episodes = [
            ({"a", "b"} if rng.random() < 0.5 else {"a2", "b2"})
            | ({"c", "d"} if rng.random() < 0.5 else {"c2", "d2"})
            for _ in range(40)
        ]
        assert phi_of(episodes) < 0.1

    def test_coupled_halves_score_high(self):
        """The same items, but now one half determines the other."""
        rng = random.Random(0)
        episodes = []
        for _ in range(40):
            hot = rng.random() < 0.5
            episodes.append(
                ({"a", "b"} if hot else {"a2", "b2"})
                | ({"c", "d"} if hot else {"c2", "d2"})
            )
        assert phi_of(episodes) > 2.0

    def test_partial_coupling_lands_in_between(self):
        rng = random.Random(1)

        def episodes_at(agreement):
            out = []
            for _ in range(60):
                hot = rng.random() < 0.5
                other = hot if rng.random() < agreement else not hot
                out.append(
                    ({"a", "b"} if hot else {"a2", "b2"})
                    | ({"c", "d"} if other else {"c2", "d2"})
                )
            return out

        loose = phi_of(episodes_at(0.6), window=60)
        tight = phi_of(episodes_at(0.95), window=60)
        assert loose < tight

    def test_noise_scores_near_zero(self):
        rng = random.Random(2)
        episodes = [
            {k for k in "abcdefgh" if rng.random() < 0.5} for _ in range(40)
        ]
        assert phi_of(episodes) < 0.3

    def test_the_measure_is_deterministic(self):
        episodes = [{"a", "b"}, {"a", "c"}, {"b", "c"}, {"a"}, {"c"}] * 6
        assert phi_of(episodes) == phi_of(episodes)


class TestTheWindow:
    def test_only_recent_experience_counts(self):
        """A long-dead habit should not be scored as present integration."""
        graph = MemoryGraph(window=10)
        for _ in range(10):
            graph.activate({"a", "b"} if _ % 2 else {"a2", "b2"})
        integrated = graph.phi()

        for _ in range(10):
            graph.activate({"z"})
        assert graph.phi() < integrated

    def test_the_window_bounds_what_is_held(self):
        graph = MemoryGraph(window=5)
        for i in range(50):
            graph.activate({str(i)})
        assert len(graph.history) == 5


class TestCorruption:
    def test_corruption_removes_remembered_detail(self):
        graph = graph_of([{"a", "b", "c", "d"}] * 20)
        lost = graph.corrupt(RngService(1).stream("x"), 0.5)
        assert lost > 0

    def test_corruption_lowers_integration(self):
        """Where the death spiral closes: damage lowers Phi, and a lower Phi
        shields less, so the next tick takes more damage."""
        rng = random.Random(3)
        episodes = []
        for _ in range(40):
            hot = rng.random() < 0.5
            episodes.append(
                ({"a", "b"} if hot else {"a2", "b2"})
                | ({"c", "d"} if hot else {"c2", "d2"})
            )
        graph = graph_of(episodes)
        before = graph.phi()
        for _ in range(4):
            graph.corrupt(RngService(2).stream("x"), 0.4)
        assert graph.phi() < before

    def test_corruption_is_reproducible(self):
        """The bug this test exists for: iterating a `set` follows hash order,
        which Python randomises per process. Drawing from the seeded stream in
        that order made two runs of the same suite disagree about which seeds
        passed -- reproducibility lost to an invisible detail.
        """
        episodes = [{"a", "b", "c", "d", "e"}] * 20
        first = graph_of(episodes)
        second = graph_of(episodes)
        first.corrupt(RngService(9).stream("x"), 0.5)
        second.corrupt(RngService(9).stream("x"), 0.5)
        assert list(first.history) == list(second.history)

    def test_zero_probability_changes_nothing(self):
        graph = graph_of([{"a", "b"}] * 10)
        assert graph.corrupt(RngService(1).stream("x"), 0.0) == 0


class TestWhatPhiBuys:
    """OPEN-11. The point is that this is a knob, not a conclusion."""

    def _engine(self, resilience):
        base = Config()
        config = dataclasses.replace(
            base,
            world=dataclasses.replace(base.world, width=16, height=16, node_count=10),
            phi=dataclasses.replace(base.phi, resilience=resilience),
        )
        from btcs.events import EventLog as Log
        from btcs.world.agents import AgentRegistry
        from btcs.world.grid import World
        from btcs.world.physics import PhysicsEngine
        import tempfile
        from pathlib import Path

        rng = RngService(42)
        world = World(config, rng)
        path = Path(tempfile.mkdtemp()) / "e.jsonl"
        return PhysicsEngine(config, world, AgentRegistry(), rng, Log(path)), config

    def test_integration_reduces_corruption(self, make_agent):
        engine, _ = self._engine(1.0)
        dull = make_agent(phi=0.0)
        integrated = make_agent(phi=3.0)
        assert engine.corruption_probability(integrated) < engine.corruption_probability(dull)
        assert engine.sense_noise(integrated) < engine.sense_noise(dull)

    def test_the_shield_can_be_switched_off(self, make_agent):
        """The control condition: restore the pure-cost regime and watch
        selection drive Phi to the floor."""
        engine, config = self._engine(0.0)
        integrated = make_agent(phi=5.0)
        assert engine.corruption_probability(integrated) == config.agent.edge_prune_probability

    def test_integration_never_grants_immunity(self, make_agent):
        engine, _ = self._engine(2.0)
        assert engine.corruption_probability(make_agent(phi=100.0)) > 0.0


class TestDynamics:
    """Phi is measured from a life, not assigned at birth."""

    @pytest.fixture
    def live(self, small_config, tmp_path):
        runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
        runner._found()
        runner.service.begin_tick(1, EntropyLedger())
        return runner

    def test_a_founder_starts_with_no_integration(self, live):
        assert all(a.phi == 0.0 for a in live.registry.living())

    def test_a_turn_records_an_episode(self, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        with live.service.turn(agent):
            live.service.rest(token)
        graph = live.service.graph_of(agent.id)
        assert len(graph.history) == 1
        assert "act:rest" in graph.history[0]

    def test_a_refusal_is_recorded_too(self, live):
        agent = next(
            a for a in live.registry.living()
            if live.world.node_index_at(a.y, a.x) < 0
        )
        token = live.service.issue_token(agent)
        with live.service.turn(agent):
            with pytest.raises(Exception):
                live.service.ingest_energy(token, 5.0)
        assert "refused" in live.service.graph_of(agent.id).history[0]

    def test_remembering_activates_the_memory(self, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        with live.service.turn(agent):
            live.service.remember(token, "food", "north of here")
        assert "mem:food" in live.service.graph_of(agent.id).history[0]

    def test_every_brain_gets_a_graph_on_the_same_terms(self, small_config, tmp_path):
        """The fairness property. The mock brain never calls `remember`, so a
        graph built only from explicit memories would give it no integration at
        all -- and any comparison between a rule and a language model would be
        about which one used the memory tools, not about which one thought.
        """
        config = dataclasses.replace(
            small_config, run=dataclasses.replace(small_config.run, max_ticks=40)
        )
        with EventLog(tmp_path / "mock.jsonl") as log:
            runner = SimulationRunner(config, log=log)
            runner.run()
        graphs = [runner.service.graph_of(a.id) for a in runner.registry.living()]
        assert graphs and all(len(g.history) > 0 for g in graphs)

    def test_phi_moves_during_a_run(self, small_config, tmp_path):
        from btcs.events import EventType, read_events

        config = dataclasses.replace(
            small_config, run=dataclasses.replace(small_config.run, max_ticks=80)
        )
        path = tmp_path / "run.jsonl"
        with EventLog(path) as log:
            SimulationRunner(config, log=log).run()
        changes = [e for e in read_events(path) if e["type"] == EventType.PHI_CHANGED]
        assert changes, "Phi never moved; it is static again"

    def test_hysteresis_ignores_small_movements(self, small_config, tmp_path, make_agent):
        """A tier swap discards the cached prompt prefix an entire lineage
        shares, so Phi must clear a boundary decisively or not at all (OPEN-8).
        """
        runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
        runner._found()
        agent = runner.registry.living()[0]
        agent.phi = 1.0

        class Nudge:
            items: dict = {}
            history: list = []

            def phi(self, tick, recompute_every=1):
                return 1.0 + small_config.phi.tier_margin / 2.0

        assert runner.engine.update_phi(agent, Nudge(), 5) == 1.0

    def test_phi_can_be_switched_off_entirely(self, small_config, tmp_path):
        config = dataclasses.replace(
            small_config,
            run=dataclasses.replace(small_config.run, max_ticks=30),
            phi=dataclasses.replace(small_config.phi, enabled=False),
        )
        with EventLog(tmp_path / "off.jsonl") as log:
            runner = SimulationRunner(config, log=log)
            runner.run()
        assert all(a.phi == config.agent.initial_phi for a in runner.registry.living())

    def test_a_confused_organism_loses_its_experience(self, small_config, tmp_path):
        runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
        runner._found()
        agent = runner.registry.living()[0]
        graph = runner.service.graph_of(agent.id)
        for _ in range(20):
            graph.activate({"a", "b", "c", "d", "e", "f"})
        held = sum(len(episode) for episode in graph.history)

        agent.entropy = small_config.agent.s_confuse * 2
        agent.degradation = Degradation.CONFUSED
        runner._integrate(agent, tick=3)
        assert sum(len(episode) for episode in graph.history) < held


class TestInheritance:
    def test_a_child_inherits_its_parent_s_integration(self, small_config, tmp_path):
        """Phi is heritable, which is what makes 'does selection favour
        integration?' a question with an answer."""
        runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
        runner._found()
        runner.service.begin_tick(1, EntropyLedger())
        agent = runner.registry.living()[0]
        agent.energy = small_config.selection.replication_energy_threshold * 2
        agent.phi = 2.75
        token = runner.service.issue_token(agent)
        with runner.service.turn(agent):
            child_id = runner.service.replicate(token)["child"]
        assert runner.registry[child_id].phi == 2.75

    def test_a_child_starts_with_no_experience_of_its_own(self, small_config, tmp_path):
        """Integration is inherited as a starting value; the memory that
        justified it is not. A child has to earn its own."""
        runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
        runner._found()
        runner.service.begin_tick(1, EntropyLedger())
        agent = runner.registry.living()[0]
        agent.energy = small_config.selection.replication_energy_threshold * 2
        token = runner.service.issue_token(agent)
        with runner.service.turn(agent):
            child_id = runner.service.replicate(token)["child"]
        assert len(runner.service.graph_of(child_id).history) == 0
