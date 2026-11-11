"""Phase 5: reproduction, heredity, mutation, and lineage.

This is the file that decides whether the project may use the word *evolution*
about anything it has run. Three things have to hold, and they are separable:

  * **heredity** -- a child resembles its parent;
  * **variation** -- a child is not identical to its parent;
  * **differential survival** -- the difference matters.

The first two are tested here directly. The third lives in
`test_brains.py::TestSelectionSignal`, which measures it.

The accounting tests are the load-bearing ones. Birth is the single most likely
place for the ledger audit to trip, because it is the only event that creates an
organism -- and an organism is a bag of entropy that has to come from somewhere.
"""

from __future__ import annotations

import dataclasses

import pytest

from btcs.config import Config, PolicyParams
from btcs.events import EventLog
from btcs.rng import RngService
from btcs.runner import SimulationRunner
from btcs.sim.service import ErrorCode, ToolError
from btcs.world.agents import Genome, mutate
from btcs.world.ledger import EntropyLedger


@pytest.fixture
def live(small_config, tmp_path):
    runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
    runner._found()
    runner.service.begin_tick(1, EntropyLedger())
    return runner


@pytest.fixture
def rich(live, small_config):
    """One organism wealthy enough to breed, plus its token."""
    agent = live.registry.living()[0]
    agent.energy = small_config.selection.replication_energy_threshold * 2
    return agent, live.service.issue_token(agent)


class TestTheAct:
    def test_a_child_appears(self, live, rich):
        agent, token = rich
        before = live.registry.population
        with live.service.turn(agent):
            result = live.service.replicate(token)
        assert result["ok"]
        assert live.registry.population == before + 1

    def test_the_child_is_the_next_generation(self, live, rich):
        agent, token = rich
        with live.service.turn(agent):
            child_id = live.service.replicate(token)["child"]
        child = live.registry[child_id]
        assert child.generation == agent.generation + 1
        assert child.parent_id == agent.id
        assert child.birth_tick == 1

    def test_the_parent_pays_more_than_the_child_receives(self, live, rich):
        """Replication is not a transfer. Building a body is irreversible work,
        and the difference between what leaves the parent and what arrives in
        the child is the part that became heat."""
        agent, token = rich
        before = agent.energy
        with live.service.turn(agent):
            child = live.registry[live.service.replicate(token)["child"]]
        assert before - agent.energy > child.energy

    def test_replication_counts_as_the_turn_s_one_act(self, live, rich):
        agent, token = rich
        with live.service.turn(agent):
            live.service.replicate(token)
            with pytest.raises(ToolError) as caught:
                live.service.rest(token)
        assert caught.value.code == ErrorCode.ALREADY_ACTED

    def test_the_parent_records_the_outcome_like_any_other_act(self, live, rich):
        agent, token = rich
        with live.service.turn(agent):
            live.service.replicate(token)
        assert agent.last_action.kind == "replicate"
        assert agent.last_action.ok is True

    def test_offspring_are_counted(self, live, rich, small_config):
        agent, token = rich
        agent.energy = small_config.selection.replication_energy_threshold * 6
        for tick in (1, 2):
            live.service.begin_tick(tick, EntropyLedger())
            with live.service.turn(agent):
                live.service.replicate(token)
        assert agent.offspring == 2


class TestRefusals:
    def test_a_poor_organism_is_refused(self, live, small_config):
        agent = live.registry.living()[0]
        agent.energy = small_config.selection.replication_energy_threshold - 1.0
        token = live.service.issue_token(agent)
        with live.service.turn(agent):
            with pytest.raises(ToolError) as caught:
                live.service.replicate(token)
        assert caught.value.code == ErrorCode.REJECTED

    def test_a_refusal_costs_nothing_but_the_turn(self, live, small_config):
        agent = live.registry.living()[0]
        agent.energy = small_config.selection.replication_energy_threshold - 1.0
        token = live.service.issue_token(agent)
        before, population = agent.energy, live.registry.population
        with live.service.turn(agent):
            with pytest.raises(ToolError):
                live.service.replicate(token)
        assert agent.energy == before
        assert live.registry.population == population

    def test_a_full_world_refuses(self, small_config, tmp_path):
        """Carrying capacity, or a successful lineage exhausts memory before it
        exhausts food."""
        config = dataclasses.replace(
            small_config,
            selection=dataclasses.replace(small_config.selection, carrying_capacity=2),
        )
        runner = SimulationRunner(config, log=EventLog(tmp_path / "full.jsonl"))
        runner._found()
        runner.service.begin_tick(1, EntropyLedger())

        agent = runner.registry.living()[0]
        agent.energy = config.selection.replication_energy_threshold * 2
        token = runner.service.issue_token(agent)
        with runner.service.turn(agent):
            with pytest.raises(ToolError, match="full"):
                runner.service.replicate(token)

    def test_the_refusal_is_something_the_organism_can_perceive(self, live,
                                                                small_config):
        agent = live.registry.living()[0]
        agent.energy = small_config.selection.replication_energy_threshold - 1.0
        token = live.service.issue_token(agent)
        with live.service.turn(agent):
            with pytest.raises(ToolError):
                live.service.replicate(token)
        assert agent.last_action.ok is False
        assert "energy" in agent.last_action.detail


class TestIdentity:
    def test_the_child_gets_its_own_token(self, live, rich):
        agent, token = rich
        with live.service.turn(agent):
            child_id = live.service.replicate(token)["child"]
        assert live.service.resolve(live.service._by_agent[child_id]).id == child_id

    def test_the_parent_cannot_act_as_the_child(self, live, rich):
        """S1 again, through the family.

        A parent that could use its own token to act as its child would reopen
        the identity hole from the other side: the population would contain
        entities one organism could drive. The parent's token resolves to the
        parent and to nothing else.
        """
        agent, token = rich
        with live.service.turn(agent):
            child_id = live.service.replicate(token)["child"]
        assert live.service.resolve(token).id == agent.id
        assert live.service.resolve(token).id != child_id

    def test_the_child_cannot_act_during_its_parent_s_turn(self, live, rich):
        """Birth does not grant a free extra turn to the family."""
        agent, token = rich
        with live.service.turn(agent):
            child_id = live.service.replicate(token)["child"]
            child_token = live.service._by_agent[child_id]
            with pytest.raises(ToolError) as caught:
                live.service.rest(child_token)
        assert caught.value.code == ErrorCode.NOT_YOUR_TURN


class TestHeredity:
    def test_the_child_inherits_the_parent_s_strategy(self, live, rich):
        agent, token = rich
        agent.genome = Genome("hoard everything and never move", agent.genome.policy)
        with live.service.turn(agent):
            child = live.registry[live.service.replicate(token)["child"]]
        # Point mutations rearrange words; they never introduce new ones, so
        # every word of the child came from the parent.
        assert set(child.genome.strategy.split()) <= set(agent.genome.strategy.split())

    def test_the_child_inherits_disorder_and_the_parent_is_relieved_of_it(
        self, live, rich, small_config
    ):
        """A division, not a copy. An earlier version created the child's share
        out of nothing; the resulting ratchet drove every lineage extinct within
        four generations while the world was still full of food."""
        agent, token = rich
        agent.entropy = 40.0
        share = small_config.selection.child_entropy_fraction

        with live.service.turn(agent):
            child = live.registry[live.service.replicate(token)["child"]]

        assert child.entropy == pytest.approx(40.0 * share)
        assert agent.entropy == pytest.approx(40.0 * (1.0 - share))
        assert child.entropy + agent.entropy == pytest.approx(40.0)

    def test_phi_is_inherited(self, live, rich):
        agent, token = rich
        agent.phi = 3.5
        with live.service.turn(agent):
            child = live.registry[live.service.replicate(token)["child"]]
        assert child.phi == 3.5


class TestMutation:
    def test_prose_drifts_over_generations(self, small_config):
        stream = RngService(7).stream("m")
        genome = Genome("Prioritise resource sharing.", PolicyParams())
        for _ in range(40):
            genome = mutate(genome, stream, small_config)
        assert genome.strategy != "Prioritise resource sharing."

    def test_mutation_never_invents_a_word(self, small_config):
        """The safety property that closes S2 by construction.

        Mutation deletes, duplicates and transposes words the ancestor already
        had. No new token can appear, so no injected instruction can arise in a
        genome that did not start with one -- and agent-authored text can never
        become an instruction we did not write.
        """
        stream = RngService(3).stream("m")
        original = "Prioritise resource sharing and avoid hot cells."
        vocabulary = set(original.split())
        genome = Genome(original, PolicyParams())
        for _ in range(200):
            genome = mutate(genome, stream, small_config)
            assert set(genome.strategy.split()) <= vocabulary

    def test_a_genome_cannot_grow_without_limit(self, small_config):
        """Duplication makes unbounded growth reachable, and a lineage that
        grows its own prompt is a budget problem before it is anything else."""
        config = dataclasses.replace(
            small_config,
            selection=dataclasses.replace(small_config.selection, mutation_rate=0.9),
        )
        stream = RngService(11).stream("m")
        genome = Genome("grow " * 40, PolicyParams())
        for _ in range(60):
            genome = mutate(genome, stream, config)
            assert len(genome.strategy) <= config.selection.genome_max_chars

    def test_a_lineage_that_deletes_itself_keeps_its_parent_s_text(self, small_config):
        """An empty genome is not a strategy, and would make the prompt malformed."""
        config = dataclasses.replace(
            small_config,
            selection=dataclasses.replace(small_config.selection, mutation_rate=1.0),
        )
        stream = RngService(5).stream("m")
        genome = Genome("one", PolicyParams())
        for _ in range(50):
            genome = mutate(genome, stream, config)
            assert genome.strategy.strip()

    def test_policy_numbers_drift(self, small_config):
        stream = RngService(13).stream("m")
        genome = mutate(Genome("x", PolicyParams()), stream, small_config)
        assert genome.policy.dissipate_at != PolicyParams().dissipate_at

    def test_policy_numbers_stay_usable(self, small_config):
        """Clamping is not a judgement about which strategies are allowed. It
        keeps a mutant from asking the engine for a negative appetite, which
        would be a crash rather than a phenotype."""
        config = dataclasses.replace(
            small_config,
            selection=dataclasses.replace(small_config.selection, mutation_rate=0.9),
        )
        stream = RngService(17).stream("m")
        genome = Genome("x", PolicyParams())
        for _ in range(300):
            genome = mutate(genome, stream, config)
            policy = genome.policy
            assert policy.dissipate_at > 0.0
            assert 0.0 < policy.dissipate_fraction <= 1.0
            assert 0.0 <= policy.explore_bias <= 1.0
            assert policy.greed >= 0.0
            assert policy.replicate_at > 0.0

    def test_mutation_is_reproducible(self, small_config):
        genome = Genome("Prioritise resource sharing.", PolicyParams())
        a = mutate(genome, RngService(42).stream("m"), small_config)
        b = mutate(genome, RngService(42).stream("m"), small_config)
        assert a == b


class TestAccounting:
    """Birth is where the ledger audit is most likely to trip."""

    def test_the_books_balance_across_a_birth(self, live, rich):
        ledger = EntropyLedger()
        live.service.begin_tick(1, ledger)
        agent, token = rich
        agent.entropy = 40.0

        recorded_before = ledger.agent_bits
        measured_before = live.registry.total_entropy()
        with live.service.turn(agent):
            live.service.replicate(token)

        measured = live.registry.total_entropy() - measured_before
        recorded = ledger.agent_bits - recorded_before
        assert measured == pytest.approx(recorded, abs=1e-9)

    def test_a_birth_does_not_decrease_total_entropy(self, live, rich):
        """Making a new ordered thing has to be paid for somewhere."""
        ledger = EntropyLedger()
        live.service.begin_tick(1, ledger)
        agent, token = rich
        with live.service.turn(agent):
            live.service.replicate(token)
        assert ledger.total >= 0.0

    def test_energy_is_conserved_apart_from_what_became_heat(self, live, rich):
        agent, token = rich
        before = agent.energy
        with live.service.turn(agent):
            child = live.registry[live.service.replicate(token)["child"]]

        spent = before - agent.energy - child.energy
        assert spent == pytest.approx(live.config.selection.replication_cost, abs=1e-6)

    def test_a_full_run_keeps_its_invariants(self, small_config, tmp_path):
        """The real check: hundreds of births under both audits, no exceptions."""
        config = dataclasses.replace(
            small_config,
            run=dataclasses.replace(small_config.run, max_ticks=400),
            agent=dataclasses.replace(small_config.agent, initial_population=12),
        )
        with EventLog(tmp_path / "long.jsonl") as log:
            result = SimulationRunner(config, log=log).run()
        assert result.cumulative_entropy > 0.0


class TestLineage:
    def _run(self, config, path):
        import json

        with EventLog(path) as log:
            SimulationRunner(config, log=log).run()
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_a_default_run_produces_descendants(self, tmp_path):
        """The claim the project could not make before Phase 5."""
        base = Config()
        config = dataclasses.replace(
            base,
            run=dataclasses.replace(base.run, max_ticks=400),
            agent=dataclasses.replace(base.agent, initial_population=16),
        )
        events = self._run(config, tmp_path / "lineage.jsonl")
        births = [e for e in events if e["type"] == "replication"]
        assert births, "no organism ever reproduced"
        assert max(e["generation"] for e in births) >= 3

    def test_every_child_names_a_parent_that_existed(self, tmp_path):
        base = Config()
        config = dataclasses.replace(
            base,
            run=dataclasses.replace(base.run, max_ticks=200),
            agent=dataclasses.replace(base.agent, initial_population=16),
        )
        events = self._run(config, tmp_path / "tree.jsonl")
        born = {e["agent"] for e in events if e["type"] == "agent_born"}
        for event in events:
            if event["type"] == "agent_born" and event.get("parent"):
                assert event["parent"] in born

    def test_the_run_stays_reproducible_with_reproduction(self, tmp_path):
        """Mutation draws from the seeded RNG, so two identical runs stay
        identical -- reproduction must not be the thing that breaks replay."""
        base = Config()
        config = dataclasses.replace(
            base,
            run=dataclasses.replace(base.run, max_ticks=120),
            agent=dataclasses.replace(base.agent, initial_population=12),
        )
        first = self._run(config, tmp_path / "a.jsonl")
        second = self._run(config, tmp_path / "b.jsonl")
        assert first == second
