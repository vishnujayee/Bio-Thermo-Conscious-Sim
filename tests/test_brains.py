"""The mock brain: determinism, and that its policy actually reflects the genome."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from btcs.brains.base import Brain
from btcs.brains.mock import MockBrain
from btcs.config import PolicyParams
from btcs.rng import RngService
from btcs.world.agents import Genome
from btcs.world.types import Dissipate, Ingest, Move, Replicate, Rest


@pytest.fixture
def brain(small_config, rng) -> MockBrain:
    return MockBrain(small_config, rng)


def test_satisfies_the_brain_protocol(brain):
    assert isinstance(brain, Brain)


def test_is_deterministic(small_config, engine, make_agent):
    """Same seed, same percept, same decision -- otherwise runs cannot repeat."""
    agent = make_agent(y=4, x=4)
    percept = engine.perceive(agent, tick=1)

    a = MockBrain(small_config, RngService(42)).decide(percept, agent)
    b = MockBrain(small_config, RngService(42)).decide(percept, agent)
    assert a.action == b.action
    assert a.tokens == b.tokens


def test_reports_a_positive_token_cost(brain, engine, make_agent):
    """Tokens are the bridge to thermodynamics; a free thought would be a cheat."""
    percept = engine.perceive(make_agent(), tick=1)
    assert brain.decide(percept, make_agent()).tokens > 0


def test_token_cost_grows_with_what_it_looks_at(small_config, brain, engine,
                                                make_agent, world):
    """A wide-sensing strategy is charged more entropy than a narrow one."""
    world.node_y = np.array([1, 2, 3, 4])
    world.node_x = np.array([1, 2, 3, 4])
    world.node_energy = np.full(4, 300.0)
    world.node_temperature = np.full(4, 6.0)

    crowded = engine.perceive(make_agent(y=2, x=2), tick=1)

    world.node_y = np.zeros(0, dtype=np.int64)
    world.node_x = np.zeros(0, dtype=np.int64)
    world.node_energy = np.zeros(0)
    world.node_temperature = np.zeros(0)
    empty = engine.perceive(make_agent(y=2, x=2), tick=1)

    agent = make_agent()
    assert brain.decide(crowded, agent).tokens > brain.decide(empty, agent).tokens


class TestPolicy:
    def test_eats_when_standing_on_food(self, brain, engine, make_agent, world):
        world.node_y = np.array([5])
        world.node_x = np.array([5])
        world.node_energy = np.array([300.0])
        world.node_temperature = np.array([6.0])

        agent = make_agent(y=5, x=5, energy=50.0)
        decision = brain.decide(engine.perceive(agent, tick=1), agent)
        assert isinstance(decision.action, Ingest)

    def test_dissipates_when_entropy_passes_the_genome_threshold(
        self, brain, engine, make_agent
    ):
        agent = make_agent(energy=800.0, entropy=5.0)
        assert not isinstance(
            brain.decide(engine.perceive(agent, tick=1), agent).action, Dissipate
        )

        loaded = make_agent(energy=800.0, entropy=70.0)
        loaded.degradation = 0  # keep perception truthful for a clean assertion
        assert isinstance(
            brain.decide(engine.perceive(loaded, tick=1), loaded).action, Dissipate
        )

    def test_the_threshold_is_genetic(self, small_config, brain, engine, make_agent):
        """Selection acts on this number, so the policy must actually read it."""
        eager = make_agent(energy=800.0, entropy=30.0)
        eager.genome = Genome("eager", PolicyParams(dissipate_at=10.0))
        assert isinstance(
            brain.decide(engine.perceive(eager, tick=1), eager).action, Dissipate
        )

        patient = make_agent(energy=800.0, entropy=30.0)
        patient.genome = Genome("patient", PolicyParams(dissipate_at=200.0))
        assert not isinstance(
            brain.decide(engine.perceive(patient, tick=1), patient).action, Dissipate
        )

    def test_will_not_bankrupt_itself_to_forget(self, brain, engine, make_agent):
        """Erasing everything the moment you can afford nothing else is fatal."""
        agent = make_agent(energy=3.0, entropy=70.0)
        decision = brain.decide(engine.perceive(agent, tick=1), agent)
        assert not isinstance(decision.action, Dissipate)

    def test_moves_toward_visible_food(self, brain, engine, make_agent, world):
        world.node_y = np.array([8])
        world.node_x = np.array([5])
        world.node_energy = np.array([400.0])
        world.node_temperature = np.array([7.0])

        agent = make_agent(y=5, x=5, energy=50.0)
        decision = brain.decide(engine.perceive(agent, tick=1), agent)
        assert isinstance(decision.action, Move)
        assert decision.action.dy == 1  # toward y=8

    def test_ignores_food_at_thermal_equilibrium(self, brain, engine, make_agent,
                                                 world, small_config):
        """A huge cold node is not food, and chasing it is a fatal mistake."""
        world.node_y = np.array([8])
        world.node_x = np.array([5])
        world.node_energy = np.array([9999.0])
        world.node_temperature = np.array([small_config.world.ambient_temperature])

        agent = make_agent(y=5, x=5, energy=50.0)
        decision = brain.decide(engine.perceive(agent, tick=1), agent)
        assert not (isinstance(decision.action, Move) and decision.action.dy == 1)

    def test_prefers_the_cooler_of_two_equal_nodes(self, brain, engine, make_agent,
                                                   world):
        """Heat aversion: a rich node in a hot cell yields less, by Carnot."""
        world.node_y = np.array([5, 5])
        world.node_x = np.array([3, 7])
        world.node_energy = np.array([400.0, 400.0])
        world.node_temperature = np.array([7.0, 7.0])
        world.temperature[5, 7] = 5.0  # the eastern node sits in a hot cell

        agent = make_agent(y=5, x=5, energy=50.0)
        decision = brain.decide(engine.perceive(agent, tick=1), agent)
        assert isinstance(decision.action, Move)
        assert decision.action.dx == -1  # west, toward the cooler one

    def test_rests_or_explores_on_empty_ground(self, brain, engine, make_agent, world):
        world.node_y = np.zeros(0, dtype=np.int64)
        world.node_x = np.zeros(0, dtype=np.int64)
        world.node_energy = np.zeros(0)
        world.node_temperature = np.zeros(0)

        # Below the breeding threshold on purpose: a rich organism on empty
        # ground has a better option than either of these, and takes it -- see
        # test_a_rich_organism_on_empty_ground_breeds.
        agent = make_agent(y=5, x=5, energy=150.0, entropy=1.0)
        decision = brain.decide(engine.perceive(agent, tick=1), agent)
        assert isinstance(decision.action, (Rest, Move))

    def test_a_rich_organism_on_empty_ground_breeds(self, brain, engine,
                                                    make_agent, world):
        """Wealth with nothing to spend it on is what replication is for."""
        world.node_y = np.zeros(0, dtype=np.int64)
        world.node_x = np.zeros(0, dtype=np.int64)
        world.node_energy = np.zeros(0)
        world.node_temperature = np.zeros(0)

        agent = make_agent(y=5, x=5, energy=500.0, entropy=1.0)
        decision = brain.decide(engine.perceive(agent, tick=1), agent)
        assert isinstance(decision.action, Replicate)

    def test_a_mutant_never_asks_for_a_child_it_cannot_afford(self, small_config,
                                                              brain, engine,
                                                              make_agent):
        """Mutation can drift `replicate_at` below what the world will accept.

        Such an organism would spend every turn being refused and starve while
        being told no. Whether you are rich enough to breed is not privileged
        information, so the policy floors its own threshold at the world's.
        """
        threshold = small_config.selection.replication_energy_threshold
        agent = make_agent(y=5, x=5, energy=threshold - 1.0, entropy=1.0)
        agent.genome = Genome("eager", PolicyParams(replicate_at=1.0))
        decision = brain.decide(engine.perceive(agent, tick=1), agent)
        assert not isinstance(decision.action, Replicate)

    def test_resting_is_cheaper_than_deliberating(self, brain, engine, make_agent, world):
        """Doing nothing should cost fewer tokens, or idling is a losing move."""
        world.node_y = np.zeros(0, dtype=np.int64)
        world.node_x = np.zeros(0, dtype=np.int64)
        world.node_energy = np.zeros(0)
        world.node_temperature = np.zeros(0)

        agent = make_agent(y=5, x=5, energy=150.0, entropy=1.0)
        agent.genome = Genome("still", PolicyParams(explore_bias=0.0))
        decision = brain.decide(engine.perceive(agent, tick=1), agent)

        assert isinstance(decision.action, Rest)
        assert decision.tokens < 40


class TestSelectionSignal:
    """The premise of the whole project, tested at zero cost.

    Before reproduction existed this was a single assertion about mean
    population. Reproduction split it in two, because living long and leaving
    descendants stopped being the same thing -- which is a result in itself, and
    is why the second test exists rather than being folded into the first.
    """

    FRUGAL = PolicyParams(dissipate_at=45.0)
    WASTEFUL = PolicyParams(dissipate_at=10_000.0)   # never forgets anything

    def _run(self, small_config, tmp_path, policy, seed):
        import json

        from btcs.events import EventLog
        from btcs.runner import SimulationRunner

        config = dataclasses.replace(
            small_config,
            agent=dataclasses.replace(small_config.agent, policy=policy,
                                      initial_population=12),
            run=dataclasses.replace(small_config.run, max_ticks=200, seed=seed),
        )
        path = tmp_path / f"{policy.dissipate_at}_{seed}.jsonl"
        with EventLog(path) as log:
            SimulationRunner(config, log=log).run()

        lifespans, births = [], 0
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["type"] == "agent_died":
                lifespans.append(event["ticks_survived"])
            elif event["type"] == "replication":
                births += 1
        return lifespans, births

    @pytest.mark.parametrize("seed", [42, 7, 99])
    def test_a_wasteful_genome_dies_younger_than_a_frugal_one(self, small_config,
                                                              tmp_path, seed):
        """Entropy kills. Without this there is no pressure at all, and no
        reason to expect prompt evolution to work either."""
        import statistics

        frugal, _ = self._run(small_config, tmp_path, self.FRUGAL, seed)
        wasteful, _ = self._run(small_config, tmp_path, self.WASTEFUL, seed)

        assert statistics.mean(frugal) > statistics.mean(wasteful), (
            f"no selection signal: frugal={statistics.mean(frugal):.1f} "
            f"wasteful={statistics.mean(wasteful):.1f}"
        )

    @pytest.mark.parametrize("seed", [42, 7, 99])
    def test_but_the_wasteful_genome_leaves_more_children(self, small_config,
                                                          tmp_path, seed):
        """The trade-off nobody designed in.

        Erasure costs energy, and energy is what a child is made of. So the
        genome that cleans itself lives longer and breeds less, and the one that
        never cleans dies young having spent everything on offspring. Longevity
        and fecundity are different currencies here -- the life-history
        trade-off real organisms face -- and it fell out of the Landauer bill
        rather than out of a rule.

        If this ever flips, reproduction has stopped costing what it should.
        """
        _, frugal_births = self._run(small_config, tmp_path, self.FRUGAL, seed)
        _, wasteful_births = self._run(small_config, tmp_path, self.WASTEFUL, seed)

        assert wasteful_births > frugal_births, (
            f"frugal={frugal_births} wasteful={wasteful_births}"
        )
