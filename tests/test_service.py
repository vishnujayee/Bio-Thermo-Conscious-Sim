"""The service layer: identity, turn discipline, budgets, argument validation.

These are the guarantees that make "an organism cannot cheat" a property of the
code rather than of the protocol. They are tested here against the service
directly; test_mcp.py checks that the MCP transport preserves them.
"""

from __future__ import annotations

import dataclasses

import pytest

from btcs.events import EventLog
from btcs.runner import SimulationRunner
from btcs.sim.service import ErrorCode, ToolError
from btcs.world.agents import Degradation
from btcs.world.ledger import EntropyLedger


@pytest.fixture
def live(small_config, tmp_path):
    """A founded simulation with an open tick, ready for turns."""
    runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
    runner._found()
    runner.service.begin_tick(1, EntropyLedger())
    return runner


@pytest.fixture
def actor(live):
    """One organism plus its capability token."""
    agent = live.registry.living()[0]
    return agent, live.service.issue_token(agent)


class TestIdentity:
    def test_token_resolves_to_its_organism(self, live, actor):
        agent, token = actor
        assert live.service.resolve(token).id == agent.id

    def test_unknown_token_is_refused(self, live):
        with pytest.raises(ToolError) as caught:
            live.service.resolve("tk_not_a_real_token")
        assert caught.value.code == ErrorCode.UNKNOWN_TOKEN

    def test_tokens_are_stable_per_organism(self, live, actor):
        agent, token = actor
        assert live.service.issue_token(agent) == token

    def test_tokens_are_distinct_between_organisms(self, live):
        tokens = {live.service.issue_token(a) for a in live.registry.living()}
        assert len(tokens) == live.registry.population

    def test_no_tool_accepts_an_agent_id(self, live):
        """The plan's boilerplate took `agent_id` as a parameter, which let any
        organism act as any other. No tool may reintroduce that."""
        import inspect

        from btcs.sim.service import SimulationService

        tools = ["sense_gradient", "ingest_energy", "dissipate_entropy",
                 "perform_work", "rest", "remember", "recall", "perceive"]
        for name in tools:
            params = inspect.signature(getattr(SimulationService, name)).parameters
            assert "agent_id" not in params, f"{name} accepts an agent_id"
            assert "token" in params, f"{name} does not take a capability token"

    def test_an_organism_cannot_act_as_another(self, live):
        """The concrete exploit the plan's signature allowed, now impossible:
        there is no argument by which A can name B."""
        a, b = live.registry.living()[:2]
        token_a = live.service.issue_token(a)

        with live.service.turn(b):  # B's turn is open, A holds only its own token
            with pytest.raises(ToolError) as caught:
                live.service.dissipate_entropy(token_a, 10.0)
        assert caught.value.code == ErrorCode.NOT_YOUR_TURN

    def test_revoked_token_stops_working(self, live, actor):
        agent, token = actor
        live.service.revoke_token(agent.id)
        with pytest.raises(ToolError) as caught:
            live.service.resolve(token)
        assert caught.value.code == ErrorCode.UNKNOWN_TOKEN

    def test_a_dead_organism_cannot_act(self, live, actor):
        agent, token = actor
        agent.alive = False
        with pytest.raises(ToolError) as caught:
            live.service.resolve(token)
        assert caught.value.code == ErrorCode.ERASED


class TestTurnDiscipline:
    def test_acting_outside_a_turn_is_refused(self, live, actor):
        _, token = actor
        with pytest.raises(ToolError) as caught:
            live.service.rest(token)
        assert caught.value.code == ErrorCode.NOT_YOUR_TURN

    def test_acting_inside_a_turn_is_allowed(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            assert live.service.rest(token)["ok"]

    def test_only_one_physical_act_per_turn(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            live.service.rest(token)
            with pytest.raises(ToolError) as caught:
                live.service.rest(token)
        assert caught.value.code == ErrorCode.ALREADY_ACTED

    def test_sensing_does_not_consume_the_action(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            live.service.sense_gradient(token)
            live.service.sense_gradient(token)
            assert live.service.rest(token)["ok"]

    def test_turn_closes_on_exit(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            pass
        with pytest.raises(ToolError):
            live.service.rest(token)


class TestBudgets:
    def test_call_ceiling_is_enforced(self, small_config, live, actor):
        """A confused organism must not be able to spend the whole tick thrashing."""
        agent, token = actor
        ceiling = small_config.agent.max_tool_calls_per_tick
        with live.service.turn(agent):
            for _ in range(ceiling):
                live.service.sense_gradient(token)
            with pytest.raises(ToolError) as caught:
                live.service.sense_gradient(token)
        assert caught.value.code == ErrorCode.CALL_BUDGET_EXCEEDED


class TestArgumentGuardrails:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_numbers_are_rejected(self, live, actor, bad):
        """A NaN reaching the physics poisons every later tick *and* makes the
        second-law assertion pass vacuously, since NaN comparisons are false."""
        agent, token = actor
        with live.service.turn(agent):
            with pytest.raises(ToolError) as caught:
                live.service.dissipate_entropy(token, bad)
        assert caught.value.code == ErrorCode.INVALID_ARGUMENT

    @pytest.mark.parametrize("bad", [0.0, -5.0])
    def test_non_positive_amounts_are_rejected(self, live, actor, bad):
        agent, token = actor
        with live.service.turn(agent):
            with pytest.raises(ToolError) as caught:
                live.service.ingest_energy(token, bad)
        assert caught.value.code == ErrorCode.INVALID_ARGUMENT

    def test_oversized_steps_are_clamped_not_rejected(self, live, actor):
        """Asking to leap five cells yields one step: a truthful physical answer."""
        agent, token = actor
        agent.energy = 500.0
        start = (agent.y, agent.x)
        with live.service.turn(agent):
            live.service.perform_work(token, 5, -9)
        dy = (agent.y - start[0]) % live.world.height
        assert dy in (1, live.world.height - 1)

    def test_zero_displacement_is_rejected(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            with pytest.raises(ToolError) as caught:
                live.service.perform_work(token, 0, 0)
        assert caught.value.code == ErrorCode.INVALID_ARGUMENT

    def test_impossible_action_is_a_refusal_not_a_crash(self, live, actor):
        """Eating where there is nothing is information, not an error."""
        agent, token = actor
        import numpy as np

        live.world.node_y = np.zeros(0, dtype=np.int64)
        live.world.node_x = np.zeros(0, dtype=np.int64)
        live.world.node_energy = np.zeros(0)
        live.world.node_temperature = np.zeros(0)

        with live.service.turn(agent):
            with pytest.raises(ToolError) as caught:
                live.service.ingest_energy(token, 10.0)
        assert caught.value.code == ErrorCode.REJECTED


class TestPerceptIsolation:
    def test_percept_carries_no_arrays(self, live, actor):
        """The physics firewall: no simulation array may cross the boundary."""
        agent, token = actor
        with live.service.turn(agent):
            payload = live.service.sense_gradient(token)

        import json

        json.dumps(payload)  # raises if anything non-serialisable slipped through
        assert "temperature_field" not in payload
        assert "nodes" in payload

    def test_percept_never_names_another_organism(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            payload = live.service.sense_gradient(token)
        text = repr(payload)
        for other in live.registry.living():
            if other.id != agent.id:
                assert other.id not in text


class TestMemory:
    def test_remember_then_recall(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            live.service.remember(token, "food", "hot node north")
            assert live.service.recall(token, "food")["content"] == "hot node north"

    def test_recall_lists_keys(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            live.service.remember(token, "a", "one")
            live.service.remember(token, "b", "two")
            assert live.service.recall(token)["keys"] == ["a", "b"]

    def test_memory_is_private_to_its_organism(self, live):
        a, b = live.registry.living()[:2]
        token_a = live.service.issue_token(a)
        token_b = live.service.issue_token(b)

        with live.service.turn(a):
            live.service.remember(token_a, "secret", "mine alone")
        with live.service.turn(b):
            assert live.service.recall(token_b, "secret")["found"] is False

    def test_missing_key_reports_not_found(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            result = live.service.recall(token, "nothing")
        assert result["ok"] and result["found"] is False

    def test_empty_content_is_rejected(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            with pytest.raises(ToolError) as caught:
                live.service.remember(token, "k", "   ")
        assert caught.value.code == ErrorCode.INVALID_ARGUMENT

    def test_memory_has_a_capacity(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            for i in range(500):
                try:
                    live.service.remember(token, f"k{i}", "x")
                except ToolError as exc:
                    assert exc.code == ErrorCode.MEMORY_FULL
                    break
                # The call ceiling would fire first without this reset; we are
                # testing capacity, not budget.
                live.service._turn.calls = 0
            else:
                pytest.fail("memory accepted unbounded writes")

    def test_confusion_makes_recall_unreliable(self, live, actor):
        """Entropy must corrupt the map, not merely raise a number."""
        agent, token = actor
        with live.service.turn(agent):
            for i in range(30):
                live.service._turn.calls = 0
                live.service.remember(token, f"k{i}", "value")

        agent.degradation = Degradation.CONFUSED
        with live.service.turn(agent):
            recalled = live.service.recall(token)["keys"]

        assert len(recalled) < 30


class TestCognitionIsNotSelfServed:
    def test_charging_is_not_a_tool(self):
        """An organism that could choose when to be billed for thinking would
        think for free. `charge_cognition` takes an Agent, not a token, so it is
        unreachable from the tool surface."""
        import inspect

        from btcs.sim.service import SimulationService

        for name in ("charge_cognition", "charge_basal"):
            params = inspect.signature(getattr(SimulationService, name)).parameters
            assert "token" not in params

    def test_not_exposed_over_mcp(self, live):
        from btcs.mcp_server import build_server

        server = build_server(live.service)
        # Registered tool names must not include the metabolic charging methods.
        names = set(server._tool_manager._tools) if hasattr(server, "_tool_manager") else set()
        assert "charge_cognition" not in names
        assert "charge_basal" not in names


class TestActionFeedback:
    """OPEN-15: an organism must be able to perceive what came of its own act.

    Before this existed, an organism was refused, told nothing, and repeated the
    same failing act next tick -- nineteen times out of twenty-three in the first
    real-model run. These tests are the guard on the sense, not on any particular
    brain's use of it.
    """

    def test_nothing_is_remembered_before_acting(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            assert live.service.perceive(token).last_action is None

    def test_a_successful_act_is_remembered(self, live, actor):
        agent, token = actor
        with live.service.turn(agent):
            live.service.rest(token)
        assert agent.last_action is not None
        assert agent.last_action.ok is True
        assert agent.last_action.kind == "rest"

    def test_a_refused_act_is_remembered_with_its_reason(self, live):
        """The case that mattered: eating where there is no food."""
        agent = next(
            a for a in live.registry.living()
            if live.world.node_index_at(a.y, a.x) < 0
        )
        token = live.service.issue_token(agent)
        with live.service.turn(agent):
            with pytest.raises(ToolError):
                live.service.ingest_energy(token, 10.0)
        assert agent.last_action.ok is False
        assert agent.last_action.kind == "ingest"
        assert agent.last_action.detail

    def test_a_malformed_request_is_remembered_too(self, live, actor):
        """A bad argument is also something that happened as a result of choosing."""
        agent, token = actor
        with live.service.turn(agent):
            with pytest.raises(ToolError):
                live.service.ingest_energy(token, -5.0)
        assert agent.last_action.ok is False
        assert agent.last_action.kind == "ingest"

    def test_the_memory_reaches_the_next_tick(self, live, actor):
        """Feedback that expired at the end of the turn would be useless."""
        agent, token = actor
        with live.service.turn(agent):
            live.service.rest(token)

        live.service.begin_tick(2, EntropyLedger())
        with live.service.turn(agent):
            percept = live.service.perceive(token)
        assert percept.last_action is not None
        assert percept.last_action.tick == 1

    def test_it_survives_the_wire(self, live, actor):
        """sense_gradient is JSON; the sense must not be lost in translation."""
        agent, token = actor
        with live.service.turn(agent):
            live.service.rest(token)
        live.service.begin_tick(2, EntropyLedger())
        with live.service.turn(agent):
            payload = live.service.sense_gradient(token)
        assert payload["last_action"]["kind"] == "rest"
        assert payload["last_action"]["ok"] is True

    def test_a_confused_organism_forgets_what_it_did(self, small_config, live, actor):
        """Disorder corrupts the map; there is no reason it would spare this.

        The memory is still held -- the organism simply cannot reach it, which is
        the same asymmetry every other rung of the ladder uses.
        """
        agent, token = actor
        with live.service.turn(agent):
            live.service.rest(token)

        agent.entropy = small_config.agent.s_confuse * 1.5
        live.engine.resolve_degradation(agent, tick=1)
        live.service.begin_tick(2, EntropyLedger())
        with live.service.turn(agent):
            percept = live.service.perceive(token)
        assert percept.last_action is None
        assert agent.last_action is not None
