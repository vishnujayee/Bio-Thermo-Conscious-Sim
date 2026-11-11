"""The local (Ollama) brain: accounting, argument repair, and failure.

Nothing here touches a server. The transport is a scripted fake, for the same
reason the Anthropic tests use one: a test that needs a 3B model loaded is a
test that gets skipped, and a skipped test guards nothing.

The load-bearing tests in this file are the accounting ones. The local brain is
where the entropy books are most easily made to lie -- Ollama reports no cache
at all, so the split between "information this organism already held" and
"information it has just read" is *reconstructed* here rather than reported.
Get that split wrong and the physics is wrong while every test still passes.
"""

from __future__ import annotations

import dataclasses
import time

import pytest

from btcs.brains.base import Brain
from btcs.brains.local import LocalBrain, to_ollama_tools
from btcs.cognition.prompt import ACTION_TOOLS, TOOLS
from btcs.config import Config
from btcs.world.types import Dissipate, Ingest, Move, Rest


class FakeOllama:
    """Replays a scripted list of responses and records what it was sent."""

    def __init__(self, responses: list[dict] | None = None) -> None:
        self.responses = list(responses or [])
        self.payloads: list[dict] = []

    def chat(self, payload: dict) -> dict:
        self.payloads.append(payload)
        if not self.responses:
            raise AssertionError("the brain made more requests than were scripted")
        return self.responses.pop(0)

    def available(self) -> bool:
        return True


class SlowOllama(FakeOllama):
    """Takes a measurable amount of real time, so compute seconds are non-zero."""

    def chat(self, payload: dict) -> dict:
        time.sleep(0.01)
        return super().chat(payload)


class BrokenOllama:
    """A server that is not there."""

    def chat(self, payload: dict) -> dict:
        raise ConnectionError("connection refused")

    def available(self) -> bool:
        return False


def reply(name: str | None = None, arguments: dict | None = None, *,
          prompt_tokens: int = 1500, output_tokens: int = 20) -> dict:
    calls = []
    if name is not None:
        calls = [{"id": "call_1", "function": {"name": name,
                                               "arguments": arguments or {}}}]
    return {
        "message": {"role": "assistant", "content": "", "tool_calls": calls},
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
    }


@pytest.fixture
def local_config(small_config) -> Config:
    return dataclasses.replace(
        small_config,
        cognition=dataclasses.replace(small_config.cognition, brain="local"),
    )


@pytest.fixture
def make_brain(local_config, rng, log):
    def _make(responses=None, *, config=None, transport=None) -> LocalBrain:
        return LocalBrain(
            config or local_config, rng, log,
            transport=transport if transport is not None else FakeOllama(responses),
        )

    return _make


@pytest.fixture
def percept(engine, make_agent):
    return engine.perceive(make_agent(y=4, x=4), tick=1)


@pytest.fixture
def service_bridge(small_config, tmp_path):
    """A real open turn, so deliberation is tested against the actual tool path."""
    from btcs.brains.base import TurnBridge
    from btcs.events import EventLog
    from btcs.runner import SimulationRunner
    from btcs.world.ledger import EntropyLedger

    runner = SimulationRunner(small_config, log=EventLog(tmp_path / "bridge.jsonl"))
    runner._found()
    runner.service.begin_tick(1, EntropyLedger())
    agent = runner.registry.living()[0]
    token = runner.service.issue_token(agent)
    with runner.service.turn(agent):
        yield TurnBridge(runner.service, token)


class TestInterface:
    def test_satisfies_the_brain_protocol(self, make_brain):
        assert isinstance(make_brain([]), Brain)

    def test_reports_its_name(self, make_brain):
        assert make_brain([]).name == "local"


class TestToolTranslation:
    """Both brains must see identical affordances, or a comparison between them
    is about two different tool descriptions rather than about two models."""

    def test_every_tool_survives_translation(self):
        translated = to_ollama_tools(TOOLS)
        assert len(translated) == len(TOOLS)
        assert [t["function"]["name"] for t in translated] == [t["name"] for t in TOOLS]

    def test_schema_and_description_are_carried_across_unchanged(self):
        for original, translated in zip(TOOLS, to_ollama_tools(TOOLS), strict=True):
            assert translated["type"] == "function"
            assert translated["function"]["description"] == original["description"]
            assert translated["function"]["parameters"] == original["input_schema"]

    def test_no_tool_exposes_an_agent_id(self):
        """S1 again, on the second transport. The hole must be closed on both."""
        for tool in to_ollama_tools(TOOLS):
            assert "agent_id" not in (tool["function"]["parameters"].get("properties") or {})


class TestTheTurn:
    def test_an_action_tool_ends_the_turn_in_one_request(self, make_brain, percept,
                                                         make_agent):
        transport = FakeOllama([reply("ingest_energy", {"joules": 12.0})])
        decision = make_brain(transport=transport).decide(percept, make_agent())
        assert decision.action == Ingest(12.0)
        assert len(transport.payloads) == 1

    def test_a_deliberation_tool_loops_before_acting(self, make_brain, percept,
                                                     make_agent, service_bridge):
        transport = FakeOllama([
            reply("sense_gradient"),
            reply("dissipate_entropy", {"bits": 5.0}),
        ])
        decision = make_brain(transport=transport).decide(
            percept, make_agent(), service_bridge
        )
        assert decision.action == Dissipate(5.0)
        assert len(transport.payloads) == 2
        # The tool result must actually reach the model, or deliberation is theatre.
        assert any(m["role"] == "tool" for m in transport.payloads[1]["messages"])

    def test_deliberating_without_a_bridge_falls_back_rather_than_crashing(
        self, make_brain, percept, make_agent
    ):
        decision = make_brain([reply("sense_gradient")]).decide(percept, make_agent())
        assert decision.action == Rest()
        assert decision.metadata["fallback"] == "no_tool_call"

    def test_round_ceiling_is_respected(self, local_config, make_brain, percept,
                                        make_agent, service_bridge):
        """Curiosity is capped, because every extra round is real entropy."""
        config = dataclasses.replace(
            local_config,
            cognition=dataclasses.replace(local_config.cognition, max_tool_rounds=2),
        )
        transport = FakeOllama([reply("sense_gradient")] * 2)
        brain = make_brain(config=config, transport=transport)
        brain.decide(percept, make_agent(), service_bridge)
        assert len(transport.payloads) == 2

    def test_a_dead_server_costs_a_turn_but_not_the_run(self, make_brain, percept,
                                                        make_agent):
        decision = make_brain(transport=BrokenOllama()).decide(percept, make_agent())
        assert decision.action == Rest()
        assert decision.metadata["fallback"] == "request_failed"
        assert decision.tokens == 0


class TestArgumentRepair:
    """Small models omit required arguments constantly. A voided turn teaches us
    nothing about the model; a repaired one is still the model's choice."""

    def test_ingest_without_joules_eats_as_much_as_allowed(self, local_config,
                                                           make_brain, percept,
                                                           make_agent):
        decision = make_brain([reply("ingest_energy", {})]).decide(percept, make_agent())
        assert decision.action == Ingest(local_config.physics.ingest_rate_limit)

    def test_dissipate_without_bits_clears_the_excess(self, make_brain, engine,
                                                      make_agent, local_config):
        agent = make_agent(y=4, x=4, entropy=local_config.agent.s_warn * 2)
        decision = make_brain([reply("dissipate_entropy", {})]).decide(
            engine.perceive(agent, tick=1), agent
        )
        assert isinstance(decision.action, Dissipate)
        assert 0 < decision.action.bits <= local_config.agent.max_dissipate_bits

    def test_work_without_a_direction_walks_toward_the_best_node(self, make_brain,
                                                                 percept, make_agent):
        decision = make_brain([reply("perform_work", {})]).decide(percept, make_agent())
        if percept.nodes:
            assert isinstance(decision.action, Move)
        else:  # nothing in sight is a legitimate reason to stay put
            assert decision.action == Rest()

    def test_repair_can_be_switched_off(self, local_config, make_brain, percept,
                                        make_agent):
        config = dataclasses.replace(
            local_config,
            cognition=dataclasses.replace(local_config.cognition,
                                          local_repair_arguments=False),
        )
        decision = make_brain([reply("ingest_energy", {})], config=config).decide(
            percept, make_agent()
        )
        assert decision.action == Rest()
        assert decision.metadata["fallback"] == "no_tool_call"

    def test_arguments_may_arrive_as_a_json_string(self, make_brain, percept,
                                                   make_agent):
        decision = make_brain([reply("ingest_energy", '{"joules": 7.5}')]).decide(
            percept, make_agent()
        )
        assert decision.action == Ingest(7.5)

    def test_unparseable_arguments_do_not_crash_the_turn(self, make_brain, percept,
                                                         make_agent):
        decision = make_brain([reply("ingest_energy", "not json at all")]).decide(
            percept, make_agent()
        )
        # Falls through to repair: a mangled argument list is still an intention.
        assert isinstance(decision.action, Ingest)


class TestAccounting:
    """Where this brain can most easily lie about physics."""

    def test_first_sighting_of_a_prefix_is_charged_as_a_cache_write(
        self, make_brain, percept, make_agent
    ):
        brain = make_brain([reply("rest", prompt_tokens=1500)])
        meta = brain.decide(percept, make_agent()).metadata
        assert meta["cache_creation_input_tokens"] > 0
        assert meta["cache_read_input_tokens"] == 0

    def test_re_reading_the_same_prefix_becomes_a_cache_read(self, make_brain,
                                                             percept, make_agent):
        """D-11: re-reading an unchanged block is not new cognitive work."""
        brain = make_brain([reply("rest"), reply("rest")])
        agent = make_agent()
        brain.decide(percept, agent)
        second = brain.decide(percept, agent).metadata
        assert second["cache_read_input_tokens"] > 0
        assert second["cache_creation_input_tokens"] == 0

    def test_perception_is_never_free(self, make_brain, percept, make_agent):
        """The regression that matters.

        The first implementation estimated the *stable* prefix and subtracted it
        from the reported input. The estimator overshot the real tokenizer by
        enough that the estimated prefix exceeded the whole prompt, so the
        percept -- the only genuinely new information in the turn -- was swept
        into the discount and read at a tenth of its entropy. Estimating the
        volatile side instead puts the error on the prefix, where over-charging
        is the safe direction.
        """
        brain = make_brain([reply("rest"), reply("rest")])
        agent = make_agent()
        brain.decide(percept, agent)
        second = brain.decide(percept, agent).metadata
        assert second["input_tokens"] > 0, "the world was read for free"

    def test_the_split_never_invents_tokens(self, make_brain, percept, make_agent):
        """Whatever the split, it must add up to what the server reported."""
        brain = make_brain([reply("rest", prompt_tokens=1500)])
        meta = brain.decide(percept, make_agent()).metadata
        counted = (meta["input_tokens"] + meta["cache_creation_input_tokens"]
                   + meta["cache_read_input_tokens"])
        assert counted == 1500

    def test_the_discount_can_be_switched_off(self, local_config, make_brain,
                                              percept, make_agent):
        config = dataclasses.replace(
            local_config,
            cognition=dataclasses.replace(local_config.cognition,
                                          local_simulate_cache=False),
        )
        brain = make_brain([reply("rest", prompt_tokens=1500)], config=config)
        meta = brain.decide(percept, make_agent()).metadata
        assert meta["input_tokens"] == 1500
        assert meta["cache_read_input_tokens"] == 0

    def test_thinking_always_costs_something(self, make_brain, percept, make_agent):
        decision = make_brain([reply("rest")]).decide(percept, make_agent())
        assert decision.tokens > 0

    def test_compute_seconds_can_be_charged_as_entropy(self, local_config,
                                                       make_brain, percept,
                                                       make_agent):
        """The one cost here that is not a metaphor: real hardware, really working."""
        base = make_brain(transport=SlowOllama([reply("rest")])).decide(
            percept, make_agent()
        ).tokens
        config = dataclasses.replace(
            local_config,
            cognition=dataclasses.replace(local_config.cognition,
                                          compute_seconds_entropy_weight=1000.0),
        )
        charged = make_brain(
            config=config, transport=SlowOllama([reply("rest")])
        ).decide(percept, make_agent()).tokens
        assert charged > base


class TestVirtualPricing:
    def test_a_local_run_is_priced_at_the_tier_model(self, make_brain, percept,
                                                     make_agent):
        """D-12: free runs must exercise the same economics as paid ones."""
        brain = make_brain([reply("rest")])
        meta = brain.decide(percept, make_agent()).metadata
        assert meta["model"] == "qwen2.5:3b"
        assert meta["priced_as"].startswith("claude-")
        assert meta["virtual"] is True
        assert meta["cost_usd"] > 0

    def test_spend_reaches_the_budget_guard(self, make_brain, percept, make_agent):
        brain = make_brain([reply("rest")])
        brain.decide(percept, make_agent())
        assert brain.budget.spent_usd > 0

    def test_virtual_pricing_can_be_switched_off(self, local_config, make_brain,
                                                 percept, make_agent):
        config = dataclasses.replace(
            local_config,
            cognition=dataclasses.replace(local_config.cognition,
                                          virtual_pricing=False),
        )
        meta = make_brain([reply("rest")], config=config).decide(
            percept, make_agent()
        ).metadata
        assert meta["priced_as"] == "qwen2.5:3b"
        # An unpriced model must not silently cost nothing.
        assert meta["cost_usd"] > 0


def test_action_tools_are_the_ones_that_end_a_turn(make_brain, percept, make_agent):
    """A guard on the tool surface itself: every action tool must terminate."""
    for name in ACTION_TOOLS:
        decision = make_brain([reply(name, {})]).decide(percept, make_agent())
        assert "fallback" not in decision.metadata, f"{name} did not end the turn"
