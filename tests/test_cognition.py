"""Phase 3: pricing, budget, prompt structure, and the LLM brain.

Every test here runs against a scripted fake client. Nothing in this suite may
make a real API call.
"""

from __future__ import annotations

import dataclasses

import pytest

from btcs.brains.base import TurnBridge
from btcs.brains.llm import LLMBrain
from btcs.cognition.budget import BudgetExceeded, BudgetGuard
from btcs.cognition.pricing import (
    MODEL_PRICES,
    UNKNOWN_MODEL_PRICE,
    Usage,
    cost_usd,
    entropy_tokens,
    price_of,
)
from btcs.cognition.prompt import ACTION_TOOLS, LAWS, TOOLS, prefix_fingerprint, system_blocks
from btcs.config import Config, PolicyParams
from btcs.events import EventLog, EventType, read_events
from btcs.runner import SimulationRunner
from btcs.world.agents import Genome
from btcs.world.ledger import EntropyLedger
from btcs.world.types import Dissipate, Ingest, Move, Rest

from tests.fakes import FakeAnthropic, FakeUsage, tool_call, text_only
from tests import fakes


# --------------------------------------------------------------------------
# pricing
# --------------------------------------------------------------------------


class TestPricing:
    def test_known_models_are_priced(self):
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"):
            assert model in MODEL_PRICES

    def test_tier_models_all_have_prices(self):
        """A Phi tier naming an unpriced model would bill at the penalty rate."""
        for tier in Config().cognition.phi_tiers:
            assert tier.model in MODEL_PRICES, f"{tier.model} has no price"

    def test_unknown_model_is_conspicuously_expensive(self):
        """Better to over-report than to let an unknown id read as free."""
        assert price_of("claude-imaginary-9").input > MODEL_PRICES["claude-opus-5"].input
        assert price_of("claude-imaginary-9") is UNKNOWN_MODEL_PRICE

    def test_cache_read_is_a_tenth_of_input(self):
        price = MODEL_PRICES["claude-opus-5"]
        assert price.cache_read == pytest.approx(price.input * 0.1)

    def test_cost_of_a_known_request(self):
        usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert cost_usd(usage, "claude-opus-5") == pytest.approx(30.0)

    def test_cache_reads_are_cheap(self):
        fresh = cost_usd(Usage(input_tokens=100_000), "claude-opus-5")
        cached = cost_usd(Usage(cache_read_input_tokens=100_000), "claude-opus-5")
        assert cached == pytest.approx(fresh * 0.1)

    def test_haiku_is_cheaper_than_opus(self):
        usage = Usage(input_tokens=10_000, output_tokens=1_000)
        assert cost_usd(usage, "claude-haiku-4-5") < cost_usd(usage, "claude-opus-5")


class TestUsage:
    def test_addition_accumulates_every_field(self):
        total = Usage(1, 2, 3, 4) + Usage(10, 20, 30, 40)
        assert (total.input_tokens, total.output_tokens) == (11, 22)
        assert (total.cache_creation_input_tokens, total.cache_read_input_tokens) == (33, 44)

    def test_cache_hit_rate(self):
        assert Usage(input_tokens=10, cache_read_input_tokens=90).cache_hit_rate == 0.9
        assert Usage().cache_hit_rate == 0.0

    def test_from_response_tolerates_missing_fields(self):
        """Usage has gained fields over time; an accounting crash mid-run would
        cost more than an under-reported cache statistic."""

        class Sparse:
            input_tokens = 5

        usage = Usage.from_response(Sparse())
        assert usage.input_tokens == 5
        assert usage.cache_read_input_tokens == 0

    def test_from_response_reads_the_fake(self):
        usage = Usage.from_response(FakeUsage(1, 2, 3, 4))
        assert usage.to_dict() == {
            "input_tokens": 1, "output_tokens": 2,
            "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4,
        }


class TestEntropyWeighting:
    """OPEN-10: not every token is equally costly to think."""

    def test_defaults_discount_cache_reads(self):
        cognition = Config().cognition
        fresh = entropy_tokens(Usage(input_tokens=1000), cognition)
        cached = entropy_tokens(Usage(cache_read_input_tokens=1000), cognition)
        assert cached < fresh
        assert cached == pytest.approx(fresh * cognition.cache_read_entropy_weight)

    def test_weight_of_one_removes_the_discount(self):
        """Switching the pressure off must be a one-line config change."""
        cognition = dataclasses.replace(
            Config().cognition, cache_read_entropy_weight=1.0
        )
        fresh = entropy_tokens(Usage(input_tokens=1000), cognition)
        cached = entropy_tokens(Usage(cache_read_input_tokens=1000), cognition)
        assert cached == pytest.approx(fresh)

    def test_output_weight_can_make_verbosity_lethal(self):
        base = Config().cognition
        loud = dataclasses.replace(base, output_entropy_weight=5.0)
        usage = Usage(output_tokens=1000)
        assert entropy_tokens(usage, loud) == pytest.approx(
            5.0 * entropy_tokens(usage, base)
        )


# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------


class TestBudgetGuard:
    def test_records_spend(self):
        guard = BudgetGuard(ceiling_usd=100.0)
        cost = guard.record(Usage(input_tokens=1_000_000), "claude-opus-5")
        assert cost == pytest.approx(5.0)
        assert guard.spent_usd == pytest.approx(5.0)
        assert guard.requests == 1

    def test_raises_once_the_ceiling_is_crossed(self):
        guard = BudgetGuard(ceiling_usd=1.0)
        with pytest.raises(BudgetExceeded):
            guard.record(Usage(input_tokens=1_000_000), "claude-opus-5")

    def test_the_overspend_is_still_recorded(self):
        """The request already happened; pretending otherwise under-reports."""
        guard = BudgetGuard(ceiling_usd=1.0)
        with pytest.raises(BudgetExceeded):
            guard.record(Usage(input_tokens=1_000_000), "claude-opus-5")
        assert guard.spent_usd == pytest.approx(5.0)

    def test_can_be_configured_not_to_halt(self):
        guard = BudgetGuard(ceiling_usd=1.0, halt_on_exceed=False)
        guard.record(Usage(input_tokens=1_000_000), "claude-opus-5")
        assert guard.exhausted

    def test_attribution_by_model_and_agent(self):
        guard = BudgetGuard(ceiling_usd=1000.0)
        guard.record(Usage(input_tokens=1_000_000), "claude-opus-5", "A1")
        guard.record(Usage(input_tokens=1_000_000), "claude-haiku-4-5", "A2")
        assert set(guard.by_model) == {"claude-opus-5", "claude-haiku-4-5"}
        assert guard.by_agent["A1"] > guard.by_agent["A2"]

    def test_remaining_never_goes_negative(self):
        guard = BudgetGuard(ceiling_usd=1.0, halt_on_exceed=False)
        guard.record(Usage(input_tokens=1_000_000), "claude-opus-5")
        assert guard.remaining_usd == 0.0


# --------------------------------------------------------------------------
# prompt structure
# --------------------------------------------------------------------------


class TestPromptCaching:
    """The prefix must be stable, or the whole economy stops working."""

    def test_laws_are_identical_across_organisms(self, small_config, make_agent):
        """The frozen block is shared by the entire population, so every
        organism reads one cached prefix."""
        a, b = make_agent(), make_agent()
        b.genome = Genome("a different strategy", PolicyParams())
        assert system_blocks(a, small_config)[0] == system_blocks(b, small_config)[0]

    def test_genome_block_differs_by_lineage(self, small_config, make_agent):
        a, b = make_agent(), make_agent()
        b.genome = Genome("hoard everything", PolicyParams(dissipate_at=99.0))
        assert system_blocks(a, small_config)[1] != system_blocks(b, small_config)[1]

    def test_clones_share_a_prefix(self, small_config, make_agent):
        """Replication raises the cache hit rate. Selection for metabolic
        efficiency falls out of the billing system without being designed in."""
        parent = make_agent()
        clone = make_agent()
        clone.genome = parent.genome
        assert prefix_fingerprint(parent, small_config) == prefix_fingerprint(clone, small_config)

    def test_breakpoint_is_on_the_last_system_block(self, small_config, make_agent):
        blocks = system_blocks(make_agent(), small_config)
        assert "cache_control" not in blocks[0]
        assert blocks[-1]["cache_control"] == {"type": "ephemeral"}

    def test_prefix_carries_nothing_volatile(self, small_config, make_agent):
        """A tick number, an agent id or a timestamp in the prefix would
        invalidate the cache on every single request."""
        agent = make_agent()
        prefix = "".join(b["text"] for b in system_blocks(agent, small_config))
        assert agent.id not in prefix
        for volatile in ("2026", "tick", "Tick"):
            assert volatile not in prefix

    def test_prefix_is_stable_across_calls(self, small_config, make_agent):
        agent = make_agent()
        assert system_blocks(agent, small_config) == system_blocks(agent, small_config)

    def test_position_does_not_change_the_prefix(self, small_config, make_agent):
        """Movement is per-tick state; if it reached the prefix, nothing caches."""
        agent = make_agent(y=1, x=1)
        before = prefix_fingerprint(agent, small_config)
        agent.y, agent.x = 9, 9
        assert prefix_fingerprint(agent, small_config) == before

    def test_vitals_do_not_change_the_prefix(self, small_config, make_agent):
        agent = make_agent()
        before = prefix_fingerprint(agent, small_config)
        agent.energy, agent.entropy = 1.0, 500.0
        assert prefix_fingerprint(agent, small_config) == before


class TestToolDefinitions:
    def test_tools_are_sorted(self):
        """Tools render at position zero; a reordered list invalidates the
        prefix for every organism at once."""
        names = [t["name"] for t in TOOLS]
        assert names == sorted(names)

    def test_no_tool_takes_a_credential(self):
        """The organism's token is injected by the brain when it executes a
        call, so a model cannot alter, leak, or hallucinate one it never saw."""
        for tool in TOOLS:
            properties = tool["input_schema"].get("properties", {})
            assert "token" not in properties, f"{tool['name']} exposes a token"
            assert "agent_id" not in properties

    def test_every_action_tool_is_defined(self):
        defined = {t["name"] for t in TOOLS}
        assert set(ACTION_TOOLS) <= defined

    def test_schemas_are_closed(self):
        for tool in TOOLS:
            assert tool["input_schema"].get("additionalProperties") is False

    def test_laws_mention_the_load_bearing_physics(self):
        for phrase in ("free energy", "gradient", "entropy", "waste heat"):
            assert phrase in LAWS.lower()


# --------------------------------------------------------------------------
# the brain
# --------------------------------------------------------------------------


@pytest.fixture
def live(small_config, tmp_path):
    runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
    runner._found()
    runner.service.begin_tick(1, EntropyLedger())
    return runner


def brain_for(config, live, script=None, error=None, budget=None):
    return LLMBrain(
        config, live.rng, live.log,
        client=FakeAnthropic(script, error=error), budget=budget,
    )


@pytest.fixture
def llm_config(small_config):
    return dataclasses.replace(
        small_config,
        cognition=dataclasses.replace(small_config.cognition, brain="llm"),
    )


class TestLLMBrainActions:
    @pytest.mark.parametrize(
        ("call", "expected"),
        [
            (tool_call("rest"), Rest),
            (tool_call("ingest_energy", joules=30.0), Ingest),
            (tool_call("dissipate_entropy", bits=12.0), Dissipate),
            (tool_call("perform_work", dy=1, dx=0), Move),
        ],
    )
    def test_action_tools_become_decisions(self, llm_config, live, call, expected):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [call])

        with live.service.turn(agent):
            decision = brain.decide(
                live.service.perceive(token), agent, TurnBridge(live.service, token)
            )
        assert isinstance(decision.action, expected)

    def test_arguments_are_carried_through(self, llm_config, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("dissipate_entropy", bits=17.5)])

        with live.service.turn(agent):
            decision = brain.decide(
                live.service.perceive(token), agent, TurnBridge(live.service, token)
            )
        assert decision.action.bits == pytest.approx(17.5)

    def test_one_request_in_the_common_case(self, llm_config, live):
        """The percept is pre-loaded, so a decisive organism costs one call."""
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("rest")])

        with live.service.turn(agent):
            brain.decide(live.service.perceive(token), agent, TurnBridge(live.service, token))
        assert len(brain._client.calls) == 1

    def test_malformed_arguments_do_not_crash(self, llm_config, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("ingest_energy", joules="lots")])

        with live.service.turn(agent):
            decision = brain.decide(
                live.service.perceive(token), agent, TurnBridge(live.service, token)
            )
        assert isinstance(decision.action, Rest)


class TestDeliberation:
    def test_deliberation_tools_loop_then_act(self, llm_config, live):
        """Looking before leaping is allowed -- and costs a second request."""
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("recall"), tool_call("rest")])

        with live.service.turn(agent):
            decision = brain.decide(
                live.service.perceive(token), agent, TurnBridge(live.service, token)
            )
        assert isinstance(decision.action, Rest)
        assert len(brain._client.calls) == 2

    def test_curiosity_costs_entropy(self, llm_config, live):
        """The central claim of the design: an extra round-trip is real tokens,
        and real tokens are real internal entropy."""
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)

        decisive = brain_for(llm_config, live, [tool_call("rest")])
        curious = brain_for(llm_config, live, [tool_call("recall"), tool_call("rest")])

        with live.service.turn(agent):
            percept = live.service.perceive(token)
            bridge = TurnBridge(live.service, token)
            cheap = decisive.decide(percept, agent, bridge)
            dear = curious.decide(percept, agent, bridge)

        assert dear.tokens > cheap.tokens

    def test_remember_reaches_the_service(self, llm_config, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [
            tool_call("remember", key="node", content="hot one north"),
            tool_call("rest"),
        ])

        with live.service.turn(agent):
            brain.decide(live.service.perceive(token), agent, TurnBridge(live.service, token))
        assert live.service.memory_of(agent.id)["node"] == "hot one north"

    def test_an_action_tool_wins_over_a_concurrent_lookup(self, llm_config, live):
        """Parallel tool use is on by default. An organism that has decided has
        decided; the accompanying lookup is discarded."""
        from tests.fakes import FakeBlock, FakeResponse

        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        response = FakeResponse([
            FakeBlock("tool_use", name="recall", input={}),
            FakeBlock("tool_use", name="rest", input={}),
        ])
        brain = brain_for(llm_config, live, [response])

        with live.service.turn(agent):
            decision = brain.decide(
                live.service.perceive(token), agent, TurnBridge(live.service, token)
            )
        assert isinstance(decision.action, Rest)
        assert len(brain._client.calls) == 1


class TestFallbacks:
    def test_no_tool_call_rests_but_still_pays(self, llm_config, live):
        """An organism that talks instead of acting has still spent the tokens."""
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [text_only()])

        with live.service.turn(agent):
            decision = brain.decide(
                live.service.perceive(token), agent, TurnBridge(live.service, token)
            )
        assert isinstance(decision.action, Rest)
        assert decision.tokens > 0
        assert decision.metadata["fallback"] == "no_tool_call"

    def test_endless_deliberation_is_capped_and_charged(self, llm_config, live):
        """Indecision must be selectable against, so it costs what it spent."""
        config = dataclasses.replace(
            llm_config,
            cognition=dataclasses.replace(llm_config.cognition, max_tool_rounds=2),
        )
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(config, live, [tool_call("recall")] * 5)

        with live.service.turn(agent):
            decision = brain.decide(
                live.service.perceive(token), agent, TurnBridge(live.service, token)
            )
        assert decision.metadata["fallback"] == "rounds_exhausted"
        assert decision.tokens > 0
        assert len(brain._client.calls) == 2


class TestErrorHandling:
    def test_a_broken_configuration_stops_the_run(self, llm_config, live):
        """Auth and request-shape errors recur on every call; continuing would
        burn the whole run producing nothing."""
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, error=fakes.AuthenticationError("no key"))

        with live.service.turn(agent):
            with pytest.raises(fakes.AuthenticationError):
                brain.decide(
                    live.service.perceive(token), agent, TurnBridge(live.service, token)
                )

    def test_a_transient_failure_costs_the_turn_not_the_run(self, llm_config, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, error=fakes.APIConnectionError("network"))

        with live.service.turn(agent):
            decision = brain.decide(
                live.service.perceive(token), agent, TurnBridge(live.service, token)
            )
        assert isinstance(decision.action, Rest)
        assert decision.metadata["fallback"] == "request_failed"


class TestRoutingAndSpend:
    @pytest.mark.parametrize(
        ("phi", "model"),
        [(0.5, "claude-haiku-4-5"), (1.0, "claude-sonnet-5"), (2.0, "claude-opus-5")],
    )
    def test_phi_selects_the_model(self, llm_config, live, phi, model):
        agent = live.registry.living()[0]
        agent.phi = phi
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("rest")])

        with live.service.turn(agent):
            brain.decide(live.service.perceive(token), agent, TurnBridge(live.service, token))
        assert brain._client.calls[0]["model"] == model

    def test_the_cheapest_tier_sends_no_effort_or_thinking(self, llm_config, live):
        """Effort errors on Haiku, and its thinking uses the older budget_tokens
        shape we deliberately do not use."""
        agent = live.registry.living()[0]
        agent.phi = 0.5
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("rest")])

        with live.service.turn(agent):
            brain.decide(live.service.perceive(token), agent, TurnBridge(live.service, token))
        params = brain._client.calls[0]
        assert "output_config" not in params
        assert "thinking" not in params

    def test_higher_tiers_use_adaptive_thinking_and_effort(self, llm_config, live):
        agent = live.registry.living()[0]
        agent.phi = 9.0
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("rest")])

        with live.service.turn(agent):
            brain.decide(live.service.perceive(token), agent, TurnBridge(live.service, token))
        params = brain._client.calls[0]
        assert params["thinking"] == {"type": "adaptive"}
        assert params["output_config"] == {"effort": "high"}

    def test_budget_is_charged(self, llm_config, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("rest")])

        with live.service.turn(agent):
            brain.decide(live.service.perceive(token), agent, TurnBridge(live.service, token))
        assert brain.budget.spent_usd > 0.0
        assert brain.budget.requests == 1

    def test_an_exhausted_budget_stops_the_brain(self, llm_config, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("rest")],
                          budget=BudgetGuard(ceiling_usd=1e-9))

        with live.service.turn(agent):
            with pytest.raises(BudgetExceeded):
                brain.decide(
                    live.service.perceive(token), agent, TurnBridge(live.service, token)
                )


class TestReplayLogging:
    def test_every_call_is_logged_with_its_response(self, llm_config, live, tmp_path):
        """Model sampling is not reproducible, so the log is what makes an LLM
        run auditable and replayable after the fact."""
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, [tool_call("rest")])
        brain.set_tick(7)

        with live.service.turn(agent):
            brain.decide(live.service.perceive(token), agent, TurnBridge(live.service, token))
        live.log.flush()

        calls = [e for e in read_events(live.log.path) if e["type"] == EventType.MODEL_CALL]
        assert len(calls) == 1
        entry = calls[0]
        assert entry["tick"] == 7
        assert entry["agent"] == agent.id
        assert entry["ok"] is True
        assert entry["model"] == "claude-haiku-4-5"
        assert "response" in entry and "cost_usd" in entry
        assert entry["input_tokens"] > 0

    def test_failures_are_logged_too(self, llm_config, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        brain = brain_for(llm_config, live, error=fakes.APIConnectionError("down"))

        with live.service.turn(agent):
            brain.decide(live.service.perceive(token), agent, TurnBridge(live.service, token))
        live.log.flush()

        calls = [e for e in read_events(live.log.path) if e["type"] == EventType.MODEL_CALL]
        assert calls and calls[0]["ok"] is False
        assert calls[0]["error"] == "APIConnectionError"


class TestEndToEnd:
    def test_a_full_run_on_a_scripted_model(self, llm_config, tmp_path):
        """The physics must hold identically whatever is doing the deciding."""
        config = dataclasses.replace(
            llm_config, run=dataclasses.replace(llm_config.run, max_ticks=25)
        )
        from btcs.rng import RngService

        # The brain is injected with a fake client, so no real SDK client is
        # ever constructed and no credential is ever needed.
        with EventLog(tmp_path / "run.jsonl") as log:
            runner = SimulationRunner(
                config, log=log,
                brain=LLMBrain(config, RngService(config.run.seed), log,
                               client=FakeAnthropic([])),
            )
            result = runner.run()

        assert result.ticks_completed > 0
        assert all(r.ds_total >= -1e-9 for r in result.history)
        assert result.spend_usd > 0.0

    def test_the_budget_halts_a_run(self, llm_config, tmp_path):
        config = dataclasses.replace(
            llm_config,
            run=dataclasses.replace(llm_config.run, max_ticks=50),
            cognition=dataclasses.replace(llm_config.cognition, budget_usd=1e-6),
        )
        from btcs.rng import RngService

        with EventLog(tmp_path / "run.jsonl") as log:
            runner = SimulationRunner(
                config, log=log,
                brain=LLMBrain(config, RngService(1), log, client=FakeAnthropic([])),
            )
            result = runner.run()

        assert result.halted_reason == "budget_exceeded"
        assert result.ticks_completed < 50

    def test_the_graph_path_agrees_with_the_direct_path(self, llm_config, tmp_path):
        """The graph is pure structure. If it ever changes an outcome, one of
        the two paths is wrong."""
        from btcs.rng import RngService

        def go(use_graph: bool):
            config = dataclasses.replace(
                llm_config,
                run=dataclasses.replace(llm_config.run, max_ticks=12),
                cognition=dataclasses.replace(llm_config.cognition, use_graph=use_graph),
            )
            with EventLog(tmp_path / f"g{use_graph}.jsonl") as log:
                runner = SimulationRunner(
                    config, log=log,
                    brain=LLMBrain(config, RngService(config.run.seed), log,
                                   client=FakeAnthropic([])),
                )
                return runner.run()

        direct, graphed = go(False), go(True)
        assert direct.final_population == graphed.final_population
        assert direct.cumulative_entropy == pytest.approx(graphed.cumulative_entropy)


# --------------------------------------------------------------------------
# cost estimation
# --------------------------------------------------------------------------


class TestEstimator:
    """Pricing a run without making a call. Guards the numbers in the docs."""

    def _estimate(self, small_config, ticks: int = 40, **overrides):
        from btcs.cognition.estimate import estimate_run

        cognition = dataclasses.replace(
            small_config.cognition, brain="llm", budget_usd=1e9,
            **overrides.pop("cognition", {}),
        )
        config = dataclasses.replace(
            small_config,
            run=dataclasses.replace(small_config.run, max_ticks=ticks),
            cognition=cognition,
            **overrides,
        )
        return estimate_run(config)

    def test_produces_a_positive_cost(self, small_config):
        estimate, _ = self._estimate(small_config)
        assert estimate.total_usd > 0.0
        assert estimate.turns > 0

    def test_makes_no_api_call(self, small_config):
        """The whole point: this must work on a machine with no credentials."""
        estimate, _ = self._estimate(small_config)
        assert estimate.total_usd > 0  # ran to completion without a client

    def test_models_the_cache(self, small_config):
        """A prefix is written once and read thereafter, so a run of any length
        should show most of its input served from cache."""
        estimate, _ = self._estimate(small_config)
        assert estimate.total_usage.cache_read_input_tokens > 0
        assert estimate.total_usage.cache_hit_rate > 0.5

    def test_a_longer_run_costs_more(self, small_config):
        short, _ = self._estimate(small_config, ticks=20)
        longer, _ = self._estimate(small_config, ticks=80)
        assert longer.total_usd > short.total_usd

    def test_phi_routes_the_estimate_to_the_dearer_model(self, small_config):
        """Per-turn cost must rise with Phi even though total cost may not --
        a high-Phi population dies too quickly to run up a bill."""
        cheap, _ = self._estimate(
            small_config, agent=dataclasses.replace(small_config.agent, initial_phi=0.0)
        )
        dear, _ = self._estimate(
            small_config, agent=dataclasses.replace(small_config.agent, initial_phi=3.0)
        )
        assert "claude-haiku-4-5" in cheap.cost_by_model
        assert "claude-opus-5" in dear.cost_by_model
        assert dear.total_usd / dear.turns > cheap.total_usd / cheap.turns

    def test_report_is_human_readable(self, small_config):
        estimate, _ = self._estimate(small_config)
        report = estimate.report()
        assert "Projected cost" in report
        assert "cost per turn" in report


class TestActionFeedbackInThePrompt:
    """The sense added for OPEN-15 has to survive into the words the model reads.

    A percept field the prompt builder ignores would look correct in every
    service test and change nothing about behaviour.
    """

    def _text(self, percept, config):
        from btcs.cognition.prompt import user_message

        return user_message(percept, {}, config)["content"][0]["text"]

    def test_a_failure_is_stated_and_named(self, small_config, engine, make_agent):
        import dataclasses

        from btcs.world.types import LastAction

        percept = engine.perceive(make_agent(), tick=2)
        percept = dataclasses.replace(
            percept,
            last_action=LastAction("ingest", False, 1, "There is no node here."),
        )
        text = self._text(percept, small_config)
        assert "did NOT happen" in text
        assert "no node here" in text

    def test_a_success_is_stated_without_a_warning(self, small_config, engine,
                                                   make_agent):
        import dataclasses

        from btcs.world.types import LastAction

        percept = dataclasses.replace(
            engine.perceive(make_agent(), tick=2),
            last_action=LastAction("ingest", True, 1),
        )
        text = self._text(percept, small_config)
        assert "succeeded" in text
        assert "did NOT happen" not in text

    def test_a_first_turn_says_nothing_about_a_previous_act(self, small_config,
                                                            engine, make_agent):
        text = self._text(engine.perceive(make_agent(), tick=1), small_config)
        assert "WHAT CAME OF YOUR LAST ACT" not in text

    def test_the_feedback_stays_out_of_the_cached_prefix(self, small_config,
                                                          make_agent):
        """It changes every tick by construction, so it must be volatile.

        Putting it in a system block would invalidate the prefix for the whole
        population on every tick and quietly destroy the cache economy.
        """
        blocks = system_blocks(make_agent(), small_config)
        assert all("LAST ACT" not in block["text"] for block in blocks)
