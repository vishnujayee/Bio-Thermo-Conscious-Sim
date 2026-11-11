"""The MCP transport.

The service's guarantees are tested in test_service.py. What matters here is
that routing through JSON-RPC preserves them exactly -- a transport that
loosened identity or turn discipline would make the "physics firewall" claim
true of the internal code and false of the thing organisms actually talk to.
"""

from __future__ import annotations

import json

import anyio
import pytest
from mcp import Client

from btcs.events import EventLog
from btcs.mcp_server import TOKEN_HEADER, build_server
from btcs.runner import SimulationRunner
from btcs.world.ledger import EntropyLedger

EXPECTED_TOOLS = {
    "sense_gradient", "ingest_energy", "dissipate_entropy",
    "perform_work", "replicate", "rest", "remember", "recall",
}


@pytest.fixture
def live(small_config, tmp_path):
    runner = SimulationRunner(small_config, log=EventLog(tmp_path / "e.jsonl"))
    runner._found()
    runner.service.begin_tick(1, EntropyLedger())
    return runner


def call(server, name: str, arguments: dict) -> dict:
    """Invoke one tool over an in-process MCP session and parse the payload."""

    async def _run() -> dict:
        async with Client(server) as client:
            result = await client.call_tool(name, arguments)
            raw = result.content[0].text if result.content else ""
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Surfaced rather than swallowed: a non-JSON body means the
                # refusal escaped the guard and became a protocol error.
                return {"ok": False, "error": "non_json_response", "raw": raw}

    return anyio.run(_run)


def list_tools(server) -> list:
    async def _run():
        async with Client(server) as client:
            return (await client.list_tools()).tools

    return anyio.run(_run)


class TestToolSurface:
    def test_expected_tools_are_exposed(self, live):
        names = {t.name for t in list_tools(build_server(live.service))}
        assert names == EXPECTED_TOOLS

    def test_no_tool_exposes_an_agent_id(self, live):
        """The plan's signature took one; reintroducing it would restore the
        exploit where any organism can act as any other."""
        for tool in list_tools(build_server(live.service)):
            properties = (tool.input_schema or {}).get("properties", {})
            assert "agent_id" not in properties, f"{tool.name} exposes agent_id"

    def test_context_is_not_a_model_settable_argument(self, live):
        """The SDK decides model-facing status by annotation. An `Any`-typed
        context silently becomes a tool parameter, which this catches."""
        for tool in list_tools(build_server(live.service)):
            properties = (tool.input_schema or {}).get("properties", {})
            assert "ctx" not in properties, f"{tool.name} leaks its Context"

    def test_metabolic_charging_is_not_exposed(self, live):
        names = {t.name for t in list_tools(build_server(live.service))}
        assert not names & {"charge_cognition", "charge_basal", "check_death"}

    def test_every_tool_is_described(self, live):
        for tool in list_tools(build_server(live.service)):
            assert tool.description and len(tool.description) > 20


class TestIdentityOverTheWire:
    def test_valid_token_senses(self, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        server = build_server(live.service)

        with live.service.turn(agent):
            payload = call(server, "sense_gradient", {"token": token})
        assert payload["ok"] is True

    def test_unknown_token_is_refused(self, live):
        server = build_server(live.service)
        payload = call(server, "sense_gradient", {"token": "tk_forged"})
        assert payload["ok"] is False
        assert payload["error"] == "unknown_token"

    def test_missing_credential_is_refused(self, live):
        payload = call(build_server(live.service), "sense_gradient", {})
        assert payload["ok"] is False
        assert payload["error"] == "unknown_token"

    def test_acting_out_of_turn_is_refused(self, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        payload = call(build_server(live.service), "rest", {"token": token})
        assert payload["error"] == "not_your_turn"

    def test_one_organism_cannot_act_as_another(self, live):
        a, b = live.registry.living()[:2]
        token_a = live.service.issue_token(a)
        server = build_server(live.service)

        with live.service.turn(b):
            payload = call(server, "dissipate_entropy",
                           {"token": token_a, "bits": 10.0})
        assert payload["error"] == "not_your_turn"


class TestRefusalsAreData:
    def test_refusals_come_back_as_results(self, live):
        """An organism is meant to receive and adapt to a refusal, so it must
        arrive as a value rather than as a protocol error."""
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        server = build_server(live.service)

        with live.service.turn(agent):
            payload = call(server, "dissipate_entropy", {"token": token, "bits": -1.0})
        assert payload["ok"] is False
        assert payload["error"] == "invalid_argument"

    def test_refusals_carry_no_traceback(self, live):
        """A stack trace in a percept would hand the organism information about
        the implementation that no amount of sensing should provide."""
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        server = build_server(live.service)

        with live.service.turn(agent):
            payload = call(server, "ingest_energy", {"token": token, "joules": -3.0})

        assert payload["ok"] is False
        blob = json.dumps(payload).lower()
        for leak in ("traceback", "file \"", "line ", ".py", "btcs."):
            assert leak not in blob, f"refusal leaked {leak!r}"

    def test_error_codes_are_from_the_closed_set(self, live):
        from btcs.sim.service import ErrorCode

        valid = {str(code) for code in ErrorCode}
        payload = call(build_server(live.service), "rest", {"token": "tk_nope"})
        assert payload["error"] in valid


class TestActionsOverTheWire:
    def test_rest_succeeds_and_reports_vitals(self, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        server = build_server(live.service)

        with live.service.turn(agent):
            payload = call(server, "rest", {"token": token})
        assert payload["ok"] and {"energy", "entropy", "free_energy"} <= set(payload)

    def test_movement_changes_position(self, live):
        agent = live.registry.living()[0]
        agent.energy = 500.0
        token = live.service.issue_token(agent)
        server = build_server(live.service)
        before = (agent.y, agent.x)

        with live.service.turn(agent):
            payload = call(server, "perform_work", {"token": token, "dy": 1, "dx": 0})
        assert payload["ok"]
        assert (agent.y, agent.x) != before

    def test_dissipation_lowers_entropy(self, live):
        agent = live.registry.living()[0]
        agent.energy, agent.entropy = 800.0, 60.0
        token = live.service.issue_token(agent)
        server = build_server(live.service)

        with live.service.turn(agent):
            payload = call(server, "dissipate_entropy", {"token": token, "bits": 15.0})
        assert payload["ok"]
        assert agent.entropy < 60.0

    def test_memory_round_trips(self, live):
        agent = live.registry.living()[0]
        token = live.service.issue_token(agent)
        server = build_server(live.service)

        with live.service.turn(agent):
            assert call(server, "remember",
                        {"token": token, "key": "n", "content": "north"})["ok"]
            payload = call(server, "recall", {"token": token, "key": "n"})
        assert payload["content"] == "north"


class TestHeaderIdentity:
    def test_header_is_preferred_over_the_argument(self, live):
        """Over HTTP the credential rides in a header, so it never enters tool
        arguments -- and therefore never enters a model's context."""
        from btcs.mcp_server.server import _identity

        class FakeContext:
            headers = {TOKEN_HEADER: "tk_from_header"}

        assert _identity(FakeContext(), "tk_from_argument") == "tk_from_header"

    def test_argument_is_the_fallback(self):
        from btcs.mcp_server.server import _identity

        class NoHeaders:
            headers = None

        assert _identity(NoHeaders(), "tk_from_argument") == "tk_from_argument"

    def test_absent_credential_raises(self):
        from btcs.sim.service import ToolError
        from btcs.mcp_server.server import _identity

        class NoHeaders:
            headers = None

        with pytest.raises(ToolError):
            _identity(NoHeaders(), None)
