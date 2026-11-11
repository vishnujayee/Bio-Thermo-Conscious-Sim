"""A brain driven by a model running on this machine, through Ollama.

This exists so the interesting experiment -- real models as minds -- can be run
by anyone, on any laptop, without a credential or a bill. It is not a downgrade
of the Anthropic brain. In one respect it is a better fit for what the project
is about.

**With an API, cost is dollars, and dollars are a stand-in for thermodynamic
cost. Here the cost is your own hardware actually working.** Every request
reports how long it really took, and `compute_seconds_entropy_weight` turns
those seconds into internal entropy. Set it above zero and a slow thought
genuinely cooks the organism that had it -- the metaphor stops being a metaphor,
because the heat is real and it is in the room with you.

Token accounting still runs through the tier's Claude price, virtually. Nothing
is spent, but the budget guard, the cost reports and the Φ economics behave
exactly as they would on the API, so an experiment developed here transfers
without changing a line.

One accommodation for small models: they frequently emit a tool call with the
required arguments missing -- `ingest_energy` with no `joules`. Voiding the turn
would be defensible, but it would also mean a 3B model spends most of its life
resting, and we would learn nothing. `local_repair_arguments` fills a sensible
default instead: a vague intention is still a choice, and the organism still
pays for it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from btcs.brains.base import TurnBridge
from btcs.cognition.budget import BudgetGuard
from btcs.cognition.pricing import Usage, cost_usd, entropy_tokens
from btcs.cognition.prompt import (
    ACTION_TOOLS,
    TOOLS,
    prefix_fingerprint,
    system_blocks,
    user_message,
)
from btcs.config import Config
from btcs.events import EventLog, EventType
from btcs.rng import RngService
from btcs.world.agents import Agent
from btcs.world.types import (
    Decision,
    Dissipate,
    Ingest,
    Move,
    Percept,
    Replicate,
    Rest,
)


def to_ollama_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate our tool definitions into Ollama's OpenAI-shaped schema.

    The definitions themselves are shared with the Anthropic brain, so both
    minds see exactly the same affordances and a comparison between them is
    about the model rather than about two different tool descriptions.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


class OllamaTransport:
    """Minimal JSON-over-HTTP client. Stdlib only, so there is nothing to install."""

    __slots__ = ("base_url", "timeout")

    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def available(self) -> bool:
        """Whether a server is answering, so failure is diagnosable up front."""
        try:
            request = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(request, timeout=5):
                return True
        except Exception:
            return False


class LocalBrain:
    """Runs an organism on a locally served model. Costs time, not money."""

    __slots__ = ("_config", "_rng", "_log", "_transport", "budget", "_tick",
                 "compute_seconds", "_ollama_tools", "_seen_prefixes")

    def __init__(
        self,
        config: Config,
        rng: RngService,
        log: EventLog,
        *,
        transport: Any | None = None,
        budget: BudgetGuard | None = None,
    ) -> None:
        self._config = config
        self._rng = rng
        self._log = log
        self._tick = 0
        self.compute_seconds = 0.0
        self._ollama_tools = to_ollama_tools(TOOLS)
        # Prefixes this brain has already read, for the stability discount.
        self._seen_prefixes: set[str] = set()

        self._transport = transport if transport is not None else OllamaTransport(
            config.cognition.local_base_url, config.cognition.local_timeout_seconds
        )
        self.budget = budget if budget is not None else BudgetGuard(
            ceiling_usd=config.cognition.budget_usd,
            halt_on_exceed=config.cognition.halt_on_budget,
        )

    @property
    def name(self) -> str:
        return "local"

    def on_death(self, agent: Agent) -> None:
        return None

    def set_tick(self, tick: int) -> None:
        self._tick = tick

    def _model_for(self, tier: Any) -> str:
        return tier.local_model or self._config.cognition.local_model

    # -- the turn ----------------------------------------------------------

    def decide(
        self, percept: Percept, agent: Agent, bridge: TurnBridge | None = None
    ) -> Decision:
        cognition = self._config.cognition
        tier = cognition.tier_for(agent.phi)
        model = self._model_for(tier)

        memory = bridge.memory() if bridge is not None else {}
        # Ollama takes a flat message list, so the two system blocks that the
        # Anthropic path keeps separate for caching are joined here. The content
        # is identical -- only the transport shape differs.
        system = "\n\n".join(b["text"] for b in system_blocks(agent, self._config))
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user",
             "content": user_message(percept, memory, self._config)["content"][0]["text"]},
        ]
        fingerprint = prefix_fingerprint(agent, self._config)

        total = Usage()
        seconds = 0.0
        rounds = 0

        while rounds < cognition.max_tool_rounds:
            rounds += 1
            started = time.perf_counter()
            try:
                response = self._transport.chat(
                    {"model": model, "stream": False,
                     "messages": messages, "tools": self._ollama_tools}
                )
            except Exception as exc:  # noqa: BLE001 - any transport failure
                elapsed = time.perf_counter() - started
                seconds += elapsed
                self.compute_seconds += elapsed
                self._log.emit(
                    self._tick, EventType.MODEL_CALL, agent.id,
                    ok=False, error=type(exc).__name__, detail=str(exc)[:400],
                    model=model, round=rounds, seconds=round(elapsed, 3),
                )
                return self._fallback(total, seconds, "request_failed", tier, rounds, fingerprint)

            elapsed = time.perf_counter() - started
            seconds += elapsed
            self.compute_seconds += elapsed

            # Everything after the system block is volatile: this tick's percept,
            # and any tool results collected during deliberation. It grows each
            # round, which is correct -- a tool result is new information.
            volatile_text = "".join(
                str(message.get("content", "")) for message in messages[1:]
            )
            usage = self._usage_from(response, volatile_text, fingerprint)
            total = total + usage
            self._bill(usage, tier, model, agent, response, fingerprint, rounds, elapsed)

            action, deliberations = self._interpret(response, percept)
            if action is not None:
                return Decision(
                    action,
                    tokens=self._entropy_tokens(total, seconds),
                    note=f"{model}/{rounds}r/{seconds:.1f}s",
                    metadata=self._metadata(total, tier, model, rounds, fingerprint, seconds),
                )

            if not deliberations or bridge is None:
                return self._fallback(total, seconds, "no_tool_call", tier, rounds, fingerprint)

            messages.append(response.get("message", {"role": "assistant", "content": ""}))
            for call in deliberations:
                function = call.get("function", {})
                result = bridge.call(function.get("name", ""), _arguments(function))
                messages.append({"role": "tool", "content": json.dumps(result)})

        return self._fallback(total, seconds, "rounds_exhausted", tier, rounds, fingerprint)

    # -- interpretation ----------------------------------------------------

    def _interpret(
        self, response: dict[str, Any], percept: Percept
    ) -> tuple[Any | None, list[dict[str, Any]]]:
        calls = (response.get("message") or {}).get("tool_calls") or []
        for call in calls:
            name = (call.get("function") or {}).get("name", "")
            if name in ACTION_TOOLS:
                action = self._to_action(name, _arguments(call.get("function") or {}), percept)
                if action is not None:
                    return action, []
        return None, [
            c for c in calls
            if ((c.get("function") or {}).get("name", "")) not in ACTION_TOOLS
        ]

    def _to_action(self, name: str, arguments: dict[str, Any], percept: Percept) -> Any | None:
        """Map a tool call to an action, repairing missing arguments if allowed.

        The defaults are chosen to express the obvious reading of the intent: an
        organism that says "eat" without saying how much means "as much as I
        can"; one that says "erase" means "enough to get back under control".
        """
        repair = self._config.cognition.local_repair_arguments
        try:
            match name:
                case "rest":
                    return Rest()
                case "replicate":
                    return Replicate()
                case "ingest_energy":
                    if "joules" in arguments:
                        return Ingest(float(arguments["joules"]))
                    return Ingest(self._config.physics.ingest_rate_limit) if repair else None
                case "dissipate_entropy":
                    if "bits" in arguments:
                        return Dissipate(float(arguments["bits"]))
                    if not repair:
                        return None
                    excess = max(1.0, percept.entropy - self._config.agent.s_warn * 0.5)
                    return Dissipate(min(excess, self._config.agent.max_dissipate_bits))
                case "perform_work":
                    if "dy" in arguments or "dx" in arguments:
                        return Move(int(arguments.get("dy", 0) or 0),
                                    int(arguments.get("dx", 0) or 0))
                    if not repair:
                        return None
                    # No direction given: head for the best thing in sight, or
                    # stand still if there is nothing worth walking to.
                    if percept.nodes:
                        best = percept.nodes[0]
                        return Move(_sign(best.dy), _sign(best.dx))
                    return None
        except (TypeError, ValueError):
            return None
        return None

    # -- accounting --------------------------------------------------------

    def _usage_from(
        self, response: dict[str, Any], volatile_text: str, fingerprint: str
    ) -> Usage:
        """Split reported input into the stable prefix and the volatile remainder.

        Ollama has no prompt cache, so it reports every token as fresh. Charging
        full entropy for a prefix the organism re-reads verbatim every tick made
        local organisms accumulate disorder about fourteen times faster than the
        mock brain and die within three ticks -- an artefact of the transport,
        not a fact about cognition.

        The D-9 discount was never really about billing. It is about information
        stability: re-reading an unchanged block is not new cognitive work,
        whoever did or did not meter it. So after the first sighting of a prefix,
        its tokens are counted as a cache read, which is exactly what the API
        path would report.

        **Measure the volatile side, never the stable one.** The obvious
        implementation estimates the prefix and subtracts, and it is wrong: the
        estimator overshot the real tokenizer by enough that the estimated
        prefix exceeded the *entire* prompt, so the percept was absorbed into
        the discount and the organism read the world for a tenth of the entropy
        it should have cost. Estimating the volatile part instead puts the
        estimation error on the prefix, where over-charging is the safe
        direction: perception must never come out free.
        """
        usage = _raw_usage(response)
        if not self._config.cognition.local_simulate_cache:
            return usage

        from btcs.cognition.estimate import approx_tokens

        volatile = min(approx_tokens(volatile_text), usage.input_tokens)
        prefix_tokens = usage.input_tokens - volatile

        first_time = fingerprint not in self._seen_prefixes
        self._seen_prefixes.add(fingerprint)

        return Usage(
            input_tokens=volatile,
            output_tokens=usage.output_tokens,
            cache_creation_input_tokens=prefix_tokens if first_time else 0,
            cache_read_input_tokens=0 if first_time else prefix_tokens,
        )

    def _bill(
        self, usage: Usage, tier: Any, model: str, agent: Agent,
        response: dict[str, Any], fingerprint: str, rounds: int, elapsed: float,
    ) -> None:
        """Charge the *virtual* budget and log the call, including real seconds."""
        priced_as = tier.model if self._config.cognition.virtual_pricing else model
        cost = cost_usd(usage, priced_as)

        if self._config.cognition.log_model_calls:
            self._log.emit(
                self._tick, EventType.MODEL_CALL, agent.id,
                ok=True, model=model, priced_as=priced_as, virtual=True,
                round=rounds, prefix=fingerprint, seconds=round(elapsed, 3),
                cost_usd=round(cost, 8), **usage.to_dict(),
                response=(response.get("message") or {}).get("tool_calls"),
            )

        self.budget.record(usage, priced_as, agent.id)

    def _entropy_tokens(self, usage: Usage, seconds: float) -> int:
        """Weighted tokens, plus real compute time when that knob is turned on."""
        cognition = self._config.cognition
        tokens = entropy_tokens(usage, cognition)
        tokens += seconds * cognition.compute_seconds_entropy_weight
        return max(1, round(tokens))

    def _metadata(
        self, usage: Usage, tier: Any, model: str, rounds: int,
        fingerprint: str, seconds: float,
    ) -> dict[str, Any]:
        priced_as = tier.model if self._config.cognition.virtual_pricing else model
        return {
            "model": model, "priced_as": priced_as, "virtual": True,
            "rounds": rounds, "prefix": fingerprint,
            "seconds": round(seconds, 3),
            "cost_usd": cost_usd(usage, priced_as),
            **usage.to_dict(),
        }

    def _fallback(
        self, usage: Usage, seconds: float, reason: str,
        tier: Any, rounds: int, fingerprint: str,
    ) -> Decision:
        model = self._model_for(tier) if tier is not None else "unknown"
        return Decision(
            Rest(),
            tokens=self._entropy_tokens(usage, seconds) if usage.total_input else 0,
            note=f"fallback:{reason}",
            metadata={"fallback": reason, "model": model, "rounds": rounds,
                      "prefix": fingerprint, "seconds": round(seconds, 3),
                      **usage.to_dict()},
        )


def _raw_usage(response: dict[str, Any]) -> Usage:
    """Ollama reports prompt_eval_count and eval_count, and nothing about cache."""
    return Usage(
        input_tokens=int(response.get("prompt_eval_count") or 0),
        output_tokens=int(response.get("eval_count") or 0),
    )


def _arguments(function: dict[str, Any]) -> dict[str, Any]:
    """Ollama may hand back arguments as an object or as a JSON string."""
    raw = function.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _sign(value: int) -> int:
    return 0 if value == 0 else (1 if value > 0 else -1)
