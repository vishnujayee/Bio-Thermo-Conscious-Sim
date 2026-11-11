"""The Anthropic-backed brain: the phase where thinking starts costing money.

One decision per tick. The percept is already in the prompt, so a decisive
organism needs exactly one request; calling a deliberation tool first costs
another round-trip, whose real tokens become real entropy through law L3. That
is the whole design in a sentence: **curiosity is metabolically expensive, and
the API bill and the organism's body are readings from the same meter.**

Two structural choices worth stating plainly.

*Deliberation tools loop; action tools end the turn.* `sense_gradient`, `recall`
and `remember` are executed here and fed back, so the model can look before it
leaps. The four action tools terminate deliberation and are returned to the
runner for dispatch. Actions therefore still pass through the service's
one-act-per-turn gate rather than being trusted to this class.

*The organism's credential never enters the model's context.* The tool schemas
carry no token; this brain injects it when it executes a call. A model cannot
alter, leak, or hallucinate a credential it was never shown.

Reproducibility ends here, and the code says so rather than pretending. Model
sampling is not deterministic, so every request and response is logged in full:
an LLM run can be replayed and audited even though it cannot be reproduced.
"""

from __future__ import annotations

import json
from typing import Any

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
from btcs.brains.base import TurnBridge
from btcs.world.types import (
    Decision,
    Dissipate,
    Ingest,
    Move,
    Percept,
    Replicate,
    Rest,
)


class LLMBrain:
    """Drives an organism with a Claude model, billed in dollars and in bits."""

    __slots__ = ("_config", "_rng", "_log", "_client", "budget", "_tick")

    def __init__(
        self,
        config: Config,
        rng: RngService,
        log: EventLog,
        *,
        client: Any | None = None,
        budget: BudgetGuard | None = None,
    ) -> None:
        self._config = config
        self._rng = rng
        self._log = log
        self._tick = 0

        self._client = client if client is not None else self._build_client(config)
        self.budget = budget if budget is not None else BudgetGuard(
            ceiling_usd=config.cognition.budget_usd,
            halt_on_exceed=config.cognition.halt_on_budget,
        )

    @staticmethod
    def _build_client(config: Config) -> Any:
        """Construct the SDK client, resolving credentials from the environment.

        Imported lazily so the whole project stays importable -- and the mock
        brain stays runnable -- on a machine with no `anthropic` installed and no
        API key configured.
        """
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "the anthropic SDK is required for cognition.brain='llm'; "
                "pip install anthropic"
            ) from exc

        return anthropic.Anthropic(
            timeout=config.cognition.request_timeout_seconds,
            max_retries=config.cognition.max_retries,
        )

    @property
    def name(self) -> str:
        return "llm"

    def on_death(self, agent: Agent) -> None:
        return None

    def set_tick(self, tick: int) -> None:
        """Told by the runner, so log lines land on the right tick."""
        self._tick = tick

    # -- the turn ----------------------------------------------------------

    def decide(
        self, percept: Percept, agent: Agent, bridge: TurnBridge | None = None
    ) -> Decision:
        cognition = self._config.cognition
        tier = cognition.tier_for(agent.phi)

        memory = bridge.memory() if bridge is not None else {}
        messages: list[dict[str, Any]] = [
            user_message(percept, memory, self._config)
        ]
        system = system_blocks(agent, self._config)
        fingerprint = prefix_fingerprint(agent, self._config)

        total = Usage()
        rounds = 0

        while rounds < cognition.max_tool_rounds:
            rounds += 1
            response = self._request(system, messages, tier, agent, fingerprint, rounds)
            if response is None:
                # The call failed after retries. Charge for nothing that did not
                # happen and let the organism rest; the failure is logged.
                return self._fallback(total, "request_failed")

            usage = Usage.from_response(response.usage)
            total = total + usage
            self._bill(usage, tier.model, agent, response, fingerprint, rounds)

            action, deliberations = self._interpret(response)

            if action is not None:
                return Decision(
                    action,
                    tokens=self._entropy_tokens(total),
                    note=f"{tier.model}/{rounds}r",
                    metadata=self._metadata(total, tier.model, rounds, fingerprint),
                )

            if not deliberations:
                # No tool call at all: the model talked instead of acting.
                return self._fallback(total, "no_tool_call", tier=tier, rounds=rounds,
                                      fingerprint=fingerprint)

            if bridge is None:
                return self._fallback(total, "no_bridge", tier=tier, rounds=rounds,
                                      fingerprint=fingerprint)

            # Feed every result back in one user message. Splitting them teaches
            # the model to stop issuing parallel calls.
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(bridge.call(block.name, dict(block.input or {}))),
                }
                for block in deliberations
            ]})

        # Deliberated past the round limit without committing. Resting is the
        # honest outcome: the organism spent its tokens and decided nothing.
        return self._fallback(total, "rounds_exhausted", tier=tier,
                              rounds=rounds, fingerprint=fingerprint)

    # -- request -----------------------------------------------------------

    def _request(
        self,
        system: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        tier: Any,
        agent: Agent,
        fingerprint: str,
        round_number: int,
    ) -> Any | None:
        """One API call. Returns None if it failed in a way worth surviving."""
        params: dict[str, Any] = {
            "model": tier.model,
            "max_tokens": self._config.cognition.max_output_tokens,
            "system": system,
            "messages": messages,
            "tools": TOOLS,
        }

        # Effort and adaptive thinking are Opus/Sonnet-tier features. A tier with
        # no effort (Haiku) takes neither -- sending `effort` there is an error,
        # and Haiku's thinking uses the older budget_tokens shape we do not want.
        if tier.effort is not None:
            params["thinking"] = {"type": "adaptive"}
            params["output_config"] = {"effort": tier.effort}

        try:
            return self._client.messages.create(**params)
        except Exception as exc:  # noqa: BLE001 - classified below
            return self._handle_error(exc, agent, fingerprint, round_number)

    def _handle_error(
        self, exc: Exception, agent: Agent, fingerprint: str, round_number: int
    ) -> None:
        """Distinguish a broken configuration from a bad afternoon on the network.

        Authentication and request-shape failures are the operator's problem and
        will recur on every subsequent call, so they stop the run. Transient
        failures have already been retried by the SDK; the organism loses the
        turn and the simulation continues.
        """
        name = type(exc).__name__
        fatal = {
            "AuthenticationError",
            "PermissionDeniedError",
            "NotFoundError",
            "BadRequestError",
        }

        self._log.emit(
            self._tick, EventType.MODEL_CALL, agent.id,
            ok=False, error=name, detail=str(exc)[:500],
            prefix=fingerprint, round=round_number,
        )

        if name in fatal:
            raise
        return None

    def _bill(
        self,
        usage: Usage,
        model: str,
        agent: Agent,
        response: Any,
        fingerprint: str,
        round_number: int,
    ) -> None:
        """Charge the budget and log the call. May raise BudgetExceeded."""
        cost = cost_usd(usage, model)

        if self._config.cognition.log_model_calls:
            self._log.emit(
                self._tick, EventType.MODEL_CALL, agent.id,
                ok=True, model=model, round=round_number, prefix=fingerprint,
                cost_usd=round(cost, 8), stop_reason=getattr(response, "stop_reason", None),
                **usage.to_dict(),
                # The response itself, so a run that cannot be reproduced can at
                # least be replayed and audited.
                response=_serialise(response),
            )

        self.budget.record(usage, model, agent.id)

    # -- interpretation ----------------------------------------------------

    def _interpret(self, response: Any) -> tuple[Any | None, list[Any]]:
        """Split a response into the action it chose and the lookups it wants.

        Parallel tool use is on by default, so a model may ask for several things
        at once. If any of them is an action, that action wins and the turn ends;
        the accompanying lookups are discarded, which is the correct reading of
        an organism that has already decided.
        """
        blocks = [b for b in getattr(response, "content", []) if getattr(b, "type", None) == "tool_use"]

        for block in blocks:
            if block.name in ACTION_TOOLS:
                action = self._to_action(block.name, dict(block.input or {}))
                if action is not None:
                    return action, []

        return None, [b for b in blocks if b.name not in ACTION_TOOLS]

    def _to_action(self, name: str, arguments: dict[str, Any]) -> Any | None:
        """Map a tool call to a domain action.

        Arguments are parsed rather than trusted: current models may vary their
        JSON escaping, and a value that will not cast to a number is a refusal,
        not a crash.
        """
        try:
            match name:
                case "rest":
                    return Rest()
                case "replicate":
                    return Replicate()
                case "ingest_energy":
                    return Ingest(float(arguments["joules"]))
                case "dissipate_entropy":
                    return Dissipate(float(arguments["bits"]))
                case "perform_work":
                    return Move(int(arguments["dy"]), int(arguments["dx"]))
        except (KeyError, TypeError, ValueError):
            return None
        return None

    # -- accounting --------------------------------------------------------

    def _entropy_tokens(self, usage: Usage) -> int:
        """Weighted tokens for law L3. See CognitionConfig for the weights."""
        return max(1, round(entropy_tokens(usage, self._config.cognition)))

    def _metadata(
        self, usage: Usage, model: str, rounds: int, fingerprint: str
    ) -> dict[str, Any]:
        return {
            "model": model,
            "rounds": rounds,
            "prefix": fingerprint,
            "cost_usd": cost_usd(usage, model),
            "cache_hit_rate": usage.cache_hit_rate,
            **usage.to_dict(),
        }

    def _fallback(
        self,
        usage: Usage,
        reason: str,
        *,
        tier: Any = None,
        rounds: int = 0,
        fingerprint: str = "",
    ) -> Decision:
        """Rest, but pay for whatever thinking already happened.

        Charging for a wasted turn is the point. An organism that deliberates
        itself past the round limit has genuinely spent the tokens, and the
        entropy from those tokens is what makes indecision selectable against.
        """
        model = tier.model if tier is not None else "unknown"
        return Decision(
            Rest(),
            tokens=self._entropy_tokens(usage) if usage.total_input else 0,
            note=f"fallback:{reason}",
            metadata={"fallback": reason, "model": model, "rounds": rounds,
                      "prefix": fingerprint, **usage.to_dict()},
        )


def _serialise(response: Any) -> Any:
    """Best-effort structured form of a response, for the replay log."""
    for method in ("to_dict", "model_dump"):
        if hasattr(response, method):
            try:
                return getattr(response, method)()
            except Exception:  # pragma: no cover - SDK shape dependent
                continue
    return repr(response)[:4000]
