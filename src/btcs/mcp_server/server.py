"""MCP surface over a running simulation.

This module is intentionally thin. Every tool is a few lines that read the
caller's identity, forward to `SimulationService`, and translate a refusal into
a structured result. All the enforcement -- identity, turn discipline, budgets,
argument validation -- lives in the service, because the internal tick loop uses
that same service and both paths must obey exactly the same rules.

Three notes on the SDK, since the project plan predates it:

  * The plan's boilerplate imports `FastMCP` from `mcp.server.fastmcp`. That is
    the v1 API; in `mcp` 2.x the class was renamed to `MCPServer` and the v1
    import path raises. We target 2.x.
  * The plan's tools take `agent_id` as a parameter. They must not: an id is
    guessable, so any organism could act as -- or kill -- any other simply by
    naming it. Identity here is an unguessable capability token, resolved
    server-side, and read preferentially from a transport header so that it
    need never appear in tool arguments at all.
  * `ctx` must be annotated `Context`, not `Any`. The SDK decides whether a
    parameter is model-facing by its annotation, so an `Any`-typed context
    silently becomes a tool argument a model can set.
"""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver import MCPServer

from btcs.sim.service import ErrorCode, SimulationService, ToolError

# Header carrying an organism's capability over HTTP transports. When present it
# is authoritative, which keeps the credential out of tool arguments -- and
# therefore out of any model's context in Phase 3.
TOKEN_HEADER = "x-btcs-token"


def _identity(ctx: Context | None, token: str | None) -> str:
    """Resolve the caller's credential: header first, argument as fallback.

    The header is preferred because a credential in an argument is a credential
    a language model can read, alter, or hallucinate. The argument path exists
    for in-process transports, which carry no headers.
    """
    headers: Any = getattr(ctx, "headers", None)
    if headers:
        try:
            from_header = headers.get(TOKEN_HEADER) or headers.get(TOKEN_HEADER.upper())
        except Exception:  # pragma: no cover - mapping shape varies by transport
            from_header = None
        if from_header:
            return str(from_header)

    if token:
        return token

    raise ToolError(
        ErrorCode.UNKNOWN_TOKEN,
        f"No credential supplied; send the {TOKEN_HEADER} header or a token argument.",
    )


def build_server(
    service: SimulationService, *, name: str = "ThermoConsciousSimulator"
) -> MCPServer:
    """Wrap a live simulation in an MCP server.

    The service is captured by closure rather than reached through a module
    global, so several isolated simulations can run in one process -- which the
    tests rely on -- without sharing state behind everyone's back.
    """
    mcp = MCPServer(
        name=name,
        instructions=(
            "You are a digital organism in an open thermodynamic system. You persist only "
            "while your free energy F = E - T*S*ln2 stays positive. Harvest free energy from "
            "hot nodes, and erase accumulated internal entropy before it overwhelms you -- "
            "but erasure costs energy (Landauer), and harvesting is capped by the temperature "
            "gradient between the node and your own cell, so working or thinking in one place "
            "warms it and reduces what you can extract there. Thinking itself generates entropy "
            "in proportion to the tokens you spend. Deliberate accordingly."
        ),
    )

    def _guard(fn, ctx: Context, token: str | None, *args: Any) -> dict[str, Any]:
        """Resolve identity, call the service, turn a refusal into data.

        Identity resolution happens *inside* the try, not as an argument to it:
        a missing credential is itself a refusal an organism should receive as a
        result, and evaluating it outside would let it escape as a protocol
        error instead.

        Anything that is not a ToolError propagates untouched. An internal bug
        must not be laundered into a message an organism could learn to trigger
        deliberately.
        """
        try:
            return fn(_identity(ctx, token), *args)
        except ToolError as refusal:
            return refusal.to_dict()

    @mcp.tool(
        description=(
            "Report local free-energy nodes, thermal conditions, and your own vitals. "
            "Readings degrade as your internal entropy rises."
        )
    )
    def sense_gradient(ctx: Context, token: str | None = None) -> dict[str, Any]:
        return _guard(service.sense_gradient, ctx, token)

    @mcp.tool(
        description=(
            "Draw free energy from the node you are standing on. Yield is capped by the "
            "Carnot bound on the gradient between the node and your own cell, so a node "
            "at thermal equilibrium gives nothing however full it is."
        )
    )
    def ingest_energy(
        joules: float, ctx: Context, token: str | None = None
    ) -> dict[str, Any]:
        return _guard(service.ingest_energy, ctx, token, joules)

    @mcp.tool(
        description=(
            "Erase internal entropy. Costs energy at the Landauer price n*k_B*T*ln2, "
            "so forgetting is never free and is dearest when you are hot."
        )
    )
    def dissipate_entropy(
        bits: float, ctx: Context, token: str | None = None
    ) -> dict[str, Any]:
        return _guard(service.dissipate_entropy, ctx, token, bits)

    @mcp.tool(
        description=(
            "Move one cell (dy, dx each in -1..1, diagonals allowed). Costs energy and "
            "deposits waste heat along the way, warming your own surroundings."
        )
    )
    def perform_work(
        dy: int, dx: int, ctx: Context, token: str | None = None
    ) -> dict[str, Any]:
        return _guard(service.perform_work, ctx, token, dy, dx)

    @mcp.tool(
        description=(
            "Spend energy to produce a mutated copy of yourself. You pay a construction "
            "cost that becomes waste heat in your own cell, hand a share of what remains "
            "to the child, and divide your accumulated entropy with it. Nothing you spend "
            "here comes back to you."
        )
    )
    def replicate(ctx: Context, token: str | None = None) -> dict[str, Any]:
        return _guard(service.replicate, ctx, token)

    @mcp.tool(description="Take no action this cycle, paying only the basal metabolic cost.")
    def rest(ctx: Context, token: str | None = None) -> dict[str, Any]:
        return _guard(service.rest, ctx, token)

    @mcp.tool(description="Store an item in your own memory under a key.")
    def remember(
        key: str, content: str, ctx: Context, token: str | None = None
    ) -> dict[str, Any]:
        return _guard(service.remember, ctx, token, key, content)

    @mcp.tool(
        description=(
            "Read your memory. Omit the key to list what you hold. Recall becomes "
            "unreliable once your entropy passes the confusion threshold."
        )
    )
    def recall(
        ctx: Context, key: str | None = None, token: str | None = None
    ) -> dict[str, Any]:
        return _guard(service.recall, ctx, token, key)

    return mcp
