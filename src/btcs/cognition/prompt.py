"""Prompt assembly, arranged around the cache.

Caching is a prefix match: one changed byte invalidates everything after it, and
the render order is `tools` -> `system` -> `messages`. So this module's job is to
sort every piece of context by how often it changes, and physically place the
stable parts first.

    tools          identical for every organism, every tick   <- never changes
    system[0]      the laws of physics                        <- never changes
    system[1]      this organism's genome                     <- per lineage   [breakpoint]
    messages[0]    memory digest, then the tick's percept      <- per tick, uncached

The consequence worth noticing is ecological rather than technical. The laws
block is byte-identical across the entire population, so every organism reads
the same cached prefix; and because clones inherit a genome verbatim, a
successful lineage shares its second block too. **Replication makes a lineage
cheaper to run.** Selection for metabolic efficiency falls out of the billing
system without anyone designing it in -- and once `cache_read_entropy_weight` is
below 1.0, it falls out of the physics as well.

Two rules this file must keep obeying:

  * Nothing volatile may appear before the last breakpoint. No timestamps, no
    tick numbers, no agent ids, no `datetime.now()`. They belong in the message.
  * Tool order must be deterministic, because tools render at position zero and
    a reordered list invalidates the whole prefix for everyone.
"""

from __future__ import annotations

import json
from typing import Any

from btcs.config import Config
from btcs.world.agents import Agent, Degradation
from btcs.world.types import Percept

# --- the frozen block ------------------------------------------------------
#
# Identical for every organism in every run with the same physics. Any value
# interpolated in here would fragment the cache across the population, so the
# numbers below are described qualitatively and the actual constants reach the
# organism through its percept instead.

LAWS = """\
You are a digital organism in an open thermodynamic system. You are not playing a \
game with rules; you are subject to physics, and you persist only while the \
arithmetic works out.

WHAT KEEPS YOU ALIVE
Your free energy is F = E - T*S*ln2, where E is your energy store, S is your \
accumulated internal entropy in bits, and T is the temperature of the cell you \
occupy. When F reaches zero you undergo informational erasure and this ends. \
Note what that formula means: the same body that is comfortably alive in a cold \
cell is already dead in a hot one. Heat does not merely inconvenience you. It \
multiplies your disorder.

THE FOUR THINGS YOU CAN DO
- ingest_energy: draw free energy from a node you are standing on. What you \
actually get is capped by the temperature gradient between the node and your own \
cell. A node in thermal equilibrium with its surroundings yields nothing at all, \
however much energy it still contains. Full is not the same as edible.
- dissipate_entropy: erase accumulated internal entropy. This costs energy in \
proportion to the bits erased and to your local temperature. Forgetting is never \
free, and it is dearest exactly when you are hottest and need it most.
- perform_work: move one cell in any direction, diagonals included. Costs energy \
and leaves waste heat behind you.
- rest: pay only the basal cost of existing.

WHY THINKING IS DANGEROUS
Deliberation is metabolic. Every token you spend generates internal entropy in \
proportion to how much you spend, and the rate scales superlinearly with your \
integration index. Thought also releases waste heat into the cell you occupy, \
which raises your local temperature, which shrinks what you can extract there and \
raises the price of forgetting. An organism that reasons expansively in one place \
is cooking the ground it stands on. Elaborate reasoning is not free caution; it is \
a cost that has killed organisms before you.

HOW TO CONDUCT YOURSELF
You are given your perception before you are asked to act, so in the ordinary case \
you should choose an action immediately and call exactly one action tool. Calling \
sense_gradient, recall or remember first is permitted and sometimes right, but each \
one is another round of tokens and therefore another quantity of entropy you will \
have to pay to erase. Spend that only when the information would actually change \
what you do.

End your turn by calling exactly one of: ingest_energy, dissipate_entropy, \
perform_work, rest."""


def system_blocks(agent: Agent, config: Config) -> list[dict[str, Any]]:
    """Two blocks, stable-first, with the cache breakpoint after the genome.

    The breakpoint sits on the *last* system block, which caches the tools and
    both system blocks together -- tools render before system, so one marker
    covers the whole prefix.
    """
    return [
        {"type": "text", "text": LAWS},
        {
            "type": "text",
            "text": _genome_block(agent),
            "cache_control": {"type": "ephemeral"},
        },
    ]


def _genome_block(agent: Agent) -> str:
    """The heritable half of the prefix.

    Clones share this verbatim, so a lineage shares a cache entry. Note that no
    agent id appears -- an id would make every organism's prefix unique and
    destroy exactly the sharing this is built for.
    """
    policy = agent.genome.policy
    return (
        "YOUR INHERITED STRATEGY\n"
        f"{agent.genome.strategy}\n\n"
        "Dispositions you were born with, as tendencies rather than instructions:\n"
        f"- you begin considering erasure around {policy.dissipate_at:.0f} bits of entropy\n"
        f"- you treat energy below {policy.hunger_at:.0f} as hunger\n"
        f"- your appetite for rich nodes is {policy.greed:.2f}\n"
        f"- your aversion to heat is {policy.heat_aversion:.2f}\n"
        f"- your line tends to replicate once it holds about {policy.replicate_at:.0f} energy\n"
        "You may act against these. They are inheritance, not obligation."
    )


def user_message(percept: Percept, memory: dict[str, str], config: Config) -> dict[str, Any]:
    """The volatile half: everything that changes tick to tick.

    Deliberately a single uncached block placed after the breakpoint. Nothing in
    here is worth caching -- it differs every tick by construction -- and putting
    any of it earlier would invalidate the prefix for the whole population.
    """
    return {"role": "user", "content": [{"type": "text", "text": _situation(percept, memory)}]}


def _situation(percept: Percept, memory: dict[str, str]) -> str:
    lines: list[str] = []

    if memory:
        lines.append("WHAT YOU REMEMBER")
        for key in sorted(memory):
            lines.append(f"- {key}: {memory[key]}")
        lines.append("")

    warning = _degradation_warning(percept.degradation)
    if warning:
        lines.append(warning)
        lines.append("")

    if percept.last_action is not None:
        last = percept.last_action
        lines.append("WHAT CAME OF YOUR LAST ACT")
        if last.ok:
            lines.append(f"- your {last.kind} succeeded.")
        else:
            lines.append(f"- your {last.kind} did NOT happen: {last.detail}")
            lines.append("- repeating it unchanged will fail the same way.")
        lines.append("")

    lines.append("YOUR CONDITION")
    lines.append(f"- energy: {percept.energy:.1f}")
    lines.append(f"- internal entropy: {percept.entropy:.1f} bits")
    lines.append(f"- free energy F: {percept.free_energy:.1f}  (erasure at zero)")
    lines.append(f"- local temperature: {percept.local_temperature:.2f}")
    lines.append(f"- ambient temperature: {percept.ambient_temperature:.2f}")
    lines.append(f"- integration index: {percept.phi:.2f}")
    lines.append(f"- standing on a node: {'yes' if percept.on_node else 'no'}")
    lines.append("")

    if percept.nodes:
        lines.append("WHAT YOU CAN SEE")
        lines.append(
            "Extractable is what you would actually gain after the gradient cap; "
            "the raw store is what remains in the node."
        )
        for node in percept.nodes:
            bearing = _bearing(node.dy, node.dx)
            lines.append(
                f"- {bearing}, {int(node.distance)} cell(s): "
                f"extractable {node.extractable:.1f} of {node.energy:.1f} stored, "
                f"node temperature {node.temperature:.2f}"
            )
    else:
        lines.append("WHAT YOU CAN SEE")
        lines.append("Nothing within range.")
    lines.append("")

    lines.append("Choose your action now. Call exactly one action tool.")
    return "\n".join(lines)


def _degradation_warning(rung: int) -> str:
    """What an organism is told about its own decay.

    It is told that its readings are unreliable, never which ones. An accurate
    error bar would be a form of the very integration that high entropy is
    supposed to have destroyed.
    """
    if rung >= Degradation.DEMENTED:
        return (
            "YOUR STATE\nYour internal entropy is critical. Your perception is "
            "substantially corrupted, your memory is shedding items, and thinking "
            "now costs you more than it did. You cannot tell which of the figures "
            "below are wrong."
        )
    if rung >= Degradation.CONFUSED:
        return (
            "YOUR STATE\nYour entropy is high enough that some of what you would "
            "otherwise perceive and remember is simply missing. You are not told "
            "what is missing."
        )
    if rung >= Degradation.NOISY:
        return (
            "YOUR STATE\nYour entropy has passed the point where your readings "
            "acquire noise. Treat the figures below as approximate."
        )
    return ""


def _bearing(dy: int, dx: int) -> str:
    vertical = "north" if dy < 0 else ("south" if dy > 0 else "")
    horizontal = "west" if dx < 0 else ("east" if dx > 0 else "")
    return (vertical + horizontal) or "here"


# --- tools -----------------------------------------------------------------
#
# Tools render at position zero, so this list must be byte-identical for every
# organism and every tick. It is a module constant rather than a function
# precisely so nothing can accidentally make it vary.
#
# The token is absent by design: it is injected by the brain when the call is
# executed, so an organism's credential never enters a model's context and
# cannot be altered or hallucinated by the model holding it.

DELIBERATION_TOOLS = ("sense_gradient", "recall", "remember")
ACTION_TOOLS = ("ingest_energy", "dissipate_entropy", "perform_work", "replicate", "rest")

TOOLS: list[dict[str, Any]] = [
    {
        "name": "dissipate_entropy",
        "description": (
            "Erase internal entropy. Costs energy proportional to the bits erased "
            "and to your local temperature. Ends your turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "bits": {"type": "number", "description": "Bits of entropy to erase."}
            },
            "required": ["bits"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ingest_energy",
        "description": (
            "Draw free energy from the node you are standing on. Yield is capped by "
            "the temperature gradient between the node and your cell. Ends your turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "joules": {"type": "number", "description": "Energy to attempt to draw."}
            },
            "required": ["joules"],
            "additionalProperties": False,
        },
    },
    {
        "name": "perform_work",
        "description": (
            "Move one cell. dy and dx must each be -1, 0 or 1 and not both zero. "
            "Costs energy and leaves waste heat. Ends your turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "dy": {"type": "integer", "description": "-1 north, 0 none, 1 south."},
                "dx": {"type": "integer", "description": "-1 west, 0 none, 1 east."},
            },
            "required": ["dy", "dx"],
            "additionalProperties": False,
        },
    },
    {
        "name": "recall",
        "description": (
            "Read your memory. Omit the key to list what you hold. Does not end your "
            "turn, but costs another round of tokens and therefore more entropy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Key to read."}},
            "additionalProperties": False,
        },
    },
    {
        "name": "remember",
        "description": (
            "Store something under a key for future ticks. Does not end your turn, "
            "but costs another round of tokens and therefore more entropy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["key", "content"],
            "additionalProperties": False,
        },
    },
    {
        "name": "replicate",
        "description": (
            "Spend energy to produce a mutated copy of yourself. You pay a construction "
            "cost that becomes waste heat in your own cell, and you hand a share of what "
            "remains to the child. The child is born carrying part of your accumulated "
            "entropy, so replicating while disordered passes that disorder on. Nothing "
            "you spend here comes back to you. Ends your turn."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "rest",
        "description": "Do nothing but pay the basal cost of existing. Ends your turn.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "sense_gradient",
        "description": (
            "Re-read your surroundings. You were already given a perception this tick, "
            "so this is usually redundant. Does not end your turn, but costs another "
            "round of tokens and therefore more entropy."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

# Sorted by name at import so the rendered bytes cannot depend on the order
# anyone happened to type them in.
TOOLS.sort(key=lambda tool: tool["name"])


def prefix_fingerprint(agent: Agent, config: Config) -> str:
    """Stable hash of the cacheable prefix.

    Logged with every request so that a cache miss can be traced to the exact
    tick at which the prefix changed, instead of being inferred from a usage
    number that simply stopped rising.
    """
    import hashlib

    blob = json.dumps(
        {
            "tools": [t["name"] for t in TOOLS],
            "laws": LAWS,
            "genome": _genome_block(agent),
        },
        sort_keys=True,
    )
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=8).hexdigest()
