"""A scripted stand-in for the Anthropic client.

Every Phase 3 test runs against this. Nothing in the suite may make a real API
call: tests that cost money are tests that get skipped, and a suite that is
skipped is not a suite.

The shapes mirror the SDK closely enough for the brain's own code paths --
`response.content` blocks with `.type` / `.name` / `.input` / `.id`,
`response.usage` with the four token counters, `response.stop_reason`, and a
`to_dict()` for the replay log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 50
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass(slots=True)
class FakeBlock:
    type: str
    name: str | None = None
    input: dict[str, Any] | None = None
    id: str = "toolu_fake"
    text: str = ""


@dataclass(slots=True)
class FakeResponse:
    content: list[FakeBlock]
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "tool_use"

    def to_dict(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "content": [
                {"type": b.type, "name": b.name, "input": b.input} for b in self.content
            ],
        }


def tool_call(name: str, **arguments: Any) -> FakeResponse:
    """A response that calls exactly one tool."""
    return FakeResponse([FakeBlock("tool_use", name=name, input=arguments or {})])


def text_only(message: str = "I am thinking about it.") -> FakeResponse:
    """A response with no tool call at all -- the model talked instead of acting."""
    return FakeResponse([FakeBlock("text", text=message)], stop_reason="end_turn")


class FakeAnthropic:
    """Replays a script of responses and records every request it was given."""

    def __init__(self, script: list[FakeResponse] | None = None,
                 error: Exception | None = None) -> None:
        self.script = list(script or [])
        self.error = error
        self.calls: list[dict[str, Any]] = []

    @property
    def messages(self) -> "FakeAnthropic":
        return self

    def create(self, **params: Any) -> FakeResponse:
        self.calls.append(params)
        if self.error is not None:
            raise self.error
        if not self.script:
            # Running off the end of the script means the brain looped further
            # than the test expected; resting is the least surprising answer.
            return tool_call("rest")
        return self.script.pop(0)


class FakeAuthError(Exception):
    """Stands in for anthropic.AuthenticationError, matched by class name."""


AuthenticationError = FakeAuthError
AuthenticationError.__name__ = "AuthenticationError"


class FakeConnectionError(Exception):
    """Stands in for a transient network failure."""


APIConnectionError = FakeConnectionError
APIConnectionError.__name__ = "APIConnectionError"
