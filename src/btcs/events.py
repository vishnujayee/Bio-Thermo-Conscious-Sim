"""Event-sourced run log.

Every tick, tool call, metabolic transaction, birth and death is appended to a
JSONL file. That file is the run: the dashboard, the analysis notebooks, and
the replay tooling all read it rather than reaching into live objects. Nothing
else may be the source of truth, because anything held only in memory is gone
the moment a 6-hour run crashes.

The log is also the reproducibility check -- two runs with the same seed and
config must produce identical files, so writes are deterministic and carry no
wall-clock timestamps in the hashed portion.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self


class EventType(StrEnum):
    """Closed vocabulary of loggable events.

    Kept as an enum rather than free strings so a typo in an event name shows up
    at write time, not six hours later when the analysis finds no such events.
    """

    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    TICK_STARTED = "tick_started"
    TICK_FINISHED = "tick_finished"

    # Metabolic transactions
    INGEST = "ingest"
    DISSIPATE = "dissipate"
    WORK = "work"
    BASAL = "basal"
    COGNITION = "cognition"

    # Thermodynamic bookkeeping
    ENTROPY_LEDGER = "entropy_ledger"
    SECOND_LAW_VIOLATION = "second_law_violation"

    # Life cycle
    AGENT_BORN = "agent_born"
    AGENT_DIED = "agent_died"
    DEGRADATION = "degradation"
    PHI_CHANGED = "phi_changed"
    SURPRISE = "surprise"
    REPLICATION = "replication"

    # Cognition / cost
    MODEL_CALL = "model_call"
    BUDGET_EXCEEDED = "budget_exceeded"

    # Environment
    SCENARIO_OVERRIDE = "scenario_override"
    NODE_SPAWNED = "node_spawned"
    NODE_EXHAUSTED = "node_exhausted"


@dataclass(frozen=True, slots=True)
class Event:
    tick: int
    type: EventType
    agent_id: str | None
    data: dict[str, Any]

    def to_json(self) -> str:
        payload: dict[str, Any] = {"tick": self.tick, "type": str(self.type)}
        if self.agent_id is not None:
            payload["agent"] = self.agent_id
        payload.update(self.data)
        # sort_keys so identical events serialise identically across runs and
        # Python versions -- otherwise the reproducibility diff is meaningless.
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_coerce)


def _coerce(value: Any) -> Any:
    """Last-resort serialiser for numpy scalars, which json rejects natively."""
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


class EventLog:
    """Append-only JSONL writer with a per-tick flush.

    Buffers within a tick for throughput, flushes at tick boundaries so that a
    crashed run still leaves a log that ends on a complete tick.
    """

    __slots__ = ("_path", "_handle", "_buffer", "_count", "_closed")

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8", newline="\n")
        self._buffer: list[str] = []
        self._count = 0
        self._closed = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count

    def emit(
        self,
        tick: int,
        type: EventType,
        agent_id: str | None = None,
        /,
        **data: Any,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot emit into a closed EventLog")
        self._buffer.append(Event(tick, type, agent_id, data).to_json())
        self._count += 1

    def flush(self) -> None:
        if not self._buffer:
            return
        self._handle.write("\n".join(self._buffer) + "\n")
        self._handle.flush()
        self._buffer.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._handle.close()
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def read_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream a run log back. Used by replay, analysis, and the reproducibility test."""
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: malformed event line") from exc
