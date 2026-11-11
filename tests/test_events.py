"""Event log writing, reading, and determinism."""

from __future__ import annotations

import json

import numpy as np
import pytest

from btcs.events import EventLog, EventType, read_events


def test_writes_and_reads_back(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.emit(0, EventType.RUN_STARTED, None, seed=42)
        log.emit(1, EventType.INGEST, "Agent_01", joules=12.5)

    events = list(read_events(path))
    assert len(events) == 2
    assert events[0] == {"tick": 0, "type": "run_started", "seed": 42}
    assert events[1] == {"tick": 1, "type": "ingest", "agent": "Agent_01", "joules": 12.5}


def test_agent_key_omitted_when_absent(tmp_path):
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.emit(0, EventType.TICK_STARTED)
    assert "agent" not in next(iter(read_events(path)))


def test_serialisation_is_deterministic(tmp_path):
    """Identical event streams must produce byte-identical files.

    This is what makes the reproducibility test in Phase 1 possible: two runs
    with the same seed are compared by diffing their logs.
    """
    def write(path):
        with EventLog(path) as log:
            for tick in range(50):
                log.emit(tick, EventType.COGNITION, "Agent_01", tokens=tick, phi=tick * 0.1)

    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a)
    write(b)
    assert a.read_bytes() == b.read_bytes()


def test_keys_are_sorted_within_a_line(tmp_path):
    """Key order must not depend on kwargs order, or diffs become noise."""
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.emit(0, EventType.INGEST, "A", zebra=1, alpha=2)
    line = path.read_text(encoding="utf-8").strip()
    assert line.index('"alpha"') < line.index('"zebra"')


def test_numpy_scalars_are_serialised(tmp_path):
    """The engine is vectorized NumPy, so scalars arriving here are np types."""
    path = tmp_path / "events.jsonl"
    with EventLog(path) as log:
        log.emit(0, EventType.INGEST, "A", energy=np.float64(3.5), count=np.int64(7))

    event = next(iter(read_events(path)))
    assert event["energy"] == 3.5
    assert event["count"] == 7
    assert isinstance(event["count"], int)


def test_flush_leaves_a_readable_partial_log(tmp_path):
    """A crashed run must still leave everything up to the last flush."""
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    log.emit(0, EventType.TICK_STARTED)
    log.flush()
    assert len(list(read_events(path))) == 1
    log.close()


def test_count_tracks_emissions(tmp_path):
    with EventLog(tmp_path / "e.jsonl") as log:
        assert log.count == 0
        log.emit(0, EventType.TICK_STARTED)
        log.emit(0, EventType.TICK_FINISHED)
        assert log.count == 2


def test_emitting_after_close_raises(tmp_path):
    log = EventLog(tmp_path / "e.jsonl")
    log.close()
    with pytest.raises(RuntimeError, match="closed"):
        log.emit(0, EventType.TICK_STARTED)


def test_close_is_idempotent(tmp_path):
    log = EventLog(tmp_path / "e.jsonl")
    log.close()
    log.close()


def test_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "events.jsonl"
    with EventLog(path) as log:
        log.emit(0, EventType.TICK_STARTED)
    assert path.is_file()


def test_malformed_line_reports_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"tick":0}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        list(read_events(path))


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "gappy.jsonl"
    path.write_text('{"tick":0}\n\n{"tick":1}\n', encoding="utf-8")
    assert len(list(read_events(path))) == 2


def test_event_types_are_unique():
    values = [e.value for e in EventType]
    assert len(values) == len(set(values))
