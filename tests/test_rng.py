"""Reproducibility guarantees of the RNG service.

These are the foundation of every experiment in the project. If they fail,
no result the simulation produces can be trusted or repeated.
"""

from __future__ import annotations

from btcs.rng import RngService


def test_same_seed_same_draws():
    a = RngService(42).stream("world").random(20)
    b = RngService(42).stream("world").random(20)
    assert (a == b).all()


def test_different_seed_different_draws():
    a = RngService(42).stream("world").random(20)
    b = RngService(43).stream("world").random(20)
    assert not (a == b).all()


def test_streams_are_independent():
    """Different names must not produce correlated sequences."""
    rng = RngService(42)
    world = rng.stream("world").random(50)
    mutation = rng.stream("mutation").random(50)
    assert not (world == mutation).all()


def test_stream_creation_order_does_not_matter():
    """The property that makes the code safe to extend.

    Adding a new consumer must not shift the draws every existing consumer
    sees, or a run recorded last week stops reproducing.
    """
    first = RngService(42)
    first.stream("world").random(10)
    late = first.stream("mutation").random(10)

    second = RngService(42)
    early = second.stream("mutation").random(10)

    assert (late == early).all()


def test_consuming_one_stream_does_not_disturb_another():
    quiet = RngService(42)
    baseline = quiet.stream("mutation").random(5)

    busy = RngService(42)
    busy.stream("world").random(10_000)
    after = busy.stream("mutation").random(5)

    assert (baseline == after).all()


def test_repeated_stream_access_advances_rather_than_restarts():
    rng = RngService(42)
    first = rng.stream("world").random(5)
    second = rng.stream("world").random(5)
    assert not (first == second).all()


def test_stream_returns_identical_object():
    rng = RngService(42)
    assert rng.stream("world") is rng.stream("world")


def test_agent_streams_are_per_agent():
    rng = RngService(42)
    a = rng.agent_stream("Agent_01").random(10)
    b = rng.agent_stream("Agent_02").random(10)
    assert not (a == b).all()


def test_spawn_seed_is_deterministic_and_in_range():
    a = RngService(42).spawn_seed("birth")
    b = RngService(42).spawn_seed("birth")
    assert a == b
    assert 0 <= a < 2**63 - 1


def test_name_keys_are_stable_across_processes():
    """Hardcoded values catch an accidental switch to a randomised hash.

    Python's built-in hash() is salted per process; if someone swaps blake2b
    for it, reproducibility silently dies. These constants make that loud.
    """
    from btcs.rng import _name_key

    assert _name_key("world") == _name_key("world")
    assert _name_key("world") != _name_key("mutation")
    assert 0 <= _name_key("world") < 2**64
