"""Seeded randomness, split into independent named streams.

Reproducibility is a hard requirement: two runs with the same seed and config
must produce byte-identical event logs. A single shared generator technically
satisfies that, but it is brittle in a way that ruins long experiments -- if you
later add one extra draw in the world module, every subsequent draw in the
selection module shifts, and a run you recorded last week can no longer be
reproduced.

So each subsystem gets its own stream, derived from the master seed by name.
Streams are independent and order-insensitive: adding a draw to "world" cannot
disturb "mutation", and asking for a stream late gives the same generator as
asking for it early.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _name_key(name: str) -> int:
    """Map a stream name to a stable 64-bit spawn key.

    Deliberately not Python's hash(): that is randomised per process unless
    PYTHONHASHSEED is pinned, which would make reproducibility depend on an
    environment variable nobody remembers to set.
    """
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class RngService:
    """Deterministic generator factory keyed by subsystem name."""

    __slots__ = ("_master_seed", "_streams")

    def __init__(self, master_seed: int) -> None:
        self._master_seed = int(master_seed)
        self._streams: dict[str, np.random.Generator] = {}

    @property
    def master_seed(self) -> int:
        return self._master_seed

    def stream(self, name: str) -> np.random.Generator:
        """Return the generator for `name`, creating it on first use.

        The same name always yields the same generator object, so a caller that
        draws repeatedly advances one stream rather than restarting it.
        """
        generator = self._streams.get(name)
        if generator is None:
            seed_seq = np.random.SeedSequence(
                entropy=self._master_seed, spawn_key=(_name_key(name),)
            )
            generator = np.random.default_rng(seed_seq)
            self._streams[name] = generator
        return generator

    def agent_stream(self, agent_id: str) -> np.random.Generator:
        """Per-agent stream, so one organism's draws never perturb another's."""
        return self.stream(f"agent:{agent_id}")

    def spawn_seed(self, name: str) -> int:
        """A fresh integer seed derived from a stream.

        Used when a child organism needs its own reproducible seed at birth.
        """
        return int(self.stream(name).integers(0, 2**63 - 1))

    def __repr__(self) -> str:
        return f"RngService(master_seed={self._master_seed}, streams={len(self._streams)})"
