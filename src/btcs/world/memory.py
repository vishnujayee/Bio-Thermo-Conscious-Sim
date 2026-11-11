"""The memory graph, and the Phi-proxy computed over it.

**This is a proxy, not canonical IIT Phi, and the difference matters.** Canonical
Phi is intractable for anything non-trivial: it searches every bipartition of the
system and, for each, computes effective information over the full joint
distribution. What is computed here keeps the shape of that idea and replaces
both intractable parts with stated approximations:

  * **the partition search** becomes a spectral bipartition -- the Fiedler
    vector of the association graph, which is the standard cheap stand-in for a
    minimum-information cut;
  * **effective information** becomes the sum of *pairwise* mutual informations
    across that cut, which is the first-order term of the multi-information and
    the usual tractable approximation to it.

Both approximations are lower bounds in spirit rather than guaranteed bounds.
`docs/phi.md` states the limits at length; nothing here should be read as a claim
about consciousness.

What the measure does capture, and why it is the right shape for this project:

**Perfect correlation is not integration.** A set of items that are always active
together carries no information about each other beyond what either carries
alone, so it scores zero -- as it should. An organism that does the same thing in
every situation has a memory that is *uniform*, not integrated. Integration
requires that knowing one half genuinely tells you something about the other,
which is only possible if there is variety to be informative about.

That is also what makes Phi expensive to hold: variety in experience means acting
differently in different situations, and acting at all costs energy and disorder.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from typing import Any

import numpy as np


class MemoryGraph:
    """One organism's memory: what it holds, and what its experience associated.

    Two things live here that are usually kept apart, because in this project
    they are the same object. `items` is the keyed store an organism writes to
    with `remember`. `history` is the record of which items were active together,
    tick by tick, and it is what the association graph -- and therefore Phi -- is
    computed from.

    An organism's explicit memories are part of its experience, not a separate
    filing cabinet: writing a memory activates it, and a memory never activated
    again contributes nothing to integration however carefully it was worded.
    """

    __slots__ = ("items", "history", "_window", "_phi", "_computed_at")

    def __init__(self, window: int) -> None:
        self.items: dict[str, str] = {}
        self.history: deque[frozenset[str]] = deque(maxlen=window)
        self._window = window
        self._phi = 0.0
        self._computed_at = -1

    # -- writing -----------------------------------------------------------

    def remember(self, key: str, content: str) -> None:
        self.items[key] = content

    def forget(self, key: str) -> None:
        self.items.pop(key, None)

    def activate(self, keys: Iterable[str]) -> None:
        """Record one tick's worth of co-activation.

        An empty tick is still recorded. Silence is data: an organism that does
        nothing for twenty ticks has an experience with very little structure in
        it, and the measure should say so rather than skip the evidence.
        """
        self.history.append(frozenset(keys))

    def corrupt(self, stream: Any, probability: float) -> int:
        """Drop items from remembered episodes. Returns how many were lost.

        This is the degradation ladder reaching into experience itself. It is
        also where the death spiral closes: losing episodes lowers the mutual
        information between the halves of the graph, which lowers Phi, which --
        once Phi is buying degradation resistance -- removes the very protection
        that was slowing the corruption down. Nobody wrote that loop; it is what
        happens when the measure and the damage act on the same structure.
        """
        if probability <= 0.0 or not self.history:
            return 0

        lost = 0
        rebuilt: list[frozenset[str]] = []
        for episode in self.history:
            # `sorted`, not the set's own order. Iterating a set of strings
            # follows hash order, which Python randomises per process -- so this
            # loop would draw from the seeded stream in a different sequence on
            # every run and quietly destroy reproducibility. It did: two runs of
            # the same suite disagreed about which seeds passed.
            kept = {key for key in sorted(episode) if stream.random() >= probability}
            lost += len(episode) - len(kept)
            rebuilt.append(frozenset(kept))
        self.history = deque(rebuilt, maxlen=self._window)
        return lost

    # -- the measure -------------------------------------------------------

    def phi(self, tick: int | None = None, *, recompute_every: int = 1) -> float:
        """Integration across the spectral bipartition, in bits per element.

        Rate-limited by `recompute_every` because the eigen-decomposition is the
        most expensive thing in a tick, and because Phi that changes every tick
        makes an organism's model tier oscillate -- which throws away its cached
        prompt prefix each time it wobbles across a boundary.
        """
        if tick is not None and self._computed_at >= 0:
            if tick - self._computed_at < recompute_every:
                return self._phi

        self._phi = self._compute()
        if tick is not None:
            self._computed_at = tick
        return self._phi

    @property
    def cached_phi(self) -> float:
        return self._phi

    def _compute(self) -> float:
        keys = self._active_keys()
        if len(keys) < 2:
            # One element cannot be integrated with anything, and neither can
            # none. Both are honestly zero rather than undefined.
            return 0.0

        association = self._pairwise_information(keys)
        if not association.any():
            return 0.0

        left, right = _spectral_bipartition(association)
        if not left.any() or not right.any():
            return 0.0

        crossing = float(association[np.ix_(left, right)].sum())
        smaller = min(int(left.sum()), int(right.sum()))
        return crossing / smaller

    def _active_keys(self) -> list[str]:
        """Items that were active at least once and not always.

        An item active in every episode, or in none, carries zero entropy and
        therefore zero information about anything. Excluding them is not a
        convenience -- including them would add rows of zeros that drag the
        spectral partition toward meaningless cuts.
        """
        if not self.history:
            return []
        counts: dict[str, int] = {}
        for episode in self.history:
            for key in episode:
                counts[key] = counts.get(key, 0) + 1
        total = len(self.history)
        return sorted(k for k, n in counts.items() if 0 < n < total)

    def _pairwise_information(self, keys: list[str]) -> np.ndarray:
        """Mutual information between every pair of items, in bits.

        The first-order term of the multi-information: it captures how much
        knowing whether one item was active tells you about another. Higher-order
        structure is invisible to it, which is the main thing this proxy gives up
        relative to canonical Phi.
        """
        episodes = len(self.history)
        index = {key: position for position, key in enumerate(keys)}
        active = np.zeros((episodes, len(keys)), dtype=bool)
        for row, episode in enumerate(self.history):
            for key in episode:
                position = index.get(key)
                if position is not None:
                    active[row, position] = True

        p_on = active.mean(axis=0)
        # Joint counts for the (on, on) cell; the other three follow from the
        # marginals, so one matrix multiply gives the whole contingency table.
        joint_on = (active.T.astype(np.float64) @ active.astype(np.float64)) / episodes

        n = len(keys)
        information = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                bits = _mutual_information(p_on[a], p_on[b], joint_on[a, b])
                information[a, b] = information[b, a] = bits
        return information

    # -- prediction --------------------------------------------------------

    def surprisal(self, previous: frozenset[str], actual: frozenset[str]) -> float:
        """Bits of surprise the next moment carried, given this memory.

        **Why the physics should care.** Still, Sivak, Bell & Crooks (2012) prove
        that a system which models its environment badly *necessarily* dissipates
        more energy than one that models it well: the part of its memory that is
        not predictive of the future -- "nostalgic" information -- has to be paid
        for thermodynamically. That is this project's premise, already proven in
        the literature, and it is the better-motivated thing for integration to
        buy than the error correction that Phase 4 tried and measured as
        insufficient.

        The model is the memory graph itself, used the way it is measured. For
        each item, the strongest association it has to something observed in the
        previous moment gives a conditional probability; where no association is
        available, its marginal frequency does. Surprisal is the negative log
        likelihood of what actually happened, per item.

        An organism whose associations are sharp is surprised less. An organism
        whose memory is a heap of unrelated fragments is surprised constantly --
        and, once `phi.prediction_weight` is non-zero, pays for it in entropy.
        """
        episodes = len(self.history)
        if episodes < 2:
            return 0.0

        vocabulary = sorted({key for episode in self.history for key in episode}
                            | set(actual))
        if not vocabulary:
            return 0.0

        counts: dict[str, int] = dict.fromkeys(vocabulary, 0)
        for episode in self.history:
            for key in episode:
                counts[key] += 1

        total = 0.0
        for key in vocabulary:
            probability = self._predict_one(key, counts, previous, episodes)
            observed = key in actual
            likelihood = probability if observed else 1.0 - probability
            total -= math.log2(max(likelihood, 1e-6))
        return total / len(vocabulary)

    def _predict_one(
        self, key: str, counts: dict[str, int],
        previous: frozenset[str], episodes: int,
    ) -> float:
        """P(key active now), conditioned on the best evidence available.

        Laplace-smoothed throughout: an unsmoothed conditional built from three
        observations will happily assert probability 1.0 and then be infinitely
        surprised, which would make the charge a lottery rather than a measure.
        """
        marginal = (counts[key] + 1.0) / (episodes + 2.0)
        if not previous:
            return marginal

        # The strongest association this item has to something just observed.
        best_strength, best_probability = 0.0, marginal
        for cue in previous:
            if cue == key:
                continue
            together = follows = 0
            for earlier, later in zip(self.history, list(self.history)[1:], strict=False):
                if cue in earlier:
                    follows += 1
                    if key in later:
                        together += 1
            if follows < 2:
                continue
            strength = follows / episodes
            if strength > best_strength:
                best_strength = strength
                best_probability = (together + 1.0) / (follows + 2.0)
        return best_probability


def _mutual_information(p_a: float, p_b: float, p_ab: float) -> float:
    """I(A;B) in bits for two binary variables, from their 2x2 table."""
    table = (
        (p_ab, p_a, p_b),
        (p_a - p_ab, p_a, 1.0 - p_b),
        (p_b - p_ab, 1.0 - p_a, p_b),
        (1.0 - p_a - p_b + p_ab, 1.0 - p_a, 1.0 - p_b),
    )
    total = 0.0
    for joint, marginal_a, marginal_b in table:
        # Numerical debris: probabilities can land a hair below zero.
        if joint <= 1e-12 or marginal_a <= 0.0 or marginal_b <= 0.0:
            continue
        total += joint * math.log2(joint / (marginal_a * marginal_b))
    return max(0.0, total)


def _spectral_bipartition(association: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the graph along its Fiedler vector.

    The stand-in for searching every bipartition. The Fiedler vector of the graph
    Laplacian is the standard relaxation of the minimum-cut problem: it finds the
    split that severs the least association, which is what a minimum-information
    partition is looking for.
    """
    degree = association.sum(axis=1)
    laplacian = np.diag(degree) - association
    try:
        values, vectors = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        half = len(association) // 2
        mask = np.zeros(len(association), dtype=bool)
        mask[:half] = True
        return mask, ~mask

    order = np.argsort(values)
    fiedler = vectors[:, order[1]] if len(order) > 1 else vectors[:, order[0]]
    left = fiedler > 0.0
    if not left.any() or left.all():
        # A degenerate vector means the graph is effectively one blob; split it
        # in half rather than returning an empty side.
        left = np.zeros(len(fiedler), dtype=bool)
        left[: len(fiedler) // 2] = True
    return left, ~left
