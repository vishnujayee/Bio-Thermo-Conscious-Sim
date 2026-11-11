# Φ: what is computed, what it is not, and what it measured

Required reading before quoting any number from this project that has a Φ in it.

---

## 1. The claim this document exists to prevent

**Nothing in this project is conscious, and Φ here is not evidence about
consciousness.** What is computed is an *integration proxy*: a number that is
high when a system's parts carry information about each other and low when they
do not. That is one property associated with consciousness in one theory. It is
not consciousness, and a high score is not a mind.

The project uses the name Φ because the quantity is shaped after Integrated
Information Theory's Φ and would be dishonest under a different name. Everything
below is about the distance between the two.

---

## 2. What canonical Φ asks for, and what we do instead

Canonical Φ is intractable for anything non-trivial. It searches **every**
bipartition of a system and, for each, computes effective information over the
full joint distribution of states. Both halves are replaced here:

| Canonical | Here | What is lost |
|---|---|---|
| Search all bipartitions for the minimum-information one | **Spectral bipartition** — the Fiedler vector of the association graph | The true minimum cut may be missed; the Fiedler cut is a relaxation, not a solution |
| Effective information over the joint distribution | **Sum of pairwise mutual informations** across the cut | All higher-order structure. Three items that are informative only jointly count as zero |
| Interventional (causal) distributions | **Observational** frequencies over a window of experience | Correlation, not causation. Nothing here intervenes on the system to see what changes |

Each substitution is standard practice for making Φ-like quantities computable.
None is free, and together they mean **the number is a proxy for a proxy**.

---

## 3. What is actually computed

Every organism keeps a `MemoryGraph`: a keyed store it can write to with
`remember`, and a rolling window of **episodes** — one per turn — recording which
items were active together.

An item is active in a turn when the organism's situation or behaviour included
it. The vocabulary is deliberately small and identical for every brain:

- state: `hungry`/`sated`, `dirty`/`clean`, `hot`/`cool`, `on_node`/`bare`
- behaviour: `act:ingest`, `act:move`, `act:dissipate`, `act:rest`, `act:replicate`
- outcome: `acted` / `refused`
- any explicit memory the organism wrote or read, as `mem:<key>`

Then, over the window:

1. Drop items that were active in every episode or in none — they carry zero
   entropy and can be informative about nothing.
2. For each remaining pair, compute the mutual information `I(i;j)` in bits from
   their 2×2 contingency table.
3. Split that association graph along its Fiedler vector.
4. **Φ = (sum of `I(i;j)` across the cut) / (size of the smaller side)** — bits of
   integration per element of the part that could most easily have been severed.

### Why the vocabulary is shared between brains

A graph built only from explicit `remember` calls would give the mock brain no
integration at all, because it never calls them. Any comparison between a rule
and a language model would then be about which one used the memory tools, not
about which one thought. The automatic vocabulary is what makes the control
condition a control.

It is also the largest *design* choice in the measure, as opposed to the largest
approximation: a different discretisation of experience would give different
numbers. The numbers are comparable within this project and nowhere else.

---

## 4. Two things the measure gets right

These are the sanity checks worth knowing, because a measure that failed them
would be measuring the wrong thing entirely.

**Monotony scores zero.** Items always active together are perfectly redundant:
knowing one tells you nothing new. An organism that does the same thing in every
situation has a memory that is *uniform*, not integrated. A naive
"how strongly connected is the graph" measure gets this backwards and calls
perfect repetition maximal integration.

**Independent halves score zero.** Two subsystems varying without reference to
each other are fragments, however large. Measured: 0.005 on synthetic data.
Perfectly coupled halves of the same size score 4.5; 80% coupled scores 1.3.

---

## 5. What it measured — and what that did to the project's central claim

The claim under test: *integration is worth its thermodynamic price.*

Φ costs superlinearly under law L3 — `ΔS ∝ (1+Φ)^k` with `k = 1.4` shipped. Since
v0.9 it also **buys** something: error correction. An organism whose parts carry
information about each other can reconstruct a damaged part from the rest, so
integration divides the corruption that disorder inflicts on perception and
memory (`phi.resilience`).

### Result 1 — the protection does not pay for itself

Sweeping `phi.resilience` over 5 seeds, 600 ticks, 16 founders:

| resilience | mean population | mean lifespan | births | extinctions |
|---|---|---|---|---|
| 0.0 | 12.5 | 81.9 | 76 | 0/5 |
| 0.25 | 13.7 | 84.4 | 83 | 0/5 |
| 0.5 | 11.7 | 84.2 | 68 | 0/5 |
| 1.0 | 9.8 | 69.6 | 68 | 0/5 |
| 2.0 | 8.6 | 66.9 | 46 | **2/5** |

More protection makes things *worse*, and the mechanism is legible: shielding
integration means organisms keep more of it, and keeping it is what costs. The
loop runs the wrong way round.

### Result 2 — selection runs against integration

The selection differential — Φ of organisms that reproduced against those that
never did, the same instrument used in Phase 5 where four control traits came out
at coin flips:

| resilience | bred | never bred | difference | seeds favouring higher |
|---|---|---|---|---|
| 0.0 | 0.617 | 0.729 | −0.112 | 0/5 |
| 0.5 | 0.609 | 0.819 | −0.210 | 0/5 |
| 2.0 | 0.626 | 0.807 | −0.181 | 0/5 |

**Zero seeds out of five favour higher Φ, at any resilience.**

### Result 3 — it is the price, not a confound

The obvious objection: perhaps Φ is not being selected against at all, and this
is an artefact — a well-adapted organism in a simple world repeats itself, and
repetition is exactly what this measure scores as zero. "Boring is adaptive"
would produce the same table without integration costing anything.

That is testable. Set `phi_entropy_exponent` to 0, making Φ free:

| Φ's price | resilience | bred | never bred | difference | seeds |
|---|---|---|---|---|---|
| normal (k=1.4) | 0.0 | 0.617 | 0.729 | −0.112 | 0/5 |
| normal (k=1.4) | 0.5 | 0.609 | 0.819 | −0.210 | 0/5 |
| **free (k=0)** | 0.0 | 0.585 | 0.551 | **+0.034** | **4/5** |
| **free (k=0)** | 0.5 | 0.594 | 0.577 | +0.017 | 2/5 |

**The differential flips positive when integration is free.** So the driver is
the cost, not the confound. Integration's benefit is real but smaller than its
bill.

### Result 4 — where it breaks even

Sweeping the exponent directly:

| `k` in `(1+Φ)^k` | difference | seeds favouring higher |
|---|---|---|
| 0.0 | +0.017 | 2/5 |
| 0.3 | +0.013 | 3/5 |
| 0.6 | −0.005 | 4/5 |
| 0.9 | −0.006 | 2/5 |
| 1.2 | −0.103 | 1/5 |
| **1.4 (shipped)** | **−0.210** | **0/5** |

Roughly neutral while the price is cheap, sharply negative past `k ≈ 1`. There is
no regime in which integration is clearly *favoured* — the best it manages is
break-even when it is nearly free.

### Result 5 — a changing world does not rescue it (OPEN-19)

The strongest objection to results 1–4 was that error correction protects a model
of an environment that was never going to surprise anyone. A good map is worth
little where the right answer never changes. Phase 6's scenario controls exist to
test exactly that, and `seasons` is built for it: the climate swings, food stops
respawning, and **the nodes physically move**, so the map an organism holds goes
stale on a schedule.

| scenario | mean Φ | mean population | bred | never bred | difference | seeds favouring higher |
|---|---|---|---|---|---|---|
| none | 0.641 | 13.7 | 0.595 | 0.750 | −0.155 | 0/5 |
| bottleneck | 0.662 | 11.5 | 0.624 | 0.740 | −0.115 | 1/5 |
| drought | 0.665 | 12.9 | 0.658 | 0.678 | −0.021 | 1/5 |
| heat_wave | 0.671 | 9.8 | 0.628 | 0.768 | −0.140 | 1/5 |
| gradient_collapse | 0.649 | 11.7 | 0.609 | 0.733 | −0.123 | 1/5 |
| **seasons** | 0.669 | 9.3 | 0.633 | 0.740 | **−0.107** | **1/5** |

**Not one scenario produces a positive differential.** Organisms do become
slightly more integrated under pressure — mean Φ rises from 0.641 to ~0.67 — and
it still does not help them reproduce. The "world too easy" explanation is
refuted.

### Result 6 — prediction, and the trap in it (OPEN-18)

The remaining candidate was the best-motivated one. Still, Sivak, Bell & Crooks
(2012) prove that a system which models its environment badly *necessarily*
dissipates more energy, which would make integration pay in joules rather than in
protection. Implemented as `phi.prediction_weight`: each turn an organism's own
memory predicts the next moment, and it is charged entropy for the surprise.

The selection differential, at last, turns positive:

| condition | mean population | bred | never bred | difference | seeds |
|---|---|---|---|---|---|
| no prediction cost | 13.7 | 0.595 | 0.750 | −0.155 | 0/5 |
| prediction weight 0.5 | 7.8 | 0.665 | 0.673 | −0.008 | 2/5 |
| prediction weight 1.0 | 6.0 | 0.724 | 0.808 | −0.084 | 2/5 |
| **prediction weight 2.0** | 7.1 | 0.867 | 0.744 | **+0.123** | **5/5** |
| seasons, weight 2.0 | 8.9 | 0.881 | 0.743 | **+0.138** | **5/5** |

**Five seeds out of five, in both worlds.** Stopping here would have produced a
headline: *integration pays once it pays in joules.*

**It would have been wrong.** The claim has a mechanism, and the mechanism is
checkable: if integration buys prediction, an organism with higher Φ must be
surprised *less*. Measured within runs, across every condition:

| condition | correlation between Φ and surprise | organisms | seeds with r < 0 |
|---|---|---|---|
| weight 0.5, none | **+0.356** | 325 | 0/5 |
| weight 1.0, none | +0.293 | 224 | 0/5 |
| weight 1.0, seasons | +0.314 | 183 | 0/5 |
| weight 2.0, none | **+0.467** | 154 | 0/5 |
| weight 2.0, seasons | +0.464 | 152 | 0/5 |

**Higher Φ organisms are consistently *more* surprised, never less.** So the
positive differential is a confound, not a discovery: behavioural variety drives
Φ and surprise upward together, and the winners under a heavy surprise charge are
not better predictors — they pay more surprise cost and reproduce anyway.

The cause is structural and worth stating plainly: **the Φ proxy and the
surprisal model are computed from the same pairwise statistics over the same
window.** They cannot independently measure "integration" and "predictive
quality", because they are two readings of one set of numbers. A real test of
OPEN-18 needs a predictor that does not share the measure's basis — predicting
*external* world variables (where the food will be, what the temperature will do)
rather than the organism's own activation vocabulary. That is OPEN-21, and it is
the honest next step.

`phi.prediction_weight` therefore ships at **0.0**. A mechanism that has not been
validated does not get to be the default, however good its headline number looks.

---

## 6. What this means

**The project's central claim is not supported under this operationalisation.**
Integration does not pay for itself. Selection pushes Φ down.

That is a result, not a failure, and it is worth being precise about its scope.
It says: *given this proxy, this vocabulary of experience, this world, and error
correction as the benefit, the price L3 charges exceeds the return.* Any of those
four could be wrong, and three of them are things we chose:

1. ~~**The benefit may be the wrong one.**~~ **Tested (result 6).** Prediction
   produced a positive differential whose mechanism does not hold up: high-Φ
   organisms are more surprised, not less. Inconclusive rather than refuted — the
   test was compromised by the proxy and the predictor sharing a basis, which is
   OPEN-21.
2. ~~**The world may be too simple.**~~ **Refuted (result 5).** Six scenarios
   including moving food and a swinging climate; not one produces a positive
   differential.
3. **The price may simply be too high.** `k = 1.4` came from the project plan, not
   from measurement. Nothing justifies 1.4 over 0.9, and the differential is
   roughly neutral below `k ≈ 1`.
4. **The proxy may be too crude** — pairwise, observational, over a coarse
   vocabulary. Result 6 sharpened this from a caveat into a concrete defect: a
   measure that shares its statistics with everything it is compared against
   cannot be tested against them.

**What would change the conclusion:** a positive selection differential on Φ,
consistent across seeds, in a world where the shipped price is unchanged. Until
then the honest statement is the negative one.

---

## 7. Practical notes

- `phi.enabled: false` removes the whole mechanism; Φ stays at `initial_phi`.
- `phi.resilience: 0.0` restores the pure-cost regime — the control condition.
- **Φ is recomputed on a schedule and with hysteresis** (`recompute_every`,
  `tier_margin`), and not because the eigen-decomposition is slow. Tiers swap
  models, prompt caches are model-scoped, and an organism oscillating across a
  boundary throws away the cached prefix its whole lineage shares, every time it
  wobbles. That is the answer to OPEN-8: dynamic, but rate-limited and hysteretic.
- **The tier thresholds were measured, not chosen** (0 / 0.7 / 1.4 / 2.5), from an
  observed distribution of median 0.38, p90 1.36, p95 1.93, max 3.61. While Φ was
  a static 1.0 the thresholds were round numbers and no organism ever reached the
  second one. A tier nothing reaches is a routing rule that never fires, which is
  worse than no rule at all because it looks like one. Re-measure them if the
  world's constants move.
- **Φ is inherited at replication; the memory that justified it is not.** A child
  starts at its parent's Φ with an empty graph and has to earn its own.
