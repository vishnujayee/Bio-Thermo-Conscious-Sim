# How to Think About This Project

`README.md` says what this is. `construction.md` tracks what is built.
**This file is about how to think while building it**, and how to learn what you
need to keep going.

Read sections 0 and 1 before writing any code. The rest can wait until you need
it.

---

## 0. The whole project in one page

**The idea.** Life is a thermodynamic process: it maintains order locally by
exporting disorder, and it can only do that while there is a gradient to feed
on. Erwin Schrödinger put it in 1944 as "life feeds on negative entropy". This
project takes that literally and builds a world where it is the *only* rule.

**The mechanism.** Each organism has energy `E` and internal entropy `S`. It is
alive while

```
F = E − T · S · ln2  >  0
```

Every irreversible act — moving, thinking, remembering — raises `S`. Erasing `n`
bits of `S` costs `n · k_B · T · ln2 / η` joules, which is Landauer's bound, and
`η < 1`, so forgetting is never free. Eating is capped by the Carnot limit on
the temperature gradient between the food and the organism's own cell. Nothing
is scored; there is no fitness function anywhere in the code.

**Why an LLM is in it at all.** The tokens a model consumes are a real, measured
quantity, and this project converts them straight into internal entropy. So a
model that reasons expensively *heats up*, and heat is what kills it. That is the
one bridge between the two halves of the project, and everything else follows
from it.

**Where it stands.** Organisms live, die, compete, reproduce with mutation, and
form lineages 19 generations deep. Selection is measured, not assumed. The one
grand claim the project set out to test — that integration ("consciousness") is
worth its thermodynamic price — has been tested and **not supported**.

**The five things worth knowing before you touch it:**

| | |
|---|---|
| The discipline | Never write a rule you could derive (§1) |
| The honesty | Know what is real and what is analogy (§2) |
| The instrument | The selection differential, and its limits (§3) |
| The result | Integration does not pay for itself (§4) |
| The habit | When a result finally goes your way, check hardest (§4) |

---

## 1. The one discipline that matters

**Never write a rule you could derive.**

Every time you are tempted to add "if X then Y", stop and ask whether it could
fall out of the physics instead. The difference is everything:

| Imposed rule | Derived behaviour |
|---|---|
| "Crowding reduces food by 20%" | Waste heat raises local temperature → Carnot efficiency falls → crowds starve themselves |
| "Confused agents make mistakes" | Entropy corrupts perception → the map is wrong → decisions follow the wrong map |
| "Thinking costs 5 energy" | Tokens are irreversible writes → Landauer says erasure costs energy → thought has a price |
| "Later generations are weaker" | A child divides its parent's disorder → breeding while filthy hands the cost to your offspring |

The right-hand column is the project. When the Carnot bound was implemented
correctly, the efficiency decay from 64% to 46% appeared without anyone designing
it. **That is the standard to hold every new feature to.**

### Three questions for any new feature

1. **Does it emerge, or did I impose it?** If imposed, can it be moved one level
   down into the physics?
2. **Can it be measured?** If you cannot log a number for it, you cannot tell
   whether it worked.
3. **Does it create a trade-off?** Anything strictly better is boring. Life is
   interesting *because* everything costs something.

### A sense is not a rule

The distinction that keeps coming up. When organisms could not perceive whether
their own actions had succeeded, they repeated the same failing act forever. The
fix was to add `last_action` to the percept — and that is a **sense**, not a
rule: it says what happened, never what to do, and a brain is free to ignore it.
The mock brain does ignore it, and its runs came out bit-identical, which is how
we know the sense biased nothing.

Adding perception is almost always legitimate. Adding *behaviour* almost never is.

### The falsifiability habit

For every claim the project makes, write down what result would prove it *wrong*
before you run it. "Frugal genomes outcompete wasteful ones" is a real claim
because a run could show the opposite. "The simulation is lifelike" is not a
claim at all.

This habit is the reason §4 exists. The central claim was written down in a form
that could lose, and then it lost.

---

## 2. What is actually real here, and what is analogy

Being honest about this makes the project stronger, not weaker.

**Real:**
- The thermodynamics. Landauer's bound, the Carnot limit, the second law —
  implemented properly and asserted every tick. At perfect efficiency the books
  balance to exactly zero, which means the bounds are falling out of the
  arithmetic rather than being asserted on top of it.
- The selection pressure. Measured: a genome that never clears its disorder
  reliably loses.
- Emergence. The self-heating efficiency decay was not programmed.

**True since v0.8, and worth stating precisely:**
- **Evolution happens.** Organisms reproduce, children differ from parents, and
  the difference is associated with who reproduces. Runs reach 19–48 generations.
- **Selection is measured, not assumed.** Organisms that bred had higher entropy
  tolerance than those that did not, in 5 seeds out of 5 — while four other
  traits drifting under the identical mutation operator in the identical runs
  came out at coin flips. That contrast is what makes it a signal.

**Analogy:**
- An "organism" is a handful of numbers and a text prompt. It does not build or
  repair its own structure. Nothing here is autopoietic in the strong sense.
- Φ is an integration *proxy* — and a proxy for a proxy, since both the partition
  search and the effective-information calculation are replaced by tractable
  approximations. Nothing here is conscious, and a high Φ is not evidence about
  consciousness. `docs/phi.md` states the distance.
- The units are internally consistent, not SI. This is not a simulation of any
  real physical system.

**Not claimed:**
- **That any lineage is better adapted.** A selection differential says which
  organisms bred. It does not show a population mean that moves and *stays*
  moved, and late-run populations are small enough that drift is large. "There is
  a selection differential on one trait" is the claim; "the organisms are
  evolving to be better" is not.

---

## 3. How to measure anything here

Three instruments, in the order you should reach for them.

**1. The selection differential.** Compare a trait between organisms that
reproduced and those that never did.

```python
from btcs.analysis import load
run = load("runs/default_seed42/events.jsonl")
run.selection_differential("dissipate_at")   # (bred, never bred, n, n)
```

Its power comes from the **control traits**. Five heritable numbers drift under
the same mutation operator in the same runs; if one shows a consistent signal and
four come out at coin flips, the one is not an artefact of drift. Never report a
differential without saying what the other traits did.

Its limit: it says which organisms bred. It does not establish a *response* to
selection — a population mean that moves and stays moved.

**2. The knob-out control.** Every setting is explained in plain language in
`docs/config.md`, including which ones are safe to play with and which will break
the physics on purpose. Every mechanism here can be switched off:
`phi.resilience: 0.0`, `phi.enabled: false`, `child_entropy_fraction: 0.0`,
`cache_read_entropy_weight: 1.0`. If a result does not change when you disable
the thing you think caused it, you have not found what you think you found. This
is how the Φ result was pinned down: making Φ free flipped the sign, which proved
the price was the driver.

**3. Watching it.** Two of the most important findings — the boom-bust cycle, and
a language model trying to eat where there was no food — would have taken five
seconds to *see* and might never have been computed. So look before you compute.

```bash
python -m btcs.report runs/<name>_seed42/events.jsonl              # in the terminal
python -m btcs.viz    runs/<name>_seed42/events.jsonl -o run.html  # in a browser
```

`report` prints sparkline curves, the action mix, death causes, **refused acts**,
the selection differential and a trait-by-generation chart. `viz` writes one
self-contained HTML file — no server, no dependencies — with the world animated
tick by tick, a lineage tree of every organism that ever lived, and the
evolutionary tables. Open it and press play.

Both read the JSONL log and nothing else, so a run from six months ago is as
inspectable as one from this morning.

### What to look for when you watch a run

| What you see | What it means |
|---|---|
| Population climbs fast, then crashes | The overshoot. Waste heat from a crowd lowers everyone's Carnot efficiency — a crowd degrades its own metabolism by being a crowd. Nobody designed this. |
| Squares turning orange, then red | The degradation ladder. Disorder is corrupting perception before it kills. Red squares are usually about to be erased. |
| Squares sitting still on empty ground | Almost always a brain that cannot connect "not on food" to "move". This is exactly how the 3B model failed. |
| Blue dots vanishing in a region | Nodes exhausted. Watch whether organisms leave *before* or *after* the food runs out — leaving after is the signature of a bad map. |
| Long thick green lines in the lineage tree | Lineages that bred repeatedly. If they cluster in one generation band, something changed in the world at that moment. |
| The `refused acts` panel dominated by one reason | The most useful single diagnostic in the project. An organism failing the same way repeatedly is a design gap, not bad luck. |

If something in the picture surprises you, **compute it before you believe it** —
and if it does not surprise you, compute it anyway.

### Whatever you measure, do it across seeds

One seed is an anecdote. Everything in `docs/phi.md` is five seeds, and the
column that matters is not the mean difference — it is **how many seeds agree**.
A +0.02 mean where 2 of 5 seeds agree is noise; a −0.11 mean where 0 of 5 agree
is a result.

---

## 4. The results, and the lesson inside them

### The negative result at the centre of the project

**Integration does not pay for itself.** Φ costs superlinearly and buys error
correction, and across five seeds organisms that reproduced had *lower* Φ than
those that never did — 0 seeds out of 5 favouring higher integration, at every
level of protection tested.

The obvious objection is a good one: maybe a well-adapted organism in a simple
world just repeats itself, and repetition is exactly what this measure scores as
zero. So test it — make Φ free by setting its cost exponent to zero. **The
differential flips positive, 4 seeds out of 5.** The driver is the price, not the
confound.

Then two rescue attempts, both recorded in full in `docs/phi.md`:

- **A world worth having a map of.** Six scenarios including one that physically
  moves the food. Not one produces a positive differential. *Refuted.*
- **Prediction instead of protection.** See below.

**This is what a falsifiable claim looks like when it is actually tested**, and it
is worth more than a confirmation would have been.

### The near miss, which is the best lesson here

Making Φ pay in joules — charging organisms for being surprised by their own
world — finally produced a **positive** selection differential on integration:
+0.123, five seeds out of five, and +0.138 in the seasonal world. Exactly the
result the project was built to find. Written up there, the headline reads
*integration pays once it pays in joules.*

It would have been wrong.

The claim had a mechanism, and mechanisms are checkable. If integration buys
prediction, then an organism with higher Φ must be **surprised less**. Measured:
r = +0.29 to +0.47, and zero seeds out of five negative. High-Φ organisms are
consistently *more* surprised. The positive differential came from behavioural
variety driving both quantities up together.

The structural cause is worth internalising: **the Φ measure and the prediction
model were computed from the same statistics.** They could not independently
measure "integration" and "predictive quality", because they were two readings of
one set of numbers.

Two habits fall out of this, and they are the most valuable things in this file:

1. **When a result finally goes your way, that is when to check hardest.** The
   confirming result gets less scrutiny than the disconfirming one unless you
   deliberately reverse that.
2. **Always name the mechanism, then test the mechanism separately from the
   outcome.** "X correlates with success" is not "X causes success by doing Y".
   Measuring Y is usually cheap, and it is where the truth is.

The feature ships **off**. A mechanism that has not been validated does not get to
be a default, however good its headline number looks.

### The most useful debugging habit

**When a population dies, look at what the world still holds before you touch a
constant.** Phase 5's worst bug looked exactly like scarcity: lineages weakening,
generations dying younger, extinction at tick 437. The world still had 8,346
units of food in it. Dying of thirst in a lake is the signature of an accounting
error, never of a hostile environment — and the fix was in the books, not the
physics.

### Two more traps, paid for in full

**Changing how something is measured can silently change what it costs.** Making
Φ dynamic dropped it from a static 1.0 to a measured ~0.5, which made every
thought a third cheaper in disorder — and the project's whole premise stopped
holding in 2 of 3 seeds. If you change a measurement, **re-check the selection
signal immediately**; it moves silently.

**Never draw from a seeded stream while iterating a `set`.** Hash order is
randomised per process, so reproducibility dies intermittently and invisibly. Two
runs of the same test suite disagreed about which seeds passed.

---

## 5. What to build next

In order, and each with a reason rather than a wish.

### 1. A predictor that does not share the measure's basis  *(OPEN-21)*

The single most valuable open item, because it is what the near miss in §4 left
undone. Φ and the surprisal model are computed from the same pairwise statistics
over the same window, so they cannot independently measure integration and
predictive quality. Build a predictor over **external** world variables — where
the food will be, what the temperature will do next tick — and the test becomes
real. Everything needed is already logged.

### 2. Interaction between organisms

Right now organisms meet only by accident, through shared heat and contested
food. And the ancestor genome says *"Prioritise resource sharing"* while **no
sharing tool exists** — every organism is born instructed to cooperate in a world
with no mechanism for it. Either build the mechanism or change the genome; the
current state quietly wastes prompt tokens on an unfollowable instruction.

- **Sharing and altruism.** Does cooperation survive when helping costs you
  energy? Lineages exist now, so kin selection is directly reachable.
- **Costly signalling.** A message costs tokens, and tokens are disorder — so
  honest communication is *literally* expensive here. That is the actual biology
  of signalling theory, for free.

### 3. Maxwell's demon

The most on-theme idea available. The paradox was *resolved* by Landauer: the
demon must eventually erase its measurements, and that erasure is what pays the
entropy bill. This project already implements both halves — measurement through
the sensing tools, erasure through `dissipate_entropy`. Making measurement itself
consume information would put the resolution of a famous physics paradox inside
the simulation.

### 4. A response to selection  *(OPEN-16)*

The project measures selection *differentials* and is careful not to claim more.
Larger populations, more seeds, and a drift control would let it claim a
population mean that moves and stays moved — which is a materially stronger
statement than anything currently in `docs/phi.md`.

### Smaller, cheap, and still interesting

- **Sexual reproduction** — two parents, genome crossover instead of
  single-parent mutation. Cheap now that Phase 5 exists, and it changes
  evolutionary dynamics substantially.
- **Speciation and niches**; **horizontal gene transfer** (genome sharing between
  unrelated organisms).
- **The Baldwin effect** — within-lifetime learning versus across-generation
  inheritance. The `last_action` sense makes the first half possible.
- **A fixed-model control run** (T7.7) — Φ-tiered routing confounds a Φ effect
  with a model-capability effect, and this is what separates them.

---

## 6. Study resources

### Start here — the short list

- **Erwin Schrödinger, *What Is Life?* (1944).** Short, and the origin of the whole idea: life feeds
  on "negative entropy". Everything in this project descends from that.
- **Nick Lane, *The Vital Question*.** Energy as the thing that made life possible. The best modern
  book on why biology is fundamentally about gradients.
- **Peter Hoffmann, *Life's Ratchet*.** How molecular machines extract order from thermal noise.
  Directly relevant to why forgetting costs energy.

### Going deeper

- Ilya Prigogine, *Order Out of Chaos* — dissipative structures; the theory this project is an
  instance of.
- Stuart Kauffman, *At Home in the Universe* — self-organisation and autocatalysis.
- Melanie Mitchell, *Complexity: A Guided Tour* — the friendliest entry to complex systems.
- Christoph Adami, *Introduction to Artificial Life* — an actual textbook for this field.

### Papers worth the effort

- **Landauer (1961), "Irreversibility and Heat Generation in the Computing Process"** — law L1 of this
  project, in the original.
- **Bennett (1982), "The Thermodynamics of Computation — a Review"** — the Maxwell's demon resolution.
- **Still, Sivak, Bell & Crooks (2012), "Thermodynamics of Prediction"** — the most directly relevant
  paper to this project's premise: a system that models its environment badly *necessarily* dissipates
  more energy. Prediction failure has a thermodynamic cost. Read this one.
- **England (2013), "Statistical physics of self-replication"** — why replication is thermodynamically
  favoured.
- Friston (2010), "The free-energy principle: a unified brain theory?" — hard going, but the source
  for active inference.
- Tononi's IIT papers, for what Φ is actually supposed to mean.

### Systems to study (all of these exist and work)

- **Avida** — the serious research platform for digital evolution. Self-replicating programs, real
  published biology. The closest relative to what you are building.
- **Tierra** (Tom Ray, 1991) — the ancestor of the whole field; evolved parasites nobody designed.
- **Karl Sims, "Evolved Virtual Creatures" (1994)** — watch the video; it is still astonishing.
- **Lenia** — continuous cellular automata with startlingly lifelike results.
- Sakana AI, "Automating the Search for Artificial Life with Foundation Models" (2024) — the modern
  LLM-flavoured version of this question.

### Free courses

- **Santa Fe Institute's Complexity Explorer** (complexityexplorer.org) — free, excellent, and exactly
  on topic. Start with *Introduction to Complexity*.

### Community

- The **ALIFE conference** (International Conference on Artificial Life) — the field's home. Papers
  are largely open access and are a good sense of what counts as a result here.

---

## 7. Running it without spending money

Three brains, one interface. Nothing about the world changes between them.

| Brain | Cost | Speed | Use it for |
|---|---|---|---|
| `mock` | free | instant | Tuning physics, all tests, the permanent scientific control |
| `local` | free | ~4 s/turn | Real models as minds, on your own hardware |
| `llm` | ~$0.001–0.006/turn | fast | Frontier-model behaviour, when it matters |

```bash
python -m btcs.cli --ticks 400 --population 20              # free, instant
python -m btcs.cli --brain local --ticks 20 --population 3  # free, real model
python -m btcs.cli --estimate --ticks 200 --population 20   # price a paid run first
```

**Local runs use the Claude price table as a virtual currency.** Nothing is spent, but the budget
guard, cost reports and Φ economics behave exactly as they would on the API — so an experiment
developed locally transfers without changing a line.

### Two things local runs taught us that the API never would have

**Compute time is the more honest metabolic cost.** With an API, dollars are a *stand-in* for
thermodynamic cost. Locally, the cost is your own hardware actually working, and
`compute_seconds_entropy_weight` turns those seconds into internal entropy. Turn it up and a slow
thought genuinely cooks the organism that had it. The metaphor stops being a metaphor.

**The world was calibrated for a cheap thinker.** The mock brain "thinks" in 100 tokens. A real model
reads about 1,400 — mostly the laws and tool schemas it re-reads verbatim every single tick. Charged
at full rate, that killed every organism within three ticks. The fix was not to fudge a constant but
to notice something true: **the entropy discount was never about billing, it is about information
stability.** Re-reading an unchanged block is not new cognitive work, whether or not a provider
metered it. Once the stable prefix was counted as such, organisms survived.

That is the discipline from section 1 applied to our own accounting, and it is the kind of correction
worth looking for everywhere. It also has a sting in the tail: the *first* version of that fix
estimated the stable prefix and subtracted it, and the estimate came out larger than the whole prompt
— so the percept got swept into the discount and organisms were reading the world for a tenth of what
it should cost. **Measure the volatile side, so the estimation error lands on the prefix instead.**
Discounts are dangerous in exactly this way: they are invisible when they are too generous.

### What the first real-model run actually showed

Three organisms, twenty ticks, `qwen2.5:3b`, against the mock brain on the same seed:

| | mock | `qwen2.5:3b` |
|---|---|---|
| Alive at tick 20 | 3 / 3 | **extinct at 9** |
| Energy, start → end | 450 → 1,162 J | 450 → 0 |
| Successful meals | 29 | **0** |
| Actions refused | 0 | **19 of 23** — "no node here" |

They did not starve. They **cooked**, thinking hard about a meal that was not there, and every death
was `entropy_overwhelmed`. A hand-written rule beat a language model decisively.

Sit with that rather than fixing it away. It is the first thing this project has measured that could
not have been predicted from the code, and there are three competing explanations — the model is too
small, the prompt does not make `on_node` legible, or (the interesting one) **the organism has no way
to perceive the consequences of its own actions**. It is refused, told nothing, and repeats the
mistake next tick. Every individual is as blind at death as it was at birth.

That third explanation is a gap in the *design*, not the model, and it matters before reproduction
exists: if organisms cannot learn within a lifetime, selection has only genome prose to work on. The
fix is a sense, not a rule — which is the section 1 test, passed.
