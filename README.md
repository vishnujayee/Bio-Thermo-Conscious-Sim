# Bio-Thermo-Conscious-Sim

**A world where digital organisms have to obey thermodynamics, and where thinking
is something you pay for in heat.**

There are no points, no score, no fitness function. Each organism holds an energy
budget `E` and an internal entropy `S`, and lives or dies by one quantity:

```
F = E − T · S · ln2          F ≤ 0  →  the organism is erased
```

It harvests free energy from decaying gradients. Everything it does — moving,
thinking, remembering — writes irreversibly and so *generates disorder*, and
clearing that disorder costs energy at the Landauer bound. An organism that
reasons wastefully overheats, loses coherence, and dies. Nobody wrote that rule;
it falls out of the bookkeeping.

```bash
python -m btcs.cli --ticks 400 --population 20        # free, instant
python -m btcs.report runs/default_seed42/events.jsonl   # read the run
python -m btcs.viz   runs/default_seed42/events.jsonl    # watch the run
```

---

## Explaining it to someone in sixty seconds

> It's a little world where creatures have to obey real physics. They need to
> find energy, and *thinking costs them energy* — literally, the way erasing a
> bit of information costs energy in real computers. Think too much or too
> wastefully and you cook yourself to death. Nobody scores them; the ones that
> survive are just the ones whose behaviour happened to keep them solvent.
>
> They reproduce, their offspring inherit a slightly mutated strategy, and you
> can measure which traits the survivors actually had. So it's evolution you can
> watch, in a world where the pressure is thermodynamics rather than a rule
> someone wrote.
>
> The creatures can be driven by a simple hand-written rule, or by a real AI
> model — the world can't tell the difference. When we tried a small AI model,
> it lost to the hand-written rule, badly. That's the kind of thing this is for.

If they want one detail that makes it concrete: **crowds cook themselves.** Waste
heat raises the local temperature, which lowers the Carnot efficiency of every
meal nearby. So a crowd degrades its own metabolism by being a crowd. That
behaviour was never programmed — it appeared the day the Carnot bound was
implemented correctly.

---

## What is real, and what is analogy

Being straight about this makes the project stronger, not weaker.

**Real.** The thermodynamics: Landauer's bound on erasure, the Carnot limit on
extraction, and the second law asserted every single tick. At perfect efficiency
the books balance to exactly zero, which means the bounds fall out of the
arithmetic rather than being asserted on top of it. Selection is real and
measured. Reproduction, heredity and mutation are real.

**Analogy.** An "organism" is a handful of numbers and a text prompt. It does not
build or repair its own structure. Φ is an *integration proxy* — see
[`docs/phi.md`](docs/phi.md) — and nothing here is conscious. The units are
internally consistent, not SI.

**Not claimed.** That any lineage has become better adapted. We measure a
selection differential — which organisms bred — not a population mean that moves
and stays moved.

---

## Running it

Three brains, one interface. The physics never learns which is plugged in.

| brain | cost | speed | what it is for |
|---|---|---|---|
| `mock` | free | instant | tuning physics, the whole test suite, the **permanent control** |
| `local` | free | ~5 s/turn | a real model on your own machine, via [Ollama](https://ollama.com) |
| `llm` | ~$0.001–0.006/turn | fast | frontier-model behaviour, when the question needs it |

```bash
python -m btcs.cli --ticks 400 --population 20               # the free path
python -m btcs.cli --brain local --ticks 20 --population 3   # a real model, no API key
python -m btcs.cli --estimate --ticks 200 --population 20    # price a paid run first
python -m btcs.cli --brain llm --ticks 20 --population 3     # costs money
python -m btcs.cli --scenario seasons --ticks 600            # a world that changes
```

Local runs are billed against the Claude price table as a *virtual* currency, so
the budget guard, cost reports and Φ economics behave exactly as they would on
the API. An experiment developed for free transfers to the paid path unchanged.

### Seeing it

```bash
python -m btcs.report runs/<name>_seed42/events.jsonl              # terminal
python -m btcs.viz    runs/<name>_seed42/events.jsonl -o run.html  # browser
```

`report` prints curves, action mix, death causes, refused acts, and the selection
differential. `viz` writes **one self-contained HTML file** — no server, no
dependencies — with the world animated tick by tick, a lineage tree of every
organism that lived, and the evolutionary tables. Open it and press play.

---

## What it has actually found

Results, not features. Each was measured, and each could have come out the other
way.

**Crowds cook themselves.** Observed extraction efficiency 0.34–0.50 against a
theoretical ceiling of 0.64 — the gap *is* the organisms heating their own
neighbourhood. Emergent, not designed.

**Entropy kills, and it selects.** A genome that never clears its disorder is
reliably outlived by one that does. That is the project's premise, and it holds.

**Longevity and fecundity trade off.** Erasure costs energy; energy is what a
child is made of. The frugal genome lives ~45% longer and leaves half as many
descendants. The life-history trade-off real organisms face, out of the Landauer
bill rather than a rule.

**Selection favours postponing erasure.** Across five seeds, organisms that bred
had a higher entropy tolerance than those that never did — while four other
traits drifting under the identical mutation operator in the identical runs came
out at coin flips. That contrast is what makes it a signal.

**A 3B language model lost to a hand-written rule.** Same seed, same world: the
rule kept 3/3 alive and tripled their energy; the model went extinct at tick 9
having spent 19 of 23 turns trying to eat where there was no food. It did not
starve — it *cooked*, thinking expensively about a meal that was not there.

**Integration does not pay for itself.** The project's central claim, tested and
not supported. Φ costs superlinearly and buys error correction, and selection
pushes it *down* — 0 of 5 seeds favour higher Φ. Making a changing world did not
rescue it; making Φ free flipped the sign, which shows the driver is the price.
The full account, including a positive result that turned out to be a confound,
is in [`docs/phi.md`](docs/phi.md).

---

## The architecture, briefly

Three isolated layers. The middle one is a **physics firewall**: the cognitive
layer can never touch simulation state, only call tools.

```
cognitive layer   LangGraph cycle · Anthropic SDK / Ollama · prompt caching
      │           no simulation array ever crosses this line
      ▼
MCP tool surface  session-bound identity · turn discipline · budgets
      │           SimulationService is the ONLY path to the physics
      ▼
the world         vectorised NumPy grid · thermal field · entropy ledger
                  seeded RNG · append-only JSONL event log
```

Two properties are load-bearing:

- **An organism cannot cheat, structurally.** No tool accepts an `agent_id`;
  identity is an unguessable capability token resolved server-side. The internal
  tick loop calls the same service an MCP client does, so there is no private
  fast path that could drift from the tool surface.
- **The books are audited every tick.** `ΔS_total ≥ 0` *and* a stronger check:
  the entropy the ledger claims organisms gained must equal what they actually
  gained. Without the second, a code path could mutate entropy silently while
  the first check stayed comfortably true.

---

## Where everything is

| file | what it holds |
|---|---|
| [`GUIDE.md`](GUIDE.md) | **start here** — how to think about the project, what is real, the reading list |
| [`construction.md`](construction.md) | the living build document: every decision, task, and result |
| [`docs/config.md`](docs/config.md) | every setting in plain language: what it means, and what happens when you change it |
| [`docs/phi.md`](docs/phi.md) | what Φ is, what it is not, and the measurements that refuted the central claim |
| `src/btcs/world/` | the physics: grid, ledger, agents, memory graph |
| `src/btcs/sim/` | the service layer and scenario controls |
| `src/btcs/brains/` | mock, local (Ollama), and Anthropic brains behind one interface |
| `src/btcs/analysis.py` | reading a run back: metrics, lineage, replay |
| `configs/default.yaml` | every physics constant, and why it has the value it has |

454 tests, none of which make an API call.

```bash
pip install -e .
pytest
```

---

## What it is not

Not a claim to have built a conscious system. Not a physics engine — the units
are internally consistent, not SI. Not a general agent framework. It is a
single-purpose research sandbox for one question: **what happens to a mind that
has to pay for itself in heat?**
