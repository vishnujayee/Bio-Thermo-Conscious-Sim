# Bio-Thermo-Conscious-Sim — Construction Document

> Living build document. Every design discussion, decision, and task lands here.
> Status legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` dropped

**Created:** 2026-09-05
**Source:** `Bio-Thermo-Conscious-Sim-Project-Plan.pdf`
**Last updated:** 2026-09-07 — v0.11, **Phase 6 complete: the project can be read, watched, and explained**
**Companions:** `README.md` — what this is, how to run it, and how to explain it to someone in a minute
· `GUIDE.md` — how to *think* about it: the discipline, the instruments, the results, the reading list
· `docs/phi.md` — what Φ is, what it is not, and the measurements that refuted the project's central claim

---

## 0. Where We Are Right Now

**Phases 0–3 are complete. Phases 4–7 are not started.**

| Phase | State | Blocking issue |
|---|---|---|
| 0 Foundations | ✅ done | — |
| 1 Sandbox & physics | ✅ done | — |
| 2 MCP proxy | ✅ done | — |
| 3 Cognitive layer & token economy | ✅ done, incl. live run on a real model | — |
| 3.6 Perception of consequence | ✅ done | — |
| **5 Reproduction & selection** | ✅ **done — heredity, variation, measured selection** | — |
| 4 Φ / memory graph | ✅ **done — and the result is negative** | — |
| 5b Executable genomes | ⬜ opt-in, gated | needs sandbox before the flag can be enabled |
| 6a Scenario controls | ✅ **done — six presets, live parameter overrides** | — |
| 6b Analysis, replay, visualisation | ✅ **done — read it, watch it** | — |
| 7 Experiments & docs | 🟡 docs done; experiments open | — |

### What can now be claimed, and what cannot

**Can be claimed.** Organisms reproduce, children differ from parents, and the difference is
associated with reproductive success. All three ingredients of Darwinian evolution are present and
each is measured rather than asserted — see the v0.8 log entry. Runs reach 20–35 generations.

**Cannot be claimed.** That any lineage has become *better adapted*. A selection differential says
which organisms bred; it does not establish a sustained response to selection over many generations,
and late-run populations are small enough (1–7) that drift is large. The honest statement is: *there
is a measurable selection differential on one trait, consistent across five seeds, absent on the other
four traits in the same runs.*

**Also still true:** no organism has any way to communicate with, help, or harm another except through
shared heat and contested food.

### Pending, smallest first

| # | Item | Where |
|---|---|---|
| P-1 | **A predictor that does not share the measure's basis.** Φ and the surprisal model are computed from the same pairwise statistics over the same window, so they cannot independently measure integration and predictive quality. Predict *external* world variables instead — where the food will be, what the temperature will do. This is what a real test of OPEN-18 requires. | OPEN-21 |
| P-2 | A sharing or signalling tool — the ancestor genome still says *"Prioritise resource sharing"* in a world with no way to share | `GUIDE.md` §5 |
| P-3 | Replicate the selection result with more seeds, and test for a *response* to selection, not just a differential | OPEN-16 |
| P-3 | A sharing/communication tool — the ancestor genome says *"Prioritise resource sharing"* and no sharing mechanism exists | Phase 5+, `GUIDE.md` §3 |
| P-4 | Paid-API confirmation run (optional; the local path already closed the exit criteria) | T3.9c |
| P-5 | Offline analysis + replay from JSONL | Phase 6, T6.4–T6.5 |

---

## 1. What We Are Building

A software-only simulation sandbox in which digital organisms are driven by LLM agents. Survival is
not governed by game rules; it is governed by an energy and entropy budget. An organism that reasons
wastefully overheats, loses coherence, and is erased.

Three claims the project makes, and how each is cashed out:

| Claim | Mechanism |
|---|---|
| **Life is thermodynamic** | Agents hold energy `E` and internal entropy `S`. They must harvest free energy from decaying gradients and pay real energy to erase entropy (Landauer). `F = E − T·S ≤ 0` is death. |
| **Selection is real** | Wasteful reasoning burns tokens → tokens generate entropy → entropy corrupts memory → corrupted memory reasons worse. The loop closes without a fitness function being written by hand. |
| **Consciousness has a price** | Φ (an integration measure over the memory graph) gates which reasoning capabilities unlock, and simultaneously raises token cost and entropy production per tick. |

### 1.1 Non-Goals

- Not a claim to have built a conscious system. Φ here is an **operational proxy**, documented as such.
- Not a physics engine. Units are internal and self-consistent, not SI.
- Not a general agent framework. Single-purpose research sandbox.

---

## 2. Architecture

Three strictly isolated layers. The MCP boundary is a **physics firewall**: the cognitive layer can
never read or write simulation arrays directly, only call tools.

```
+---------------------------------------------------------+
| LAYER 3 - Cognitive & Metabolic                          |
| LangGraph cyclic graph: Sense -> Integrate -> Reason -> Act
| Anthropic SDK direct calls (cache_control + effort)      |
| Memory graph per agent - Phi-proxy computation           |
+---------------------------+-----------------------------+
                            | MCP (JSON-RPC) - only channel
+---------------------------+-----------------------------+
| LAYER 2 - MCP Server Proxy (FastMCP)                     |
| Session-bound identity - parameter guardrails            |
| Tools: sense_gradient, ingest, work, dissipate,          |
|        recall, remember, replicate                       |
+---------------------------+-----------------------------+
                            | in-process Python API
+---------------------------+-----------------------------+
| LAYER 1 - Ecosystem Simulation Sandbox                   |
| Vectorized NumPy 2D grid - free-energy nodes - thermal   |
| field - dS_tot ledger - seeded RNG - event log (JSONL)   |
+---------------------------------------------------------+
```

**Why the cognitive layer calls the Anthropic SDK directly rather than through `langchain-anthropic`:**
this project's entire economy runs on `cache_control` placement and `output_config.effort`. LangChain's
chat-model wrapper abstracts both away. LangGraph is kept for graph topology, checkpointing, and
pause/resume of long runs; the model call inside the Reason node is raw SDK.

---

## 3. Physics Specification

Sim units. Boltzmann constant `k_B = 1`. Entropy in bits, energy in arbitrary "joules" (J).

### 3.1 State

| Symbol | Meaning |
|---|---|
| `E` | Agent free-energy store (J) |
| `S` | Agent internal informational entropy (bits) |
| `T_local` | Ambient temperature at the agent's cell |
| `F = E − T_local · S · ln2` | Helmholtz free energy. `F ≤ 0` → informational erasure (death) |
| `Φ` | Integration proxy over the memory graph |

### 3.2 Laws

**L1 — Landauer cost of forgetting.** Erasing `n` bits of internal entropy costs
`ΔE = n · k_B · T_local · ln2 / η_erase`, with `η_erase < 1`. Forgetting is never free. This is what
makes `dissipate_entropy` a genuine metabolic act instead of a reset button.

**L2 — Carnot ceiling on ingestion.** A free-energy node at temperature `T_node` yields at most
`η ≤ 1 − T_cold/T_node` of its stored energy, where **`T_cold` is the temperature of the organism's
own cell** — not global ambient (see §3.4). Nodes cool toward ambient over time, so gradients — not
stockpiles — are the resource. A node in thermal equilibrium is worthless even if "full". Agents
survive by finding and exploiting gradients, and by not standing somewhere hot while they do it.

**L3 — Cognition produces entropy.** Every token processed writes irreversibly into the memory graph:
`ΔS = σ · tokens · g(Φ)`, where `g(Φ)` grows superlinearly. High-Φ agents think better and cook faster.

**L4 — Second law is an invariant, not a vibe.** Every tick the engine computes
`ΔS_total = ΔS_agents + ΔS_environment` and asserts `ΔS_total ≥ 0`. A violation is a bug and fails
the test suite. This is the single most important correctness property in the project.

**L5 — Work.** Structural work on the environment (moving, building, altering a cell) draws `E` and
deposits waste heat into `T_local`, warming the agent's own neighbourhood. Crowds cook themselves.

### 3.3 Death & Degradation Ladder

Not a cliff — a gradient, so selection has something to grip:

1. `S > S_warn` — sensory noise injected into `sense_gradient` returns
2. `S > S_confuse` — memory graph edges pruned at random; recall degrades
3. `S > S_dement` — random text injected into context window (the PDF's "systemic dementia"), and
   the agent's effort level is forcibly downgraded
4. `F ≤ 0` — informational erasure. Registry entry deleted, memory graph freed, lineage record sealed.

### 3.4 Implementation Notes (added in Phase 1)

**The Carnot cold reservoir is the organism's own cell, not global ambient.** This was found during
implementation and is not a detail. An engine can only reject waste heat into the reservoir it
actually touches; using ambient lets an organism standing in a hot cell extract more work than the
second law permits, and the entropy ledger goes negative within a few ticks. Getting this right also
produces the crowding dynamics for free: waste heat from work and cognition warms the neighbourhood,
which raises `T_cold`, which shrinks everyone's extraction efficiency. **A crowd degrades its own
metabolism by being a crowd** — observed efficiency in a live run is 0.34–0.50 against a theoretical
ceiling of 0.64.

**The second law is guaranteed by construction, not by assertion.** Each entropy-reducing process
covers its own deficit: dissipation removes `n` bits from an organism but files `n/η_erase` bits to
the environment (net-positive exactly because `η_erase < 1`); ingestion rejects waste heat into the
same reservoir whose temperature set the efficiency (net zero in the ideal case, positive otherwise);
erasure at death moves bits one-for-one. Tests confirm both ideal cases balance to zero — the Landauer
bound and the Carnot theorem, recovered from bookkeeping alone.

**A second, stronger invariant was added: ledger consistency.** `ΔS_total ≥ 0` proves nothing if a
code path mutates `agent.entropy` without filing an entry — the total stays comfortably positive while
the accounting becomes fiction. So each tick also asserts that the entropy the ledger *claims*
organisms gained equals what they *actually* gained. A test proves this check has teeth.

---

## 4. Consciousness Index (Φ)

**Honest framing: this is a Φ-proxy, not canonical IIT Φ.** Canonical Φ is computationally
intractable for anything non-trivial. What we compute is an integration measure with the same spirit.

- Agent memory is a **graph**: nodes = memory items, edges = co-activation / reference strength.
- Each tick, record which memory nodes were touched → build an empirical co-activation matrix.
- `Φ_proxy` = effective information across the **minimum information bipartition**, approximated by
  spectral bipartition (Fiedler vector) rather than exhaustive search, normalized by graph size.
- High Φ = a memory that acts as one integrated whole. Low Φ = disconnected fragments.

**Φ coupling (the price of mind):**

| Φ band | Model | Effort | Context budget | Entropy multiplier |
|---|---|---|---|---|
| Low | `claude-haiku-4-5` | n/a | small | ×1.0 |
| Mid | `claude-sonnet-5` | `low` | medium | ×1.8 |
| High | `claude-opus-5` | `medium` | large | ×3.2 |
| Peak | `claude-opus-5` | `high` | max | ×6.0 |

Consciousness costs more — in entropy inside the sim, and in dollars outside it. The metaphor and the
invoice agree.

---

## 5. Token Economy & Cost Control

Cost is the top project risk. 20 agents × 500 ticks × 4 graph nodes ≈ 40,000 model calls.

- **T0 — Mock brain first.** A deterministic rule-based policy implementing the identical interface as
  the LLM brain. The entire simulation must run, be tuned, and pass its physics tests at **zero API
  cost** before a single real call is made. This is non-negotiable and reorders the PDF's week plan.
- **Prompt cache layout** (prefix order is `tools` → `system` → `messages`; volatile content last):
  1. Frozen physics laws + world rules · `cache_control` breakpoint
  2. Genome strategy block · `cache_control` breakpoint
  3. Memory digest · `cache_control` breakpoint
  4. Volatile per-tick sense data — **uncached, always last**
- **Emergent efficiency:** clones share a genome, so a successful lineage shares a cache prefix.
  Replication raises the cache hit rate. Selection for metabolic efficiency falls out of the billing
  system for free.
- **Verification:** assert `usage.cache_read_input_tokens > 0` across repeated ticks. Zero means a
  silent invalidator (timestamp in system prompt, unsorted JSON, varying tool list).
- **Hard budget guard:** per-run USD ceiling read from config; accumulate real `usage`; halt the run
  and checkpoint when exceeded.
- **Batch API** for generational (non-interactive) runs at 50% cost.

### Current model reference (2026)

| Model | ID | Input $/Mtok | Output $/Mtok |
|---|---|---|---|
| Opus 5 | `claude-opus-5` | 5.00 | 25.00 |
| Sonnet 5 | `claude-sonnet-5` | 2.00 | 10.00 |
| Haiku 4.5 | `claude-haiku-4-5` | 1.00 | 5.00 |

(The PDF's "Claude 3.5 / GPT-4o" is superseded.)

### 5.1 Three brains, one interface (added v0.6)

Nothing about the world changes between them. The physics never learns what is thinking.

| Brain | Cost | Speed | What it is for |
|---|---|---|---|
| `mock` | free | instant | Tuning physics, the whole test suite, the **permanent scientific control** |
| `local` | free | ~4–5 s/turn | A real model, with real tool calls and real token counts, on your own hardware (Ollama) |
| `llm` | $0.0013–$0.0064 / turn | fast | Frontier-model behaviour, when the question actually needs it |

```bash
python -m btcs.cli --ticks 400 --population 20              # free, instant, reproducible
python -m btcs.cli --brain local --ticks 20 --population 3  # free, real model
python -m btcs.cli --estimate --ticks 200 --population 20   # price a paid run before making it
python -m btcs.cli --brain llm --ticks 20 --population 3    # costs money
```

**Local runs are billed at the Claude price table as a virtual currency** (`virtual_pricing: true`).
Nothing is spent, but the budget guard, the spend ledger, the cost report and the Φ economics behave
exactly as they would on the API. An experiment developed locally transfers to the paid path without
changing a line — which is the whole point, and the reason virtual pricing is on by default.

**Compute time is the more honest metabolic cost, and it is only available locally.** On the API,
dollars are a *stand-in* for thermodynamic cost. Locally the cost is your own hardware actually doing
work, and `compute_seconds_entropy_weight` converts those seconds into internal entropy. At the
default `0.0` accounting stays token-based; raise it and a slow thought genuinely cooks the organism
that had it. This is the closest the project comes to charging real joules.

---

## 6. Security & Integrity

| # | Risk | Mitigation |
|---|---|---|
| S1 | **Identity spoofing.** PDF tools take `agent_id` as a parameter — Agent_A can act as Agent_B. | Bind identity to the MCP session/token. `agent_id` is never accepted from the caller. |
| S2 | **Prompt injection via evolved text.** Genomes mutate; agent-authored text re-enters prompts. | All agent-authored text is untrusted data. Delimit, never interpolate into instruction position. Genome mutations pass a validator. |
| S3 | **Physics tampering.** | Arrays never cross the MCP boundary. Tools return derived scalars/strings only. |
| S4 | **Runaway spend.** | Budget guard + kill switch + per-agent per-tick token ceiling. |
| S5 | **Mutated code escaping host** *(only if genomes are executable — see OPEN-1)* | Docker or WASM sandbox, no network, memory/CPU capped, wall-clock timeout. |

---

## 7. Decisions

### 7.1 Settled

Decided 2026-09-05.

| ID | Decision | Consequence |
|---|---|---|
| **D-1** | **Genome is prompt text; executable policy code deferred to Phase 5b.** | Phases 1–5 evolve strategy prose only. The Docker/WASM sandbox is *not* built until 5b, and 5b is opt-in behind `selection.executable_genomes`. That flag currently fails validation on purpose — turning it on without the sandbox would run mutated code unsandboxed on the host. |
| **D-2** | **Φ-tiered model routing.** Haiku → Sonnet → Opus as Φ rises, with effort rising alongside. | The Φ table in §4 is the live spec, mirrored in `configs/default.yaml`. Consciousness costs more in entropy *and* in dollars. Trade-off accepted: model capability is not held constant, so a Φ effect and a model effect are confounded — Phase 7 must include a fixed-model control run (T7.7) to separate them. |
| **D-3** | **Full Landauer + Carnot physics.** | §3 is implemented as specified, not as scalar counters. `assert_second_law` runs every tick. Constants are config-tunable so a debug preset can relax them, but the default run is strict. |
| **D-4** | **Logs first, dashboard last.** | Phases 1–5 emit JSONL only; analysis is offline. The live dashboard is built in Phase 6 once real runs have shown which metrics matter. Avoids building panels for metrics that turn out to be uninteresting. |
| **D-5** | **`SimulationService` is the only path to the physics.** The tick loop uses it exactly as an MCP client does. | No private fast path can drift from the tool surface. Never add one — a "just this once" direct `engine.apply()` call in the runner silently voids the anti-cheating guarantee. |
| **D-6** | **Identity is an unguessable capability token, not an `agent_id`.** Preferred over a transport header; a tool argument is the fallback for in-process transports. | Closes S1. Tokens come from the seeded RNG so runs stay reproducible — they are in-process capability handles, **not cryptographic secrets**. Binding the server beyond loopback requires switching to `secrets` and giving up byte-identical logs. |
| **D-7** | **MCP transport: in-process for the tick loop and tests; streamable-http when organisms need real sessions.** stdio rejected. | stdio means one OS process per organism, which is absurd for 20–40 organisms sharing one world. |
| **D-8** | **One API request per tick in the common case** (resolves OPEN-6). The percept is pre-loaded, so a decisive organism needs exactly one call. Deliberation tools (`sense_gradient`, `recall`, `remember`) loop; the four action tools end the turn. | Per-graph-node requests would have cost ~4× for almost no extra behaviour. And because every extra round-trip is real tokens and real tokens are real entropy, **curiosity is metabolically expensive by construction** — capped by `cognition.max_tool_rounds`. |
| **D-9** | **Cache reads generate less entropy than fresh input** (resolves OPEN-10), weight 0.1 by default, matching their price. | Makes cache-stable lineages measurably fitter in the physics as well as on the invoice. Set `cache_read_entropy_weight: 1.0` to switch the pressure off and test whether it was doing any work. |
| **D-10** | **Reproducibility is scoped to `brain=mock`.** Every model request and response is logged in full instead. | Model sampling is not deterministic. An LLM run can be *replayed and audited*, never *reproduced* — and the code says so rather than pretending otherwise. |
| **D-11** | **The entropy discount is about information stability, not billing.** D-9 discounted cache reads because they are cheaper to buy. That was the right rule for the wrong reason. The rule is: **re-reading an unchanged block is not new cognitive work, whether or not a provider metered it.** | Generalises D-9 to any backend. Ollama reports every token as fresh input; without this, an organism re-reading the identical laws block each tick is charged full entropy for information it already holds — which killed every organism inside 3 ticks on the first local run. `local_simulate_cache` counts the stable prefix as a cache read after its first sighting. Set it false to see the un-discounted world. |
| **D-12** | **`local` is a first-class third brain, not a debugging toy**, and local runs use the Claude price table as a virtual currency. | Removes money as a precondition for every interesting experiment. The budget guard, spend ledger and Φ routing are exercised identically on the free path, so the paid path is not a separate, less-tested code path. Cost: `virtual_pricing` means a reported "$" figure is only real when `brain=llm` — every report must say which. |
| **D-14** | **An organism perceives the outcome of its own last act** (closes OPEN-15). `Percept.last_action` carries kind, success, and the refusal reason; the memory is lost at `CONFUSED`, like everything else disorder takes. | Recorded in `SimulationService._perform` and `_recording`, which is the one place every act — accepted, physically refused, or malformed — passes through, so the MCP and internal paths cannot diverge. **This is a sense, not a rule**: it says what happened, never what to do, and a brain is free to ignore it. The mock brain does, and its runs are unchanged to the bit — which is what makes it still a clean control. |
| **D-21** | **Scenario interventions are split into `set` (change a rule) and `do` (change the state).** Config overrides are rebound across world, engine and service in **one** function. | The two answer different questions: `set` asks what organisms adapted to one world do in another, `do` asks what they do after a shock in the same one. And the single rebinding point is what keeps invariant 4 true — two `Config` instances means the engine charges one price while the world applies another. |
| **D-22** | **`phi.prediction_weight` ships at 0.0** even though turning it up produces the project's only positive Φ result. | The mechanism does not hold: high-Φ organisms are measurably *more* surprised, not less, so the positive differential is a confound rather than a discovery. A mechanism that has not been validated does not get to be the default, however good its headline number looks. |
| **D-18** | **Φ buys error correction** (closes OPEN-11). Integration divides the corruption disorder inflicts on perception and memory, scaled by `phi.resilience`. | The defensible benefit: a system whose parts carry information about each other can reconstruct a damaged part from the rest. **It was measured and it does not repay Φ's price** — see §9 v0.9. The knob remains, set to 0.0 for the pure-cost control. |
| **D-19** | **Co-activation is scored as pairwise mutual information, not as counts.** | A count-based graph calls perfect repetition maximal integration, which is backwards: items always active together are redundant, not integrated. MI scores monotony at exactly zero. This is the single most important detail of the measure. |
| **D-20** | **Φ's vocabulary of experience is automatic and shared by every brain**, not built from `remember` calls alone. Tier thresholds are re-measured from the observed distribution rather than chosen. | The mock brain never calls `remember`, so an explicit-memory graph would give the control condition zero integration and make every brain comparison about tool use rather than thought. And a tier no organism reaches is a routing rule that never fires — worse than no rule, because it looks like one. |
| **D-15** | **Replication is an action an organism chooses, not an event that happens to it.** A fifth action tool, gated by a heritable threshold `policy.replicate_at`. | Reproduction competes with survival for the same energy, which is the trade-off worth simulating. It also gives selection something concrete to act on: *when* to breed is now a number that mutates and is measured. |
| **D-16** | **A child divides the parent's entropy; it does not copy it.** Child gets `child_entropy_fraction` of the parent's disorder, parent keeps the rest, total unchanged by the split. | The first implementation *created* the child's share, and the resulting ratchet drove every lineage extinct within four generations while the world still held most of its food. The division also makes replication a way to shed disorder — a real strategy (asymmetric division), paid for honestly, since the child is less viable for it and the lineage is selected against through that child. |
| **D-17** | **Mutation is point mutations over words: delete, duplicate, transpose. Nothing is ever written in.** | Two reasons. *Scientific:* if a lineage's strategy improves, no one wrote the improvement — the alternative, asking a model to "rewrite this slightly", makes every adaptation partly the model's design and partly ours. *Safety:* it closes S2 by construction. Mutation is closed over the ancestor's vocabulary, so no new token — and therefore no injected instruction — can appear in a genome that did not start with one. A test asserts this over 200 generations. |
| **D-13** | **The stable prefix includes the tool schemas.** Prompt render order is `tools → system → messages`, so the schemas sit ahead of the laws block and are just as unchanging. | 721 tokens of invariant JSON were being charged as fresh thought every single turn — more than the laws block itself. Any future change to the tool surface is therefore a *physics* change too: it moves the boundary between stable and volatile tokens. |

### 7.2 Still Open

| ID | Question | Raise by |
|---|---|---|
| OPEN-7 | Target scale for a headline experiment. Now answerable with numbers: a 200-tick / 20-organism run projects at **$4.16** on Haiku or **~$1.25** on Opus 5 (fewer turns, because high-Φ organisms die faster). | Phase 7 |
| OPEN-9 | The mock brain is the permanent control condition. Should Phase 7 compare LLM vs mock on *identical seeds and worlds*, or on matched-difficulty distributions? Affects what the headline result can claim. | Phase 7 (T7.2) |
| OPEN-21 | **The Φ proxy and the surprisal model share a basis.** Both are computed from the same pairwise statistics over the same window, so they cannot independently measure "integration" and "predictive quality" — they are two readings of one set of numbers, which is why the OPEN-18 test came out confounded. A real test needs a predictor over *external* world variables: where the food will be, what the temperature will do. | **The next real experiment** |
| OPEN-18 | ~~Prediction is the better-motivated thing for Φ to buy.~~ **Tested in v0.10 and inconclusive** — the differential turns positive at `prediction_weight` 2.0, 5/5 seeds, but high-Φ organisms are *more* surprised (r = +0.29 to +0.47, 0/5 seeds negative), so the mechanism is refuted and the result is a confound. Superseded by OPEN-21. | — | Error correction was defensible and it failed. Still, Sivak, Bell & Crooks (2012) prove that a system which models its environment badly *necessarily* dissipates more energy — so integration would pay in joules rather than in protection, which is both closer to this project's premise and closer to real physics. Concretely: have an organism record a prediction, measure the error, and charge for it. | **Next, with Phase 6** |
| OPEN-19 | ~~Is the world too easy to model?~~ **Refuted in v0.10.** Six scenarios including a swinging climate and food that physically moves; not one produces a positive differential. Organisms do become slightly more integrated under pressure (mean Φ 0.641 → 0.67) and it still does not help them reproduce. | — |
| OPEN-20 | `phi_entropy_exponent = 1.4` came from the project plan, not from measurement. The selection differential is roughly neutral below `k ≈ 1` and sharply negative above it. Nothing justifies 1.4 over 0.9 except that the plan said "superlinear". | Phase 7 |
| OPEN-16 | **A selection differential is not a response to selection.** v0.8 measured that organisms which reproduced had higher `dissipate_at` than those that did not, 5/5 seeds. It did not establish that the population mean moves and stays moved. Needs more seeds, larger populations, and a drift control. | Phase 7 |
| OPEN-17 | **Longevity and fecundity now trade off, and selection favours fecundity.** Frugal genomes live ~45% longer; wasteful ones leave ~2× the children. Selection therefore pushes `dissipate_at` *up* — toward dirtier, shorter-lived organisms. Is that the interesting result, or a sign that replication is underpriced relative to erasure? | Phase 7 |
| OPEN-12 | **Does a 3B local model actually decide better than the mock policy?** *First answer, v0.6: no — decisively worse.* On matched seeds it went extinct at tick 9 while the mock kept 3/3 alive and tripled their energy. It tried to eat where there was no food in 19 of 23 turns and never once dissipated entropy. Open question is now the sharper one: **is that the model, the prompt, or the affordances?** | Phase 7 (T7.2) — first data in hand |
| OPEN-13 | Local and API runs are not the same experiment: different model, no real cache, ~50× slower per turn. Which claims are allowed to travel from a local run to a paid one? Proposal: *mechanism* travels (does behaviour X emerge at all), *magnitude* does not. | Phase 7 |
| OPEN-14 | `compute_seconds_entropy_weight` makes hardware speed part of the physics — the same genome is fitter on a faster machine. Worth it for the honesty, or does it destroy comparability between runs? Currently `0.0`, so unanswered by default. | Phase 6/7 |

---

## 8. Task Breakdown

Reordered from the PDF's 8 weeks so that **physics is validated before money is spent**.
PDF week mapping noted per phase.

### Phase 0 — Foundations *(new; prerequisite)* — ✅ COMPLETE

- [x] T0.1 Python version confirmed: **stay on 3.14.3**. Full stack resolves — `numpy` 2.5.2, `anthropic` 1.4.0, `mcp` 2.1.1, `langgraph` 1.2.11. No downgrade needed.
- [x] T0.2 Repo skeleton, `pyproject.toml`, `.venv`, dependencies installed
- [x] T0.3 Typed config system — `src/btcs/config.py`. Strict loading: unknown keys raise. All physics constants live in `configs/default.yaml`.
- [x] T0.4 Seeded RNG service — `src/btcs/rng.py`. Named independent substreams, so adding a consumer never shifts another's draws.
- [x] T0.5 Event-sourced JSONL logger — `src/btcs/events.py`. Deterministic serialisation, per-tick flush, closed event vocabulary.
- [x] T0.6 `pytest` harness — **53 tests passing**
- [x] T0.7 `.gitignore`, `.env.example`, `git init` done (no commit made yet)

**Note:** `anthropic` resolved to the **1.x** major line. Phase 3 must be written against the 1.x API surface, not 0.x patterns.

### Phase 1 — Ecosystem Simulation Sandbox *(PDF Weeks 1–2a)* — ✅ COMPLETE

- [x] T1.1 2D toroidal grid: temperature field + free-energy nodes, vectorized NumPy — `world/grid.py`
- [x] T1.2 Node generation, cooling, leak, diffusion, radiative sink coupling, respawn
- [x] T1.3 Agent registry with `E`, `S`, `Φ`, position, generation, genome — `world/agents.py`
- [x] T1.4 Law L1 — Landauer erasure cost
- [x] T1.5 Law L2 — Carnot-bounded ingestion *(cold reservoir = local cell; see §3.4)*
- [x] T1.6 Law L3 — token→entropy translator, superlinear in Φ
- [x] T1.7 Law L5 — work + waste-heat deposition at origin and destination
- [x] T1.8 `ΔS_tot` ledger + **second-law assertion** (L4) — `world/ledger.py`
- [x] T1.9 Degradation ladder + death/erasure — perception corrupts before the organism dies
- [x] T1.10 **Mock brain** — deterministic, genome-driven — `brains/mock.py`
- [x] T1.11 Headless runner + CLI — `runner.py`, `cli.py`
- [x] T1.12 Physics test suite — **166 tests passing**

**Exit criteria — all met (verified 2026-09-05):**

| Criterion | Result |
|---|---|
| 200-tick, 20-agent mock run completes | ✅ 200 ticks, 15 alive, 1.01s, **0 API calls** |
| `ΔS_tot ≥ 0` on every tick | ✅ minimum per-tick ΔS = **+99.9 bits**; cumulative 43,283 bits, monotonic |
| Same seed → byte-identical logs | ✅ sha256 `331c7a21…` reproduced exactly |

### Phase 2 — MCP Server Proxy *(PDF Weeks 1–2b)* — ✅ COMPLETE

- [x] T2.1 MCP server scaffold — `mcp_server/server.py`. **SDK is `mcp` 2.x: `FastMCP` was renamed `MCPServer`** and the plan's v1 import path raises.
- [x] T2.2 Capability-token identity (closes S1) — `agent_id` appears in no tool, anywhere
- [x] T2.3 `sense_gradient` · [x] T2.4 `ingest_energy` · [x] T2.5 `dissipate_entropy`
- [x] T2.6 `perform_work` + `rest` · [x] T2.7 `remember` / `recall`
- [x] T2.8 Guardrails: turn discipline, per-turn call ceiling, one act per turn, finite-number checks, step clamping
- [x] T2.9 Structured error contract — closed `ErrorCode` set, no tracebacks reach an organism
- [x] T2.10 MCP integration tests — **219 tests passing**

**Exit criteria — met.** The headless run produces an *identical event multiset* to Phase 1
(8,329 events, 15 survivors, 43,283 bits cumulative ΔS). Three log lines reordered — a refusal is now
recorded when it happens rather than after degradation — and rejection lines carry a new `code` field.
Physics unchanged.

**Architectural decision taken here (D-5): the service layer is the *single* path.** The tick loop does
not reach the engine directly; it calls `SimulationService` exactly as an MCP client would, using a
capability token. Had the runner kept a private fast path, the MCP surface would have been a second
implementation free to drift, and "an organism cannot cheat because it can only act through tools"
would have been true of the protocol but not of the code.

### Phase 3 — Cognitive Layer & Token Economy *(PDF Weeks 3–4)* — ✅ COMPLETE

- [x] T3.1 LangGraph Sense → Integrate → Reason → Act cycle — `cognition/graph.py`, behind `cognition.use_graph`
- [x] T3.2 LangGraph `MemorySaver` checkpointer for pause/resume
- [x] T3.3 Anthropic SDK brain — `brains/llm.py`, same interface as the mock brain
- [x] T3.4 Prompt cache layout — `cognition/prompt.py`, breakpoint on the last system block
- [x] T3.5 Cache-hit verification + silent-invalidator guards (eight tests on prefix stability)
- [x] T3.6 Real `usage` → entropy, with the OPEN-10 weights — `cognition/pricing.py`
- [x] T3.7 Φ-band model/effort router
- [x] T3.8 Budget guard + kill switch + clean halt — `cognition/budget.py`
- [x] T3.9a **Cost estimator** — `cognition/estimate.py`, prices a run with zero API calls
- [x] T3.9b **First live run on a real model: 3 organisms, 20 ticks** — run on a local model, because
      there is no `ANTHROPIC_API_KEY` on this machine and the exit criterion is about *a real model
      making real tool calls*, not about which vendor billed for it.
- [ ] T3.9c Paid-API confirmation run — optional, and no longer blocking. Needs a credential.

#### Phase 3.5 — Free minds *(added v0.6; unplanned, and the highest-leverage thing built so far)*

- [x] T3.10 Ollama brain — `brains/local.py`. Same `Brain` protocol as mock and llm; the physics
      cannot tell them apart.
- [x] T3.11 Tool-schema translation Anthropic → Ollama, plus `local_repair_arguments`: small models
      routinely omit a required argument, and filling a sensible default is kinder — and more
      interesting — than voiding the turn. A vague intention is still a choice.
- [x] T3.12 Virtual Claude pricing for local runs (D-12), so the budget guard and Φ economics are
      exercised on the free path.
- [x] T3.13 Simulated prefix cache (D-11, D-13) — the fix that made local organisms survivable.
- [x] T3.14 `compute_seconds_entropy_weight` — real hardware seconds as internal entropy, off by
      default (OPEN-14).
- [x] T3.15 `--brain local` / `--local-model` CLI wiring, config validation, tests.

**286 tests passing, none of which make an API call.** The suite runs against a scripted fake client
(`tests/fakes.py`); a test that costs money is a test that gets skipped. The local brain is tested the
same way — against a fake transport, not a running server.

**Exit criteria — all three met, total spend $0.00:**

| Criterion | Status |
|---|---|
| Cache hit rate > 80% | ✅ **85–86% projected**, measured from the real prompt-assembly path |
| Cost per agent-tick documented | ✅ $0.00125 (Haiku) to $0.00640 (Opus 5, effort high) |
| Live agents survive multiple ticks | ✅ on `qwen2.5:3b` via Ollama — they survive, then die of their own thinking. See the v0.6 log entry; the *result* is in §10.2c |

### Phase 4 — Consciousness Index (Φ) *(PDF Weeks 3–4b)* — ✅ COMPLETE

- [x] T4.1 Memory graph — `world/memory.py`. Keyed store plus a rolling window of episodes recording
      what was active together.
- [x] T4.2 Co-activation → **pairwise mutual information**, not raw counts. Perfect correlation is
      redundancy, not integration, and a count-based graph scores monotony as maximal.
- [x] T4.3 Φ-proxy via spectral bipartition (Fiedler vector), normalised per element of the smaller side.
- [x] T4.4 **Φ buys error correction** (D-18) — integration divides the corruption disorder inflicts.
      This was OPEN-11's answer, and the measurement refuted it.
- [x] T4.5 Φ → cost coupling — already live since Phase 1; tier thresholds re-measured (D-20).
- [x] T4.6 30 tests, including the two that matter: monotony scores zero, independent halves score zero.
- [x] T4.7 `docs/phi.md` — the proxy's limits, and every number behind the negative result.
- [x] T4.8 Dynamic Φ with hysteresis and rate limiting (closes OPEN-8).

**Exit criteria — met, with a negative headline result:**

| Criterion | Result |
|---|---|
| Fragmented memory scores low, integrated high | ✅ 0.005 vs 4.5 on synthetic data; monotony 0.0 |
| Φ is dynamic and inherited | ✅ measured per organism from its own experience; children start at the parent's Φ with an empty graph |
| Φ buys something measurable | ✅ error correction, and it is a config knob |
| **Φ is worth its price** | ❌ **no.** Selection favours higher Φ in 0 of 5 seeds. See §9 v0.9 and `docs/phi.md`. |

#### Phase 3.6 — Perception of consequence *(v0.7; closes OPEN-15)*

- [x] T3.16 `LastAction` on the percept — kind, success, refusal reason, tick.
- [x] T3.17 Recorded at the single choke point in `SimulationService`, covering
      physically-refused *and* malformed acts.
- [x] T3.18 Lost at `CONFUSED`, like every other thing disorder takes. The organism keeps the record;
      it simply cannot reach it.
- [x] T3.19 Rendered into the volatile half of the prompt only — a test asserts it never enters a
      cached system block, since it changes every tick by construction.
- [x] T3.20 11 tests. The mock control is bit-identical afterwards (3/3 alive, 3,312.4 bits), which is
      how we know the sense costs nothing and biases nothing.

### Phase 5 — Natural Selection & Epigenetics *(PDF Weeks 5–6)* — ✅ COMPLETE

- [x] T5.1 `replicate()` — a fifth action tool (D-15). Construction cost as waste heat, endowment
      transferred, entropy divided (D-16), generation and parent recorded.
- [x] T5.2 Genome mutation on both halves — point mutations over words plus gaussian drift on the
      policy numbers (D-17), with length cap and range clamping.
- [x] T5.3 Lineage recording — `replication` and `agent_born` events carry parent, generation and the
      child's full genome, so a tree is reconstructable from the log alone.
- [x] T5.4 Dementia injection at `S_dement` — already live since Phase 1; confirmed under reproduction.
- [x] T5.5 Carrying capacity enforced; extinction detection already in the runner.
- [x] T5.6 Selection measured on the free brain (see below). The paid and local brains inherit the
      same tool, unchanged.
- [x] T5.7 Selection tests — split in two, because reproduction made longevity and fecundity different
      things (OPEN-17).
- [x] T5.8 `tests/test_selection.py` — 30 tests: the act, refusals, identity through the family,
      heredity, mutation safety, birth accounting, lineage, reproducibility.

**Exit criteria — all met:**

| Criterion | Result |
|---|---|
| Organisms reproduce | ✅ 52–153 births per 1,200-tick run |
| Children differ from parents | ✅ both genome halves drift; prose drift verified over 40 generations |
| Lineages reach depth | ✅ 12–35 generations |
| The books balance across a birth | ✅ ledger audit passes; a test checks the split sums exactly |
| Runs stay reproducible | ✅ mutation draws from the seeded RNG; two identical runs are byte-identical |
| Selection is measurable | ✅ see the v0.8 log entry |

### Phase 5b — Executable Genomes *(opt-in; deferred by D-1)*

Only built once text-genome selection is observed working. Gated behind
`selection.executable_genomes`, which currently fails config validation by design.

- [ ] T5b.1 `policy_code` genome field + safe AST-validated parsing
- [ ] T5b.2 Docker or WASM sandbox — no network, memory/CPU capped, wall-clock timeout
- [ ] T5b.3 Wall-clock execution time charged as metabolic energy (slow code literally starves)
- [ ] T5b.4 Code mutation operators distinct from prose mutation
- [ ] T5b.5 Sandbox escape tests

### Phase 6 — Macro-Scenario Controls & Dashboard *(PDF Weeks 7–8)* — 🟡 6a done

**Promoted by the Phase 4 result**, and it earned the promotion: T6.1 and T6.2 were never dashboard
scaffolding, they were the apparatus for the experiment that could have overturned the headline result.
It did not — see §9 v0.10 — but the question is now answered rather than open.

- [x] T6.1 Environment override protocol — `sim/scenario.py`. Dotted config paths rebound across
      world, engine and service in one function (D-21), so invariant 4 cannot be broken by an override.
- [x] T6.2 Preset scenarios — `heat_wave`, `drought`, `gradient_collapse`, `bottleneck`, `seasons`,
      plus `none`. **Severities were measured, not chosen**: a scenario that reliably wipes the
      population out measures extinction rather than adaptation, and one nothing notices measures
      nothing. Each was tuned until survival was in doubt but not settled.
- [x] T6.2b `--scenario` CLI flag; unknown names and unknown actions fail at *load*, not at tick 300.
- [x] T6.2c Prediction cost (`phi.prediction_weight`) — the Still et al. mechanism, implemented,
      measured, and shipped **off** because the mechanism did not hold (D-22).
- [x] T6.3 Metrics collector — `analysis.py`. Population, energy, entropy, temperature, food,
      lineage depth, action mix, death causes, refusals, spend.
- [x] T6.4 Offline analysis from the JSONL log — `analysis.py`. **Every result in §9 was found by a
      throwaway script; they are now library code with tests.** The selection differential, the
      instrument behind every evolutionary claim the project makes, lives here.
- [x] T6.5 Replay — positions are *reconstructed*, not stored: birth records a starting cell, every
      move records the new one, and the initial node layout is a deterministic function of the seed
      the log carries. Nothing had to be snapshotted.
- [x] T6.6 Terminal report — `report.py` (Rich). Falls back to ASCII where the console cannot encode
      block characters, because a report that raises `UnicodeEncodeError` halfway through a table is
      worse than one drawn in punctuation.
- [x] T6.7 **Visual playback** — `viz.py` writes one self-contained HTML file: the world animated tick
      by tick, a lineage tree of every organism that lived, the selection table, and the scenario
      timeline. No server, no dependencies, no network.
- [-] T6.8 Live web dashboard — still dropped by D-4. `viz.py` renders a *finished* run, which is
      what the analysis actually needed; a live panel would add a server for no measured benefit.

### Phase 7 — Experiments & Documentation

- [ ] T7.1 Reproducible experiment configs
- [ ] T7.2 Baseline: mock brains vs LLM brains under identical conditions
- [ ] T7.3 Experiment: does Φ rise under environmental pressure?
- [ ] T7.4 Experiment: do frugal genomes dominate under scarcity?
- [x] T7.5 `README.md` — what it is, how to run it, what it found, and a sixty-second explanation
      for someone who has never seen it.
- [x] T7.6 Limitations — stated in three places rather than one: `README.md` (real vs analogy vs not
      claimed), `GUIDE.md` §2, and `docs/phi.md` §1 and §6.
- [ ] T7.7 **Fixed-model control run** — required by D-2. Φ-tiered routing confounds a Φ effect with a model-capability effect; a run holding the model constant while varying only effort separates them.

---

## 9. Progress Log

### 2026-09-05 — v0.1

- Read and analyzed the project plan PDF.
- Produced this construction document: architecture, physics spec, Φ approach, cost model, security
  model, 7-phase task breakdown.
- **Six changes proposed against the PDF:** real thermodynamic formulation (Landauer + Carnot) over
  scalar counters; Φ-proxy with honest framing over an integer; session-bound identity to close the
  `agent_id` spoofing hole; mock-brain-first build order to validate physics at zero cost; direct
  Anthropic SDK for the model call so `cache_control`/`effort` stay reachable; refreshed model IDs.
- Four open decisions raised (§7). No code written pending answers.

### 2026-09-05 — v0.2 — Phase 0 complete

**Decisions taken:** D-1 text genomes first (sandbox deferred to 5b) · D-2 Φ-tiered model routing ·
D-3 full Landauer + Carnot physics · D-4 logs first, dashboard last. Recorded in §7.1.

**Dependency verification.** The whole stack resolves and imports on Python 3.14.3 — no downgrade
needed. `anthropic` came in at **1.x**, so Phase 3 targets the 1.x surface.

**Built:**

| File | Purpose |
|---|---|
| `src/btcs/config.py` | Typed config. Strict loading — an unknown key raises instead of silently running the default. `validate()` enforces cross-field invariants (thresholds ordered, efficiencies in range, node temperature above ambient). |
| `src/btcs/rng.py` | Named independent RNG substreams from one master seed. Adding a consumer cannot shift another's draws, so runs stay reproducible as the code grows. |
| `src/btcs/events.py` | Append-only JSONL log with deterministic serialisation (sorted keys, no wall-clock in the payload) and per-tick flush, so a crashed run still ends on a complete tick. |
| `configs/default.yaml` | Every physics constant, matching the dataclass defaults — a test asserts the two never drift apart. |

**Tests: 53 passing.** The load-bearing ones are the RNG order-independence tests and the
byte-identical serialisation test — together they are what will let Phase 1 assert that two runs with
the same seed produce identical logs.

**Two guard rails worth noting.** `selection.executable_genomes: true` currently *fails* config
validation: flipping it on before Phase 5b exists would run mutated code unsandboxed on the host, so
the failure is deliberate. And the config rejects a world whose nodes start at or below ambient
temperature — under the Carnot bound (L2) such a world contains no extractable free energy at all,
and would look like a physics bug rather than a configuration mistake.

**Next:** Phase 1 — the NumPy world, the five laws, the mock brain, and the headless runner.

### 2026-09-05 — v0.3 — Phase 1 complete

**Exit criteria met.** 200-tick / 20-agent mock run: 200 ticks, 15 alive, 1.01s, **zero API calls**.
Minimum per-tick ΔS = +99.9 bits. Same seed reproduced a byte-identical log. **166 tests passing.**

**Built:**

| Module | Purpose |
|---|---|
| `world/ledger.py` | Double-entry entropy bookkeeping. Per-process attribution so a violation names the law that caused it. |
| `world/grid.py` | Toroidal temperature field + free-energy nodes. Diffusion, radiative sink coupling, node cooling and respawn. |
| `world/agents.py` | Organisms and registry. Deliberately cannot mutate their own energy or entropy. |
| `world/physics.py` | The five laws. The only module permitted to change organism state. |
| `world/types.py` | `Percept` / `Action` / `Decision` — the brain contract, shaped to become the Phase 2 MCP tool surface. |
| `brains/mock.py` | Deterministic genome-driven policy. Permanent control condition, not a placeholder. |
| `runner.py`, `cli.py` | Tick loop with both audits; headless CLI. |

**Two discoveries worth recording.**

*The Carnot cold reservoir must be the organism's own cell.* Using global ambient let an agent in a
hot cell extract more work than the second law permits — the ledger went negative within a few ticks.
Fixing it produced the crowding dynamic for free: waste heat raises `T_cold`, which shrinks everyone's
efficiency. Observed extraction efficiency in live runs is 0.34–0.50 against a 0.64 ceiling, and the
gap *is* the organisms cooking themselves.

*A second invariant was needed.* `ΔS_total ≥ 0` is satisfiable by an implementation whose books are
fiction, so each tick now also asserts the ledger's account of organism entropy equals the organisms'
actual change. A test proves the check has teeth by subclassing the engine to leak entropy silently.

**Calibration.** First run drove everything extinct by tick 30 — every death `entropy_overwhelmed`,
meals yielding 8 J against ~11 J/tick of thought and erasure. Measured the per-agent-tick budget,
found `node_cooling_rate: 0.02` gave gradients a ~35-tick half-life, so the world was thermally dead
by tick 150 regardless of behaviour. Retuned nine constants (recorded in `configs/default.yaml`) so
scarcity comes from competition rather than from the clock. The population now settles at a carrying
capacity (~9–15 from 16–20) instead of dying out, which is what gives selection room to act. A test
(`TestShippedDefaults`) fails if a future change collapses the population again.

**Selection signal confirmed at zero cost.** `TestSelectionSignal` shows a genome that never
dissipates is reliably outlived by one that dissipates sensibly. That was worth establishing before
Phase 5 — and before spending money on whether prompt evolution does the same.

**Open decisions raised:** OPEN-8, OPEN-9 (§7.2).

**Next:** Phase 2 — the MCP proxy, with session-bound identity closing the `agent_id` spoofing hole.

### 2026-09-05 — v0.4 — Phase 2 complete

**219 tests passing.** The headless run produces an *identical event multiset* to Phase 1 — 8,329
events, 15 survivors, 43,283 bits cumulative ΔS. Three log lines reordered (a refusal is now recorded
when it happens rather than after degradation) and rejection lines gained a `code` field. The physics
is untouched.

**Built:** `sim/service.py` (identity, turn discipline, budgets, argument validation, memory),
`mcp_server/server.py` (seven tools over that service), and a rewiring of the runner so the tick loop
plays through the service using capability tokens exactly as an MCP client would.

**Decisions taken — D-5, D-6, D-7 in §7.1:** the service is the single path to the physics; identity
is an unguessable capability token rather than an `agent_id`; transport is in-process for the loop and
the tests, streamable-http for real sessions, with stdio rejected as one OS process per organism.

**Three SDK realities the plan predates.** `mcp` 2.x renamed `FastMCP` to `MCPServer`, and the v1
import path raises outright. `ctx` must be annotated `Context` or the SDK treats it as a model-facing
tool argument — it leaked into all seven schemas until caught, and a test now guards it. And identity
resolution had to move *inside* the error guard: evaluated as an argument, a missing credential
escaped as a protocol error instead of the structured refusal an organism is meant to receive.

**S1 is closed and tested.** No tool takes an `agent_id`; a test asserts that of every service
signature and every published schema. The concrete exploit the plan allowed — organism A calling
`process_cognitive_step("Agent_B", 99999)` to kill a rival — now has no argument by which A can name B.

**Added §10, Forward Guidance,** at the user's request: the invariants that must survive future
changes, the Phase 3 API specifics verified against the installed SDK, and the phase interactions that
are invisible from either side alone. The sharpest of those: **Φ tier changes invalidate the prompt
cache**, because tiers swap models and caches are model-scoped — which is why Phase 4 should give Φ
hysteresis rather than recomputing it every tick.

**Also recorded:** reproducibility does not survive an LLM brain. Phase 3 must log every model request
and response so runs can be replayed and audited even though they cannot be reproduced, and the
reproducibility tests must be scoped to `brain=mock`.

**Next:** Phase 3 — the LangGraph loop, the Anthropic SDK brain, prompt caching, and the first run
that costs money.


### 2026-09-06 — v0.5 — Phase 3 built; live run blocked on credentials

**286 tests passing, none making an API call.** The suite drives a scripted fake client. There is no
`ANTHROPIC_API_KEY`, no `ant` CLI and no `.env` on this machine, so T3.9b — the first live run — is
the one task left open.

**Built:** `cognition/pricing.py` (prices, usage, the token→entropy weighting), `cognition/budget.py`
(spend ledger and kill switch), `cognition/prompt.py` (cache-structured prompt assembly),
`cognition/graph.py` (the LangGraph cycle), `cognition/estimate.py` (prices a run without calling
anything), and `brains/llm.py`.

**Decisions D-8, D-9, D-10 in §7.1.** One request per tick, with deliberation tools looping and action
tools ending the turn; cache reads weighted at 0.1 for entropy as well as for price; reproducibility
scoped to the mock brain with full request/response logging in its place.

**Measured, without spending anything.** The estimator runs the real prompt-assembly path through a
mock population, so these numbers move when the prompt does:

| Run | Model | Projected | Per turn | Cache hits |
|---|---|---|---|---|
| 200 ticks × 20 organisms | Haiku 4.5 (Φ=1) | $4.16 | $0.00125 | 86% |
| 200 ticks × 20 organisms | Sonnet 5 (Φ=3) | $1.31 | $0.00256 | 85% |
| 200 ticks × 20 organisms | Opus 5 (Φ=6) | $1.25 | $0.00640 | 85% |
| 20 ticks × 3 organisms | Haiku 4.5 | $0.08 | $0.00130 | 84% |

**The finding worth staring at: total cost *falls* as Φ rises.** Per-turn cost goes up fivefold on
Opus, but a high-Φ population generates entropy superlinearly under L3 and dies roughly 28× sooner, so
it never survives long enough to run up a bill. Consciousness is expensive per thought and cheap per
lifetime, because it shortens the lifetime. That is a real property of the model as specified, not a
bug — but it does mean **Φ currently has cost and no benefit**, which is now OPEN-11 and is the single
most important thing for Phase 4 to address. If higher Φ does not buy better decisions, selection will
simply drive Φ to zero and the project's central claim goes untested.

**Cache economics confirmed at 85–86%.** The frozen laws block is byte-identical for every organism in
the population, and clones inherit a genome verbatim, so a lineage shares its whole prefix.
Replication makes a lineage cheaper to run — and with D-9, physically cooler too.

**Two SDK details verified rather than recalled.** LangGraph decides whether to inject the invocation
config from the *annotation*: a node typed `Any` gets `None` silently and the live context never
arrives, and under `from __future__ import annotations` even `RunnableConfig | None` fails the check —
it must be the bare `RunnableConfig`. And Haiku takes neither `output_config.effort` nor adaptive
thinking, so the cheapest Φ tier sends neither parameter.

**Next:** Phase 4 — the memory graph and Φ. Start with OPEN-11: Φ must buy something, or it is only a
tax. And give Φ hysteresis, because a tier change throws away the organism's cached prefix.

### 2026-09-06 — v0.6 — Phase 3 closed on local models; the first real-model result

**Phase 3 is complete. Total spend: $0.00.**

T3.9b asked for a live run of 3 organisms over 20 ticks on a real model. There is still no
`ANTHROPIC_API_KEY` on this machine, so it was run on `qwen2.5:3b` through Ollama instead — the
criterion is about a real model making real tool calls, not about which vendor sent an invoice. The
paid confirmation run (T3.9c) is now optional rather than blocking.

**Built (Phase 3.5, unplanned):** `brains/local.py` — an Ollama brain behind the same `Brain` protocol
as the other two, so the physics cannot tell what is thinking. Tool schemas are translated from the
shared definitions rather than rewritten, so a comparison between brains is about the models and not
about two different descriptions of the same world. Plus virtual Claude pricing (D-12), the simulated
prefix cache (D-11, D-13), `compute_seconds_entropy_weight`, and `tests/test_local.py` — 27 tests
against a fake transport, **313 passing in total, still none of them making a request**.

**Decisions D-11, D-12, D-13 in §7.1.** The important one is D-11: the entropy discount was never
about billing, it is about **information stability**. Re-reading an unchanged block is not new
cognitive work whether or not a provider metered it. Without that, the first local run killed every
organism inside three ticks — Ollama reports every token as fresh, so an organism was charged full
entropy each tick for the laws block and tool schemas it had already read. D-13 then folded the
721-token tool schemas into the same stable prefix, which makes any future change to the tool surface
a *physics* change as well.

**And a correction to that correction, found by writing the tests.** The first implementation
estimated the stable prefix and subtracted it from the reported input — and the estimator overshot the
real tokenizer by enough that the estimated prefix exceeded the *entire* prompt. Volatile input came
out at exactly zero: the percept, the only genuinely new information in a turn, was being swept into
the discount and read at a tenth of its entropy cost. **The organisms were perceiving the world for
free.** The fix is to measure the volatile side instead, so estimation error lands on the prefix,
where over-charging is the safe direction. Verified against live logs: 181–228 fresh tokens per turn
where there had been 0, and the three parts still sum exactly to what the server reported.

**The result — the first thing this project has measured that the code could not have told us.**
Three organisms, twenty ticks, matched seed, mock control alongside:

| | mock control | `qwen2.5:3b` |
|---|---|---|
| Survivors at tick 20 | 3 / 3 | **0 — extinct at tick 9** |
| Lifespans | — | 6, 8, 9 ticks |
| Energy, start → end | 450 → 1,162 J | 450 → 0 |
| Successful meals | 29 | **0** |
| Dissipations | 10 | **0** |
| Moves | 21 | 1 |
| Actions refused | 0 | **19 of 23 turns**, all "no node here" |
| Entropy per turn | 101 tokens | 571 tokens |

Every death was `entropy_overwhelmed`. **They did not starve; they cooked** — thinking expensively
about a meal that was not there. A hand-written rule beat a language model decisively, and did it
while spending a sixth of the entropy per turn.

The run was performed twice, before and after the accounting correction. The correction raised entropy
per turn from 367 to 571 and pulled extinction in from tick 17 to tick 9 — but the behaviour was
*identical*: zero meals, one move, the same refusal loop. The subsidy was hiding how badly the
organisms were doing; it was not the reason they were doing badly.

**What it points at (OPEN-15, and now P-0).** The most interesting of the three explanations is not
about the model at all: **an organism cannot perceive the consequences of its own actions.** `Percept`
carries vitals, terrain and visible nodes, but not what happened last tick. It is refused, told
nothing, and repeats the mistake. Every individual is as blind at death as it was at birth — so there
is no within-lifetime learning available to any brain, however good. That is worth closing before
Phase 5, because otherwise selection has only genome prose to work on.

**Phases reordered: Phase 5 (reproduction) now comes before Phase 4 (Φ).** Until organisms replicate
with mutation there is no heredity, no variation and no lineage, so the project cannot claim evolution
about anything it has run — only ecology. Φ is blocked on OPEN-11 in any case, and nearly every
question worth asking about Φ needs lineages to exist first.

**Also added:** `GUIDE.md`, a companion on how to think about the project — the "never write a rule you
could derive" discipline, an honest split of what here is real physics and what is analogy, a concept
backlog, and a reading list.

**Next:** Phase 5. Close OPEN-15 first (it is small), then `replicate()`, mutation and lineages.

### 2026-09-06 — v0.7 — organisms can perceive what they did

Closes OPEN-15. `Percept.last_action` carries the kind of the previous act, whether it happened, and
why not. It is recorded in `SimulationService` at the one point every act passes through, so a
physically impossible request and a malformed one are both remembered and the MCP and internal paths
cannot drift.

Two things it is careful about. **It is a sense, not a rule** — it reports what happened and never
what to do, so it is not a behaviour the physics should have produced on its own. And it is **lost at
`CONFUSED`**, like everything else disorder takes: the organism keeps the record and simply cannot
reach it, which is the asymmetry the rest of the ladder already uses.

**Measured on the free brain first:** the mock control is bit-identical afterwards — 3/3 alive, 3,312.4
bits — so the sense costs nothing and biases nothing for a brain that ignores it.

**Measured on a real model:** refusals fell from 19 of 23 turns to 10 of 19, and organisms began
sensing before acting instead of eating blindly. So the model *does* read it. Lifespans nonetheless got
slightly *shorter* (6/8/9 → 5/7/7 ticks), because the extra text and the extra deliberation rounds cost
entropy that a 3B model's improved decisions did not repay. **The sense works; the model is still too
weak to survive.** Both halves of that are worth keeping.

### 2026-09-06 — v0.8 — Phase 5: reproduction, and the first measured selection

**361 tests passing.** Organisms reproduce. This is the entry that lets the project say *evolution*
about anything it has run, so it is worth being exact about what was and was not shown.

**Built:** `Replicate` as a fifth action (D-15), entropy division rather than copying (D-16), word-level
point mutation over both genome halves (D-17), a heritable `policy.replicate_at`, carrying-capacity
enforcement, lineage events carrying the child's full genome, and `tests/test_selection.py` (30 tests).

**Two bugs found by running it rather than by reasoning about it.**

*The entropy ratchet.* The first version **created** the child's entropy instead of dividing the
parent's. Every birth inflated total organism disorder by half a parent, mean lifespan fell 117 → 70 →
16 across three generations, and the population went extinct at tick 437 **with 8,346 units of food
still in the world**. Dying of thirst in a lake is the signature of an accounting error, not of
scarcity. Division fixed the ratchet.

*Organisms starving while being told no.* `replicate_at` is heritable, so mutation drifts it below the
world's minimum — and an organism there spends every turn asking for a child it cannot afford. 38
wasted turns in one run. The policy now floors its own threshold at the world's, on the grounds that
knowing whether you are rich enough to breed is not privileged information.

**Calibration by measurement, as in Phase 1.** A sweep over the birth economics settled
`child_entropy_fraction` at 0.15 and `child_energy_fraction` at 0.5. At 0.4 the newborn cannot pay the
Landauer bill on the disorder it inherits, and lineages run down within a few generations.

**The trade-off nobody designed in.** Erasure costs energy; energy is what a child is made of. So on
matched seeds:

| | frugal (`dissipate_at` 45) | wasteful (never erases) |
|---|---|---|
| Mean lifespan | 39–55 ticks | 36–38 ticks |
| Children left | 19–26 | 39–48 |

The frugal genome lives ~45% longer and leaves half as many descendants. **Longevity and fecundity are
different currencies here**, which is the life-history trade-off real organisms face, and it fell out
of the Landauer bill rather than out of a rule. Both directions are now separate tests.

**The measured result: a selection differential on entropy tolerance.** Five seeds, 1,200 ticks,
12–35 generations, comparing organisms that reproduced against those that never did:

| Trait | Bred | Never bred | Difference | Seeds favouring higher |
|---|---|---|---|---|
| **`dissipate_at`** | **54.44** | **50.47** | **+3.98** | **5 / 5** |
| `replicate_at` | 237.66 | 237.94 | −0.28 | 3 / 5 |
| `greed` | 0.97 | 1.00 | −0.03 | 2 / 5 |
| `heat_aversion` | 0.94 | 0.94 | −0.00 | 3 / 5 |
| `explore_bias` | 0.19 | 0.19 | −0.00 | 2 / 5 |

**Selection favours postponing erasure** — dirtier, shorter-lived organisms that spend on children what
they would otherwise have spent on cleaning. The four traits with no signal are the internal control:
they drift under the identical mutation operator in the identical runs and come out at coin flips, so
the one signal is not an artefact of the mutation process.

**What this does not show (OPEN-16).** A selection differential says which organisms bred. It does not
establish a sustained *response* to selection — a population mean that moves and stays moved. Late-run
populations are 1–7 organisms, where drift is large. More seeds and larger populations are needed
before this is a result rather than an observation.

**Next:** Phase 4 — Φ, now unblocked because lineages exist and Φ is already inherited. Start with
OPEN-11: Φ must buy something, or selection will drive it to zero and answer the project's central
question by default instead of by experiment.

### 2026-09-06 — v0.9 — Phase 4: Φ is measured, and it does not pay for itself

**391 tests passing.** This entry records a negative result about the project's central claim. It is
the most valuable thing here so far, and `docs/phi.md` carries every number behind it.

**Built:** `world/memory.py` — a per-organism memory graph and the Φ-proxy over it. Φ is now computed
from an organism's own experience rather than pinned at a constant, is recomputed on a schedule with
hysteresis (closing OPEN-8), and is inherited at replication while the memory that justified it is not.

**The measure, and the one detail that matters most.** Co-activation is scored as **pairwise mutual
information**, not as counts (D-19). A count-based graph calls perfect repetition maximal integration,
which is backwards: items always active together are redundant, not integrated. With MI, monotony
scores exactly 0.0, independent halves score 0.005, and perfectly coupled halves score 4.5. Those three
numbers are the measure's licence to be trusted at all.

**Φ's vocabulary is automatic and shared by every brain** (D-20). The mock brain never calls
`remember`, so a graph built from explicit memories alone would have given the control condition zero
integration and made every brain comparison about tool use rather than about thought.

**What Φ buys (D-18, closing OPEN-11): error correction.** A system whose parts carry information about
each other can reconstruct a damaged part from the rest, so integration divides the corruption disorder
inflicts on perception and memory. Defensible, implemented, and measured — and the measurement went
against it.

**Result 1 — the protection does not repay its heat.** Sweeping `phi.resilience` over 5 seeds:
mean population 12.5 → 13.7 → 11.7 → 9.8 → 8.6 at resilience 0 / 0.25 / 0.5 / 1 / 2, with two
extinctions at the top. More protection is *worse*, and the mechanism is legible: shielding integration
means organisms keep more of it, and keeping it is what costs.

**Result 2 — selection runs against Φ.** The selection differential — the same instrument as Phase 5,
where four control traits came out at coin flips:

| resilience | bred | never bred | difference | seeds favouring higher |
|---|---|---|---|---|
| 0.0 | 0.617 | 0.729 | −0.112 | 0/5 |
| 0.5 | 0.609 | 0.819 | −0.210 | 0/5 |
| 2.0 | 0.626 | 0.807 | −0.181 | 0/5 |

**Result 3 — it is the price, not a confound.** The obvious objection is that a well-adapted organism
in a simple world repeats itself, and repetition is exactly what this measure scores as zero — so
"boring is adaptive" would produce that table with integration costing nothing. Testable: set
`phi_entropy_exponent` to 0 and make Φ free. **The differential flips positive, +0.034 in 4 of 5
seeds.** The driver is the cost. Integration's benefit is real and smaller than its bill.

**Result 4 — where it breaks even.** Sweeping the exponent: +0.017 (k=0), +0.013 (0.3), −0.005 (0.6),
−0.006 (0.9), −0.103 (1.2), −0.210 (1.4, shipped). Roughly neutral while cheap, sharply negative past
k ≈ 1. There is no regime where integration is clearly *favoured*.

**So: the project's central claim is not supported under this operationalisation.** That is a result,
not a failure — it is what a falsifiable claim looks like when it is actually tested. Three of the four
things it depends on are choices we made, and `docs/phi.md` §6 states which.

**Two calibration corrections, both worth keeping.**

*Dynamic Φ silently made thinking cheaper.* The world was tuned with Φ pinned at 1.0, where
g(Φ) = 2^1.4 = 2.64; measured Φ sits near 0.5, where g = 1.75. Every thought became a third cheaper in
disorder without anyone deciding it should, and the selection signal that is the project's whole
premise held in only 1 of 3 seeds. `entropy_per_token` 0.010 → 0.015 restores it in 3 of 3 and puts the
population back where it was. Note 0.015 / 0.010 = 1.5 against a predicted 2.64 / 1.75 = 1.51: **the
constant was derived from the cause, not fitted to the symptom.**

*The tier thresholds had never fired.* While Φ was a static 1.0 they were round numbers — 0 / 2 / 5 / 8
— and no organism ever reached the second. Re-measured against the observed distribution (median 0.38,
p90 1.36, p95 1.93, max 3.61) they are now 0 / 0.7 / 1.4 / 2.5, and a test fails if the top tier ever
rises above anything observed. `initial_phi` went from 1.0 to 0.0: a founder with no experience has no
integration, which is what the measure returns for an empty graph anyway.

**And a reproducibility bug worth remembering.** Iterating a `set` of strings follows hash order, which
Python randomises per process. Consuming the seeded RNG in that order made two runs of the same test
suite disagree about which seeds passed — determinism lost to an invisible detail, in code that looked
obviously correct. A test now guards it. **Never draw from a seeded stream while iterating a set.**

**Next:** Phase 6, promoted ahead of the dashboard. Error correction is worth little in a world where
the right answer rarely changes, so the strongest remaining explanation for Φ's failure is that the
environment is too easy to model (OPEN-19). Scenario controls are the apparatus for the experiment that
would overturn the headline result — and OPEN-18, prediction as what Φ buys, is the better-motivated
benefit the project skipped.

### 2026-09-07 — v0.10 — Phase 6a: the world can change, and two explanations are eliminated

**428 tests passing.** Phase 4 left four candidate explanations for why integration does not pay for
itself. Two of them are now closed by measurement, and the second closing is the more instructive.

**Built:** `sim/scenario.py` — live parameter overrides and world-state interventions, six presets, a
`--scenario` flag. Config changes are rebound across the world, the engine and the service in **one
function** (D-21), because two `Config` instances means the engine charges one price while the world
applies another and the ledger stops balancing in a way that is genuinely hard to trace. Also
`phi.prediction_weight`: the Still et al. (2012) mechanism, where being surprised by your own world
costs entropy.

**Scenario severities were measured, not chosen.** A scenario that reliably wipes the population out
measures extinction rather than adaptation; one nothing notices measures nothing at all. The first
`seasons` left a single survivor and the first `heat_wave` and `drought` went extinct outright, so all
three were softened until survival was in doubt but not settled.

**Result: a changing world does not rescue Φ (OPEN-19, refuted).** `seasons` swings the climate, stops
the food respawning, and **physically moves the nodes**, so an organism's map goes stale on a schedule.
Across six scenarios, five seeds each:

| scenario | mean Φ | mean population | difference | seeds favouring higher Φ |
|---|---|---|---|---|
| none | 0.641 | 13.7 | −0.155 | 0/5 |
| bottleneck | 0.662 | 11.5 | −0.115 | 1/5 |
| drought | 0.665 | 12.9 | −0.021 | 1/5 |
| heat_wave | 0.671 | 9.8 | −0.140 | 1/5 |
| gradient_collapse | 0.649 | 11.7 | −0.123 | 1/5 |
| **seasons** | 0.669 | 9.3 | **−0.107** | **1/5** |

Not one is positive. Organisms *do* become slightly more integrated under pressure — mean Φ rises from
0.641 to about 0.67 — and it still does not help them reproduce.

**The result that nearly became a false headline (OPEN-18).** With prediction charged as entropy, the
selection differential finally turns positive: **+0.123 at `prediction_weight` 2.0, five seeds out of
five**, and +0.138 under seasons. Written up at that point it would have read *integration pays once it
pays in joules*.

**The mechanism check killed it.** If integration buys prediction, higher Φ must mean *less* surprise.
Measured within runs: **r = +0.29 to +0.47, and 0 of 5 seeds negative in every condition.** High-Φ
organisms are consistently *more* surprised. The positive differential is a confound — behavioural
variety drives Φ and surprise upward together — and the winners are not better predictors, they pay
more surprise cost and reproduce anyway.

The cause is structural: **the Φ proxy and the surprisal model are computed from the same pairwise
statistics over the same window.** They cannot independently measure integration and predictive
quality, because they are two readings of one set of numbers. A real test needs a predictor over
*external* variables — where the food will be, what the temperature will do. That is OPEN-21 and it is
the next real experiment.

`phi.prediction_weight` therefore ships at 0.0 (D-22). **A mechanism that has not been validated does
not get to be the default, however good its headline number looks.** That principle is worth more than
the feature.

**Where the four candidate explanations now stand:** the world being too simple is refuted; prediction
as the benefit is inconclusive and confounded; the price being too high (OPEN-20) and the proxy being
too crude (now sharpened into OPEN-21) remain.

**Next:** T6.4–T6.5, offline analysis and replay from the JSONL log. Every result in the last three
entries was computed by a throwaway script reading the log; that work belongs in the repository where
it can be re-run and checked.

### 2026-09-07 — v0.11 — Phase 6b: the project can be read, watched, and explained

**454 tests passing.** Nothing new happens to the organisms in this entry. What
changed is that the project can now be *inspected* by someone who was not here when it was built —
which, for a project whose whole output is measurements, was the largest remaining gap.

**`analysis.py` — the results move into the repository.** Every finding in the entries above was
computed by a throwaway script reading the log. That was the right way to find them and the wrong way
to keep them: **a number nobody can recompute is a number nobody can check.** The selection
differential, trait-by-generation drift, lineage reconstruction, death causes and refusal tallies are
now library code with tests that check them against a live simulation rather than against a fixture
that agrees with them by construction.

**Replay reconstructs; it does not store.** Positions were never snapshotted and did not need to be:
birth records a starting cell, every move records the new one, and the initial node layout is a
deterministic function of the seed the log already carries. So the entire history of a run is
recoverable from an append-only text file — which is also why a run from months ago is exactly as
inspectable as one from this morning.

**`report.py` — the terminal panel (T6.6), built last as D-4 required**, on the metrics that real runs
proved worth watching. The refusal panel earned its place: it is where the finding about a language
model failing the same act nineteen times out of twenty-three came from, and no aggregate metric would
have shown it. It falls back to ASCII where the console cannot encode block characters, because a
report that raises `UnicodeEncodeError` halfway through a table is worse than one drawn in punctuation.

**`viz.py` — you can now watch it.** One self-contained HTML file per run: the world animated tick by
tick, organisms coloured by their rung on the degradation ladder, a lineage tree of every organism that
ever lived, the selection table, and the scenario timeline. No server, no dependencies, no network.

The honesty rule applied to a picture: **per-node energy is not logged, so it is not drawn.** A node
is shown present or absent and the legend says why, rather than inventing a brightness that would look
like data.

**Why the visual matters more than it sounds.** Two of the most important findings in this project —
the boom-bust overshoot, and organisms standing on empty ground trying to eat — are things you would
see in five seconds and might never think to compute. The `GUIDE.md` section on watching a run is a
table of what each visible pattern actually means, so that is transferable rather than folklore.

**`README.md`, which did not exist.** What the project is, how to run it, what it has found, and a
sixty-second explanation for someone who has never seen it. The limitations are stated in three places
now rather than one, because the place someone reads is not always the place you wrote them.

**`GUIDE.md` rewritten around what has actually been learned.** It opens with the whole project on one
page, then the discipline, then a new section on *how to measure anything here* — the selection
differential and its limits, the knob-out control, and watching the run — then the results and the
lesson inside them. The near miss is now the centrepiece: a positive result, five seeds out of five,
that dissolved under a mechanism check.

**Next:** the experiments in Phase 7, and the open item that matters most is OPEN-21 — a predictor
over *external* world variables, which is what turns the inconclusive OPEN-18 test into a real one.

---

## 10. Forward Guidance

Written while the context is fresh, for whoever picks up the next phase. Everything here is a trap
already identified, a decision already reasoned about, or an interaction between phases that is not
obvious from either phase alone.

**See also `GUIDE.md`** — the companion document. This section is *what to watch out for*; that one is
*how to think*: the "never write a rule you could derive" discipline, an honest split of what in this
project is real physics versus analogy, the concept backlog, and the reading list.

### 10.1 Invariants that must survive every future change

These are load-bearing. Breaking one does not produce a test failure in the phase that breaks it —
it produces a wrong result three phases later.

1. **Never add a second path to the physics.** The runner reaches the engine only through
   `SimulationService` (D-5). A direct `engine.apply()` in new code is not a shortcut, it is a hole in
   the anti-cheating guarantee.
2. **Every mutation of `agent.entropy` files a ledger entry.** The per-tick consistency audit
   (`LedgerInconsistency`) catches omissions, but only if the mutation happens inside a bracketed
   tick. New lifecycle events — birth in Phase 5 especially — must file explicitly.
3. **Config lives in two places that a test keeps in sync.** Add a field to the dataclass *and* to
   `configs/default.yaml`, or `test_shipped_yaml_matches_dataclass_defaults` fails. That test exists
   because the YAML doubles as documentation.
4. **`World` and `PhysicsEngine` must share one `Config` instance.** Enforced in the engine's
   constructor. Two configs means the engine charges one price while the world applies another, and
   the ledger stops balancing in a way that is genuinely hard to trace.
5. **No NumPy array crosses the service boundary.** Percepts are plain data; a test asserts the
   payload is JSON-serialisable.

### 10.2 Phase 3 — Cognitive Layer *(the expensive one; read before writing any API code)*

**The reproducibility guarantee does not survive contact with an LLM.** Everything so far is
byte-identical across runs. Model sampling is not. Do not pretend otherwise — instead **log every
model request and response in full**, so an LLM run can be *replayed and audited* even though it
cannot be *reproduced*. The reproducibility tests must be scoped to `brain=mock`.

**API specifics, verified against the installed SDK rather than recalled:**

- `anthropic` resolved to the **1.x** major line. Do not write 0.x patterns.
- Models: `claude-opus-5` ($5/$25 per Mtok), `claude-sonnet-5` ($2/$10), `claude-haiku-4-5` ($1/$5).
- Thinking is `{"type": "adaptive"}`. `budget_tokens` is **rejected with a 400** on Opus 5 / Sonnet 5.
- Effort belongs in `output_config: {"effort": ...}`, not at top level.
- Assistant prefill is removed on these models — use structured outputs or system instructions.
- Stream anything with a large `max_tokens`, and take the result via `.get_final_message()`.

**Two cost interactions that are not obvious from either phase alone:**

- **The prompt cache is model-scoped.** A Φ tier change swaps the model, which discards that
  organism's cached prefix. This couples Phase 4 directly to Phase 3's bill — see §10.3.
- **Cache reads cost roughly a tenth of fresh input.** Decide deliberately whether cached tokens
  should generate *less* entropy under L3. There is a real argument that re-reading a cached prefix is
  less cognitive work than reading it afresh, and it would make frugal, cache-stable organisms
  measurably fitter. Raised as **OPEN-10**.

**Token accounting must come from real `usage`** — input + output + cache reads, weighted however
OPEN-10 settles. The mock brain's invented counts exist only to exercise L3.

**OPEN-6 is settled (D-8): one API request per tick** in the common case. The percept is pre-loaded,
so a decisive organism needs exactly one call; deliberation tools loop and action tools end the turn,
capped by `cognition.max_tool_rounds`.

**Verified during the build, not recalled:**

- LangGraph decides whether to inject the invocation config from the *annotation*. A node typed `Any`
  is silently handed `None` and the live context never arrives; under `from __future__ import
  annotations` even `RunnableConfig | None` fails the check. Use the bare `RunnableConfig`.
- Haiku takes neither `output_config.effort` nor adaptive thinking, so the cheapest Φ tier must send
  neither parameter. A tier with `effort: null` is the signal for that.
- Measured cache hit rate is **85–86%**, and cost per turn runs $0.00125 (Haiku) to $0.00640 (Opus 5
  at effort high). Run `python -m btcs.cli --estimate` before any live run; it prices the real
  prompt-assembly path with zero API calls.

**Do not parallelise organisms within a tick** without first deciding what happens to turn order. The
service enforces one open turn at a time deliberately; concurrent turns would make node contention
racy and would destroy even mock-brain reproducibility.

### 10.2b Running on local models *(added v0.6)*

**What a local run is good for, and what it is not.** It is a real model, choosing real tools, and
generating real token counts that become real entropy. It is *not* the same experiment as a paid run:
different model, no provider cache, and roughly 50× slower per turn. The working rule (OPEN-13) is
**mechanism travels, magnitude does not** — "organisms that never dissipate die" is a claim a local
run can support; "the median lifespan is 41 ticks" is not.

**The trap that will recur: the world was calibrated for a cheap thinker.** The mock brain "thinks" in
~100 tokens per turn. A real model reads ~1,400 — and most of that is the laws block and tool schemas
it re-reads verbatim every single tick. Charged at full rate under L3, that is 14× the disorder the
physics was tuned for, and the first local run killed every organism inside three ticks. **The fix was
not to soften a constant.** It was noticing that the entropy discount had always been about
information stability rather than billing (D-11, D-13). Expect this shape of problem again: whenever
a change makes organisms die immediately, ask what the accounting is now counting that it should not,
*before* reaching for a config knob.

**Practical notes:**

- Ollama has no prompt cache and reports every token as fresh input. `local_simulate_cache` supplies
  the equivalent by fingerprinting the stable prefix; it is on by default and turning it off is a
  legitimate experiment, not a bug.
- Small models omit required tool arguments constantly. `local_repair_arguments` fills a default
  rather than voiding the turn — deliberately, since a vague intention is still a choice.
- Local runs are billed in virtual dollars (D-12). **Any report of a "$" figure must say which brain
  produced it**, or a free run will eventually be quoted as a real cost.
- `--brain local --ticks 20 --population 3` is about five minutes. Budget for that; local runs are
  slow enough that a 200-tick sweep is an overnight job, not an afternoon one.
- Reproducibility is scoped to `brain=mock` (D-10) and that includes local. Sampling is sampling.

### 10.2c The finding from the first real-model run *(v0.6 — read this before Phase 5)*

A 3B local model, three organisms, twenty ticks, matched seed against the mock control:

| | mock control | `qwen2.5:3b` |
|---|---|---|
| Survivors at tick 20 | 3 / 3 | **0 — extinct at tick 9** |
| Lifespans (ticks) | — | 6, 8, 9 |
| Energy, start → end | 450 → 1,162 J | 450 → 0 |
| Successful ingests | 29 | **0** |
| Dissipations | 10 | **0** |
| Moves | 21 | 1 |
| Actions refused | 0 | **19 of 23 turns** ("no node here") |
| Entropy per turn | 101 tokens | 571 tokens |

Every death was `entropy_overwhelmed`. The organisms did not starve — they cooked, thinking hard about
a meal that was not there. **This is a result, not a bug**, and it is the first thing the project has
measured that a hand-written rule could not have told us.

Three readings of it, and they are testable against each other:

1. **The model is bad at this.** Plausible at 3B, and the cheap test is a larger local model.
2. **The prompt does not make the situation legible.** `on_node: false` is one field among many, and
   the model may simply not be reading it.
3. **The organism has no feedback loop (OPEN-15).** It is refused, learns nothing, and repeats the
   mistake next tick. This is the reading to take seriously, because it is a gap in the *design*
   rather than in the model: an organism that cannot perceive the consequences of its own actions
   cannot learn within its lifetime, and Phase 5 would then be selecting only over genome prose while
   every individual stays as blind as it was born.

**Do not fix this by softening the physics.** The physics is behaving correctly; something that thinks
expensively and acts uselessly should die. Fix perception, or change the model, and measure again.

**A methodological note worth keeping.** The same run was performed twice, before and after the
accounting correction described in §10.2b. The correction raised entropy per turn from 367 to 571 and
pulled extinction in from tick 17 to tick 9 — but the *behaviour was identical*: zero meals, one move,
the same refusal loop. The subsidy was hiding how badly the organisms were doing; it was not the
reason they were doing badly. When a fix changes the numbers and leaves the behaviour alone, it has
told you which of the two was the artefact.

### 10.3 Phase 4 — Φ *(built in v0.9; the result was negative)*

**Read `docs/phi.md` before touching any of this.** It carries the measurements, and the conclusion is
that integration does not pay for itself: selection favours higher Φ in 0 of 5 seeds at the shipped
price. Three things follow for anyone continuing.

**Do not quietly re-tune until the result goes away.** It is a real finding under a stated
operationalisation, and the four things it depends on are named in `docs/phi.md` §6. Change one
deliberately, re-measure, and say which — do not drift the constants until the table looks better.

**The two experiments that could legitimately overturn it:**

- **Prediction instead of protection (OPEN-18).** Error correction was defensible but it is not the
  best-motivated benefit. Still et al. (2012) prove a system that models its environment badly
  *necessarily* dissipates more energy — so integration would pay in joules, which is both closer to
  this project's premise and closer to real physics. Have an organism record a prediction, measure the
  error, and charge for it.
- **A world worth having a map of (OPEN-19).** A good map is worth little where the right answer
  rarely changes. Phase 6's scenario controls are the apparatus, which is why they were promoted ahead
  of the dashboard.

**Traps already paid for:**

- **Making Φ dynamic changes the global cost of thought.** g(Φ) was 2.64 at the old static 1.0 and is
  ~1.75 at measured values, so thinking got a third cheaper and the project's premise stopped holding
  in 2 of 3 seeds. If you change how Φ is computed, **re-check the selection signal immediately** — it
  is the load-bearing measurement and it moves silently.
- **A tier no organism reaches is a routing rule that never fires**, which is worse than no rule
  because it looks like one. Thresholds are measured from the observed distribution; a test fails if
  the top tier rises above anything ever seen.
- **Never draw from a seeded stream while iterating a `set`.** Hash order is randomised per process, so
  reproducibility dies silently and intermittently. It cost a confusing hour here.
- Φ recomputation stays rate-limited and hysteretic. Tiers swap models, prompt caches are model-scoped,
  and an organism wobbling across a boundary discards the cached prefix its whole lineage shares.

### 10.4 Phase 5 — Selection *(built in v0.8; what to know before changing it)*

**The lesson worth carrying forward.** Two bugs here were found by *running* the thing, not by
reasoning about it, and both looked like ecology until the numbers were read:

- Extinction at tick 437 **with 8,346 units of food still in the world**. Dying of thirst in a lake is
  the signature of an accounting error, never of scarcity. It was the entropy ratchet (D-16).
- 38 turns spent asking for a child the organism could not afford. Heritable thresholds drift below
  what the world accepts, and a policy that does not know the world's minimum will starve politely.

When a population dies, **check what the world still holds before touching a constant.**

**Still true and still unresolved:**

- The ancestor genome says *"Prioritise resource sharing"* while no sharing tool exists. Every organism
  is born instructed to cooperate in a world with no mechanism for it. Either add the tool — costly
  signalling is thematically perfect, since a message costs tokens and tokens are disorder, making
  honest communication literally expensive — or change the genome. It currently wastes prompt tokens
  on an instruction that cannot be obeyed, and with lineages in place kin selection is now reachable.
- Selection favours *shorter-lived, dirtier* organisms (OPEN-17), because fecundity beats longevity
  under the current prices. That may be the interesting result or a sign that replication is
  underpriced against erasure. Deciding which needs Phase 6's analysis tooling, not another guess.
- A selection differential is not a response to selection (OPEN-16). Do not let the v0.8 table become
  a claim it does not support.

**Things that will break if you are careless:**

- Mutation must keep drawing from the seeded RNG, or reproducibility on `brain=mock` dies with it.
- Mutation must stay *closed over the ancestor's vocabulary* (D-17). The moment a mutation operator can
  introduce a new word, S2 reopens and evolved text becomes an injection surface. A test guards this
  over 200 generations; do not weaken it to make prose evolve faster.
- Birth files two ledger entries that sum to zero. If a future change makes birth create or destroy
  entropy, file it explicitly — the audit will catch the omission, but only inside a bracketed tick.

### 10.5 Phase 5b — Executable genomes *(only if it earns its place)*

- `selection.executable_genomes: true` **deliberately fails config validation today**. Removing that
  guard is step one, and it must not happen before the sandbox exists.
- Charge wall-clock execution time as metabolic energy. That is what makes "unoptimized code wastes
  compute" literally true rather than thematic.
- Mutated code is untrusted by construction: no network, capped memory and CPU, hard wall-clock kill.

### 10.6 Phases 6–7

- D-4 stands: analysis and replay from JSONL first; the live panel last, built on metrics that real
  runs proved worth watching.
- **T7.7 is not optional.** Φ-tiered routing (D-2) confounds a Φ effect with a model-capability effect;
  the fixed-model control run is what lets the headline result claim anything at all.
- The mock brain is the permanent control condition, not scaffolding. Every experiment reports it.

### 10.7 Known sharp edges

| Edge | Why it bites |
|---|---|
| `ctx` must be annotated `Context` in MCP tools | The SDK decides model-facing status by annotation; an `Any` context silently becomes a tool argument a model can set. It leaked into all seven schemas before this was caught; a test guards it now. |
| Identity resolution belongs *inside* the error guard | Evaluated as an argument, a missing credential escapes as a protocol error instead of the structured refusal an organism is meant to receive. |
| `mcp` 2.x renamed `FastMCP` to `MCPServer` | The plan's boilerplate does not import at all. |
| `thermal_diffusion` must stay at or below 0.24 | Explicit 2D diffusion goes numerically unstable at 0.25. Validated at load. |
| Toroidal antisymmetry fails at exactly half the grid | Both directions are equally short and the tie-break picks one. Correct behaviour, not a bug. |
| Reproducible tokens are not secrets | A deliberate trade of secrecy for reproducibility. Correct for loopback, wrong for a network. |
