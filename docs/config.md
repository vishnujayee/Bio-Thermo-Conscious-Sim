# Every knob, in plain language

A guide to `configs/default.yaml`. Every number that governs this world lives in
that file and nowhere else — if a constant appears inside the code, that's a bug,
because an experiment has to be reproducible from a config file plus a seed.

**A typo fails immediately.** Unknown keys are rejected when the file loads, so
`node_respwn_rate` stops the run in a second rather than silently using the
default for eight hours.

**How to read this document:** each entry says what the setting *is*, then what
actually happens when you change it. The 🔧 marks are the ones worth playing with
first. The ⚠️ marks will break the physics if you move them, which is sometimes
exactly what you want to see.

---

## `run:` — the run itself

| setting | what it is |
|---|---|
| `name` | Label for the run. Becomes the folder name under `runs/`. |
| `seed` | 🔧 The single number that determines *everything* random. Same seed → byte-identical run. Change it to get a different world with the same rules — this is how you tell a real result from a fluke. |
| `max_ticks` | How many turns to simulate. A tick is one round in which every living organism acts once. |
| `output_dir` | Where logs go. |
| `stop_on_extinction` | Stop early when everyone is dead, instead of spinning through empty ticks. |

---

## `world:` — the place they live

### Shape and climate

| setting | what it is |
|---|---|
| `width`, `height` | Grid size, 48×48 = 2,304 cells. The world **wraps around** — walk off the right edge, arrive on the left. There are no corners to get trapped in. |
| `ambient_temperature` | ⚠️ The temperature the world wants to be. **This is the floor everything is measured against**, and it's more important than it looks — see the box below. |

> **Why ambient temperature matters so much**
>
> You cannot get energy out of heat alone. You can only get energy out of a
> *difference* in heat — hot thing next to cold thing. That's the Carnot limit,
> and it's real physics, not a game rule.
>
> So `ambient_temperature` sets how much of the world's energy is **useless**.
> Raise it and food at temperature 4.0 becomes much less nourishing without
> losing a single joule of its stored energy. Raise it to 4.0 and the food is
> worthless even though it's full.
>
> This is the whole reason the `heat_wave` scenario is dangerous.

### Food (called "nodes")

| setting | what it is |
|---|---|
| `node_count` | 🔧 How many food sources exist at the start. Fewer = harsher competition. |
| `node_energy_mean` | Average energy stored in one food source. |
| `node_energy_sigma` | How much they vary. `0` would make every node identical. |
| `node_temperature_mean` | Average food temperature. **Must be above ambient** or it isn't food at all — the config refuses to load otherwise, because a world with no gradient looks like a physics bug rather than a mistake. |
| `node_temperature_sigma` | How much food temperature varies. |
| `node_cooling_rate` | 🔧 **The master clock of scarcity.** How fast food cools toward ambient each tick. At `0.002` a gradient half-life is hundreds of ticks. At `0.02` the world is thermally dead by tick 150 no matter what anyone does — that was the original bug that killed everything by tick 30. |
| `node_leak_rate` | How fast food bleeds energy into the surrounding air even when nobody eats it. |
| `node_exhaustion_threshold` | Below this energy, a node is spent and disappears. |
| `node_respawn_rate` | 🔧 Chance per tick that a new food source appears. **Set this to 0 for a closed, dying world** — that's a real experiment worth running. |

### Heat

| setting | what it is |
|---|---|
| `thermal_diffusion` | ⚠️ How fast heat spreads sideways between cells. **Hard-capped at 0.24**: the maths this uses goes numerically unstable at 0.25 and the temperature field explodes into nonsense. The config refuses higher values. |
| `ambient_coupling` | How fast each cell radiates heat away to the outside. **This is what makes the world open rather than sealed** — energy comes in via respawning food and leaves through here. Without it, all heat accumulates forever and nothing can stay cool enough to live. |
| `cell_heat_capacity` | 🔧 How much energy it takes to warm one cell by one degree. **Low values make organisms cook their own neighbourhood within a tick or two.** This single number controls how brutal crowding is. |
| `sense_radius` | How far an organism can see (6 cells in every direction, including diagonals). |

---

## `physics:` — the five laws

This section is the heart of the project. Each block is one law.

| setting | what it is |
|---|---|
| `k_boltzmann` | The constant linking energy and entropy. Kept at `1.0` so the two live on the same scale. Real physics uses a tiny number; here it just sets units. |

### L1 — forgetting costs energy (Landauer)

| setting | what it is |
|---|---|
| `landauer_enabled` | ⚠️ 🔧 **Turn this off and forgetting becomes free.** The second law breaks and the whole premise evaporates — which makes it a genuinely instructive thing to watch once. Turn off `assert_second_law` too, or the run will stop and tell you the books don't balance. |
| `erase_efficiency` | 🔧 How wasteful cleaning up is. At `0.6`, erasing 10 bits' worth of mess costs you enough energy for 16.7 bits. **It must be below 1.0** — that inefficiency is precisely what keeps the second law satisfied. Real machines never reach 1.0 either. |

> **What Landauer's principle actually says**
>
> Erasing one bit of information *must* release at least `k·T·ln2` of heat. This
> is real, proven physics from 1961, and it's been measured in laboratories. It's
> the reason a computer gets warm.
>
> In this project it means: **a creature's memory of its own mess is physical,
> and clearing it costs real energy.** Not a metaphor.

### L2 — eating is capped (Carnot)

| setting | what it is |
|---|---|
| `carnot_cap_enabled` | ⚠️ Turn off and organisms extract energy regardless of temperature. |
| `ingest_efficiency` | How close to the theoretical maximum a real meal gets. `0.85` = 85% of the physically permitted amount, since no real engine reaches its own limit. |
| `ingest_rate_limit` | 🔧 Most energy drawn from one food source in one turn. This is a "mouth size" — it stops an organism swallowing a whole node instantly. |

> **The detail that made the project work**
>
> The cold side of the Carnot calculation is **the organism's own cell**, not the
> world's average temperature. That's not a technicality — it's why crowds
> starve. Your waste heat warms the cell you're standing in, which raises your
> own cold-side temperature, which shrinks the gradient you're eating from.
>
> Get this wrong (use the global average) and organisms extract more energy than
> physics permits, and the entropy books go negative within ticks. Getting it
> right produced the crowding dynamic **for free** — nobody wrote it.

### L3 — thinking makes mess

| setting | what it is |
|---|---|
| `entropy_per_token` | 🔧 **The bridge between the two halves of this project.** Mess generated per token of thought. This is what turns "the AI used 400 tokens" into "the creature is 6 bits messier". Raise it and thinking becomes lethal; lower it and minds are cheap. |
| `energy_per_token` | Energy burned per token, released as heat in your own cell. Thinking costs power *and* makes mess. |
| `phi_entropy_exponent` | The `k` in `(1+Φ)^k`. How much more expensive it is to think in an integrated way. At `1.4` it's superlinear — doubling integration more than doubles the cost. **The measured break-even is around `k ≈ 1`**, so the shipped value is firmly in the "integration doesn't pay" regime. See `docs/phi.md`. |

### L4 — the second law

| setting | what it is |
|---|---|
| `assert_second_law` | ⚠️ Checks every tick that total disorder never decreased. **Leave this on.** If it ever fires, a result computed after that point is meaningless. |
| `second_law_tolerance` | Slack for floating-point rounding error (`0.000000001`). |

### L5 — moving costs

| setting | what it is |
|---|---|
| `work_cost_per_step` | Energy to move one cell. |
| `waste_heat_fraction` | What share of that energy becomes heat where you *started* rather than where you arrived. At `0.7` a moving organism warms its own trail behind it. |

### Just existing

| setting | what it is |
|---|---|
| `basal_metabolic_cost` | Energy burned per tick doing absolutely nothing. **You cannot rest your way out of trouble** — this is why. |
| `basal_entropy_gain` | Mess accumulated per tick doing nothing. Even a perfectly still organism slowly decays. |
| `min_temperature` | ⚠️ Floor on temperature. Temperature *divides* in two formulas, so allowing zero would divide by zero and poison the run. |

---

## `agent:` — the organisms

| setting | what it is |
|---|---|
| `initial_population` | 🔧 How many founders. |
| `initial_energy` | What each founder starts with. |
| `initial_entropy` | Starting mess. Nobody starts perfectly clean. |
| `initial_phi` | Starting integration. `0.0` — a creature with no experience has nothing to integrate. |

### The degradation ladder

Death isn't a cliff, it's a slide — so selection has something to grip.

| setting | what happens at this much mess |
|---|---|
| `s_warn` (40) | **Noisy.** Readings get fuzzy. The organism is *not told* they're fuzzy. |
| `s_confuse` (60) | **Confused.** Things vanish from perception and memory entirely. Still not told. |
| `s_dement` (80) | **Demented.** Junk fills the mind, and thinking now costs *more* than it did. This is the death spiral in one line: mess makes thinking expensive, expensive thinking makes more mess. |

| setting | what it is |
|---|---|
| `sense_noise_scale` | How badly the readings blur at "noisy". |
| `edge_prune_probability` | Chance any one remembered thing is lost at "confused". |
| `dementia_injection_tokens` | Extra tokens charged for the junk clogging a demented mind. |
| `max_tool_calls_per_tick` | Ceiling on actions per turn, so a confused organism can't thrash. |
| `max_dissipate_bits` | Most mess clearable in one act. No instant reset. |
| `vision_node_limit` | Most food sources held in mind at once. |

> **Why they're never told their senses are failing**
>
> An accurate error bar would itself be a form of the clear thinking that high
> disorder is supposed to have destroyed. A creature that knows exactly which of
> its beliefs are wrong isn't confused — it's well-informed about being confused.
> **That asymmetry is the point.**

### `policy:` — the heritable strategy 🔧

**These seven numbers are the genome.** They mutate at every birth, and they're
what selection actually acts on. Changing them is the most interesting thing you
can do in this file.

| setting | what it means |
|---|---|
| `dissipate_at` | Mess level at which this creature starts cleaning. **The trait with a measured selection signal** — creatures that bred had *higher* values, meaning selection favours cleaning later. |
| `dissipate_fraction` | How much of the excess to clear in one go. |
| `hunger_at` | Energy level below which finding food beats everything else. |
| `greed` | How strongly a rich food source attracts it. |
| `heat_aversion` | How strongly it avoids hot cells. |
| `explore_bias` | Chance of wandering randomly when nothing is visible. |
| `replicate_at` | Energy at which it spends itself on a child. **The most consequential number here** — too eager and you impoverish yourself, too reluctant and you're simply outbred. |

---

## `selection:` — birth and inheritance

| setting | what it is |
|---|---|
| `replication_energy_threshold` | The world's minimum. Below this, no child, whatever the genome wants. |
| `replication_cost` | 🔧 Energy burned *building* the child — becomes waste heat in your own cell. **Breeding densely warms the very neighbourhood your children need.** |
| `child_energy_fraction` | Share of what's left handed to the child. Calibrated by measurement: at `0.4` newborns couldn't afford to clean the mess they inherited and lineages died out within a few generations. |
| `child_entropy_fraction` | 🔧 **Share of the parent's mess handed down. A division, not a copy** — the parent keeps the rest, and the total is unchanged. An earlier version *created* the child's share out of nothing, and the resulting ratchet drove every lineage extinct by tick 437 while the world still held 8,346 units of food. Set to `0.0` for clean-slate births. |
| `mutation_rate` | 🔧 How much drift per birth. `0.07` = about 7% of words changed and 7% wobble on each number. |
| `genome_max_chars` | Cap on evolved text. Mutation can *duplicate* words, so without a ceiling a lineage grows its own prompt without limit. |
| `ancestor_strategy` | The text every founder is born with. **Currently says "Prioritise resource sharing" in a world with no way to share** — a known, deliberate gap. |
| `carrying_capacity` | Hard population ceiling. |
| `executable_genomes` | ⚠️ **Deliberately fails validation if you set it true.** Turning it on before the sandbox exists would run mutated code unsandboxed on your machine. The failure is the safety feature. |

---

## `scenario:` — making the world change

| setting | what it is |
|---|---|
| `name` | 🔧 `none`, `heat_wave`, `drought`, `gradient_collapse`, `bottleneck`, or `seasons`. |
| `events` | Your own interventions, if you don't want a preset. |

The presets, and what each is for:

| scenario | what happens |
|---|---|
| `heat_wave` | Ambient temperature rises. Every gradient shrinks; food stays full but stops being nourishing. |
| `drought` | New food almost stops appearing. |
| `gradient_collapse` | Every food source drops to ambient at once. Total, instantaneous famine amid plenty. |
| `bottleneck` | Two lean years, then recovery. Tests whether a squeezed population bounces back. |
| `seasons` | Climate swings *and the food physically moves*. Built to test whether a changing world makes a good map worth paying for. (It didn't.) |

Severities were **measured, not chosen**: a scenario that always wipes everyone
out measures extinction rather than adaptation, and one nobody notices measures
nothing. Each was tuned until survival was in doubt but not settled.

---

## `phi:` — the integration measure

Read `docs/phi.md` before trusting any number here. Nothing in this project is
conscious.

| setting | what it is |
|---|---|
| `enabled` | 🔧 Turn the whole mechanism off. |
| `window` | How many turns of experience the measure looks back over. |
| `recompute_every` | How often to recalculate. It's the most expensive computation in a tick. |
| `tier_margin` | How decisively Φ must cross a boundary before the model tier actually changes. Prevents an organism flip-flopping and throwing away its cached prompt each time. |
| `maximum` | Ceiling, so one pathological memory can't route an organism to the priciest model forever. |
| `resilience` | 🔧 **What integration buys**: protection against disorder corrupting perception and memory. Set to `0.0` for the pure-cost control — that's the experiment that showed selection drives Φ to the floor. |
| `prediction_weight` | 🔧 Mess charged for being *surprised* by your own world. **Ships at `0.0` on purpose.** Turning it up produces the project's only positive Φ result — which turned out to be a confound. See `docs/phi.md` §6. |

---

## `cognition:` — the mind

| setting | what it is |
|---|---|
| `brain` | 🔧 `mock` (free, instant, a hand-written rule), `local` (a real model via Ollama, free, slow), or `llm` (Claude, costs money). **The world can't tell which is plugged in.** |
| `max_output_tokens` | Ceiling on one reply. |
| `budget_usd` | Hard spending limit for one run. Crossing it halts cleanly. |
| `halt_on_budget` | Whether to stop or keep going once the money runs out. |
| `max_tool_rounds` | Round-trips allowed within one turn. The situation is already in the prompt, so a decisive organism needs exactly one. **Every extra round is real tokens and therefore real mess — curiosity is metabolically expensive by construction.** |

### Token weighting

| setting | what it is |
|---|---|
| `cache_read_entropy_weight` | 🔧 Mess charged for re-reading something unchanged, at 1/10 rate. **The principle: re-reading an unchanged block is not new mental work**, whether or not a provider billed for it. Set to `1.0` to switch that pressure off and see whether it was doing anything. |
| `cache_write_entropy_weight` | First time reading a block. |
| `input_entropy_weight` | Genuinely new information. |
| `output_entropy_weight` | The organism's own words. **Raise above 1.0 to make verbosity itself lethal.** |

### Machinery

| setting | what it is |
|---|---|
| `use_graph` | Route turns through the explicit Sense→Integrate→Reason→Act cycle. Pure structure — a test asserts both paths make the same decision. |
| `request_timeout_seconds`, `max_retries` | Network patience. |
| `log_model_calls` | Log every request and reply. Model sampling isn't reproducible, so **this is what makes an AI run auditable after the fact** even though it can't be re-run identically. |

### `phi_tiers:` — consciousness costs more

As integration rises, an organism is routed to a bigger model *and* charged more
mess per token. The metaphor and the invoice agree.

| field | what it is |
|---|---|
| `min_phi` | Integration needed to reach this tier. **Thresholds were measured from the observed distribution, not chosen** — a tier nothing ever reaches is a rule that never fires, which is worse than no rule because it looks like one. |
| `model` | Which Claude model. Also sets the *virtual* price on free local runs. |
| `effort` | How hard the model is asked to think. |
| `max_context_items` | How much memory it may carry. |
| `entropy_multiplier` | Extra mess multiplier for this tier. |

---

## Five experiments worth running

Predict the answer out loud **before** you run each one. Being wrong is the point.

```bash
# 1. Make thinking nearly free.   physics.entropy_per_token: 0.001
#    Does the population explode, or does something else break?

# 2. Turn off Landauer.           physics.landauer_enabled: false
#                                 physics.assert_second_law: false
#    Forgetting is now free. Is the world still interesting?

# 3. Seal the world.              world.node_respawn_rate: 0.0
#    No new food ever. How long until the last organism dies?

# 4. Make crowding brutal.        world.cell_heat_capacity: 50.0
#    Cells heat up 5x faster. Do they spread out, or all die?

# 5. Stop inheriting damage.      selection.child_entropy_fraction: 0.0
#    Children born clean. Do lineages last longer?
```

After each: `python -m btcs.report runs/<name>_seed42/events.jsonl` and look at
the death causes and the refusal panel first.

**And always try more than one seed.** One run is an anecdote.
