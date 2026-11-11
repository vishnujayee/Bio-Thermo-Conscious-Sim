"""Typed configuration for a simulation run.

Every physics constant in the project lives here and nowhere else. If a number
appears in an engine module, that is a bug -- experiments must be reproducible
from a config file plus a seed, and a hardcoded constant is invisible to both.

Loading is strict: an unknown key in the YAML raises rather than being ignored,
so a typo in an experiment config fails loudly instead of silently running the
default value for eight hours.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is malformed, or names a field that does not exist."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    name: str = "default"
    seed: int = 42
    max_ticks: int = 500
    output_dir: str = "runs"
    # Stop early once the population is extinct rather than spinning to max_ticks.
    stop_on_extinction: bool = True


@dataclass(frozen=True, slots=True)
class WorldConfig:
    width: int = 48
    height: int = 48

    # The thermal floor. Free energy is only extractable relative to this, so as
    # nodes cool toward ambient the world genuinely runs down (see physics.L2).
    ambient_temperature: float = 1.0

    node_count: int = 90
    node_energy_mean: float = 200.0
    node_energy_sigma: float = 30.0
    node_temperature_mean: float = 4.0
    node_temperature_sigma: float = 1.0

    # Fraction of the gap to ambient a node closes per tick. This is the master
    # clock of scarcity: raise it and the world dies young.
    node_cooling_rate: float = 0.002

    # Fraction of a node's remaining energy that leaks to the thermal field per
    # tick even when nobody eats it.
    node_leak_rate: float = 0.005

    # Lateral smoothing of the temperature field, per tick. Explicit 2D
    # diffusion is unstable above 0.25; validate() enforces the margin.
    thermal_diffusion: float = 0.15

    # Fraction of the gap to ambient each cell closes per tick by radiating to
    # the external sink. This is what makes the world an *open* system: energy
    # enters via node respawn and leaves through this coupling.
    ambient_coupling: float = 0.08

    # Energy stored per unit temperature in one cell. Converts the temperature
    # field into an energy field for the ledger.
    cell_heat_capacity: float = 250.0

    # A node below this energy is spent and gets removed.
    node_exhaustion_threshold: float = 0.5

    # Chebyshev radius an agent can sense.
    sense_radius: int = 6

    # Probability per tick that a fresh node spawns somewhere. Set to 0 for a
    # strictly closed, terminal world.
    node_respawn_rate: float = 0.6


@dataclass(frozen=True, slots=True)
class PhysicsConfig:
    # Boltzmann constant in simulation units. Keep at 1.0 unless you want energy
    # and entropy to live on different scales.
    k_boltzmann: float = 1.0

    # --- L1: Landauer cost of forgetting -------------------------------------
    landauer_enabled: bool = True
    # Thermodynamic efficiency of erasure. Must be in (0, 1]; 1.0 is the ideal
    # reversible limit, which no real machine reaches.
    erase_efficiency: float = 0.6

    # --- L2: Carnot ceiling on ingestion -------------------------------------
    carnot_cap_enabled: bool = True
    # Practical efficiency applied *below* the Carnot bound. Real engines never
    # reach the bound; this is the gap between ideal and actual.
    ingest_efficiency: float = 0.85
    # Maximum energy an agent may draw from one node in one tick.
    ingest_rate_limit: float = 60.0

    # --- L3: cognition produces entropy --------------------------------------
    # Bits of internal entropy generated per token processed (sigma).
    #
    # Raised from 0.010 in v0.9, and the reason is worth keeping. The world was
    # calibrated with Phi pinned at 1.0, where g(Phi) = 2^1.4 = 2.64. Measured
    # Phi sits near 0.5, where g = 1.75 -- so making Phi dynamic quietly made
    # every thought a third cheaper in disorder, and the selection signal that
    # is the project's whole premise held in only 1 of 3 seeds. 0.015 restores
    # it in 3 of 3 and puts the mean population back where it was. Note that
    # 0.015 / 0.010 = 1.5, against a predicted 2.64 / 1.75 = 1.51: the constant
    # was derived from the cause, not fitted to the symptom.
    entropy_per_token: float = 0.015
    # Energy drawn per token processed. Thinking costs power as well as order:
    # this becomes waste heat in the organism's own cell, which then degrades
    # its own extraction efficiency via the Carnot bound.
    energy_per_token: float = 0.015
    # g(Phi) = (1 + Phi) ** phi_entropy_exponent. Superlinear, so higher Phi
    # costs disproportionately more -- the exponential overhead the plan calls for.
    phi_entropy_exponent: float = 1.4

    # --- L4: second law ------------------------------------------------------
    # When true, the engine asserts dS_total >= 0 every tick. Leave on.
    assert_second_law: bool = True
    # Numerical slack for that assertion, to absorb float error.
    second_law_tolerance: float = 1e-9

    # --- L5: work ------------------------------------------------------------
    work_cost_per_step: float = 1.5
    # Fraction of work energy that becomes waste heat in the agent's own cell.
    # Crowds warm their neighbourhood and destroy their own gradients.
    waste_heat_fraction: float = 0.7

    # Energy drawn per tick simply by existing.
    basal_metabolic_cost: float = 0.4
    # Entropy accrued per tick simply by existing.
    basal_entropy_gain: float = 0.15

    # Temperature floor. Temperature divides in both the Carnot bound and the
    # heat-to-entropy conversion, so it may never reach zero.
    min_temperature: float = 0.05


@dataclass(frozen=True, slots=True)
class PolicyParams:
    """Numeric strategy genome, read by the deterministic mock brain.

    The LLM brain evolves prose; the mock brain evolves these numbers. Keeping
    both means selection pressure can be tested end-to-end at zero API cost --
    a wasteful parameter set must reliably lose to a frugal one before we spend
    a cent finding out whether a wasteful *prompt* does.
    """

    # Bits of internal entropy that trigger dissipation.
    dissipate_at: float = 45.0
    # Fraction of the excess above `dissipate_at` to erase in one act.
    dissipate_fraction: float = 0.5
    # Energy below which foraging outranks everything else.
    hunger_at: float = 90.0
    # Weight on a node's extractable energy when choosing a target.
    greed: float = 1.0
    # Weight against moving into hot cells (heat raises the death threshold).
    heat_aversion: float = 1.0
    # Probability of a random exploratory step when nothing is visible.
    explore_bias: float = 0.2
    # Energy at which this lineage chooses to spend itself on a child. The
    # single most consequential heritable number in the project: too eager and
    # an organism impoverishes itself into an early death, too reluctant and it
    # is simply outbred by lineages that were not. Selection decides, not us.
    replicate_at: float = 240.0


@dataclass(frozen=True, slots=True)
class AgentConfig:
    initial_population: int = 12
    initial_energy: float = 150.0
    initial_entropy: float = 10.0
    # Founders begin with no experience, so their integration genuinely is zero
    # -- which is what the measure returns for an empty graph anyway. The old
    # 1.0 was a placeholder from before Phi was computed from anything, and it
    # started every founder one tier up the model ladder.
    initial_phi: float = 0.0

    # Degradation ladder thresholds, in bits (see construction.md section 3.3).
    s_warn: float = 40.0
    s_confuse: float = 60.0
    s_dement: float = 80.0

    # Severity knobs for each rung of the ladder.
    sense_noise_scale: float = 0.25
    edge_prune_probability: float = 0.08
    dementia_injection_tokens: int = 24

    max_tool_calls_per_tick: int = 6
    # Ceiling on bits erased in one dissipation act.
    max_dissipate_bits: float = 30.0
    # Most nodes an organism can hold in perception at once. Also caps the
    # prompt size once the LLM brain arrives.
    vision_node_limit: int = 5

    policy: PolicyParams = field(default_factory=PolicyParams)


@dataclass(frozen=True, slots=True)
class ScenarioEventConfig:
    """One intervention in a scenario, as it appears in a config file."""

    at: int = 0
    # Dotted config paths to new values, e.g. {"world.ambient_temperature": 1.6}.
    set: dict[str, Any] = field(default_factory=dict)
    # A world-state action by name; see sim/scenario.py ACTIONS.
    do: str | None = None
    note: str = ""


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """How the world changes under the organisms while they live in it.

    Every run before v0.10 was in a world whose rules were fixed at tick zero,
    and that turned out to matter: Phase 4 found integration does not pay for
    itself, and the strongest remaining explanation is that a good map is worth
    little where the right answer never changes. `seasons` is the scenario built
    to test exactly that.
    """

    # A preset name, or a label for explicit `events`.
    name: str = "none"
    events: list[ScenarioEventConfig] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PhiConfig:
    """The integration index, and what it costs and buys.

    Phi was a static number until v0.9: it raised the price of thought and chose
    the model, and did nothing else. A trait that is pure cost gets selected to
    zero, which would have answered this project's central question by default
    rather than by experiment -- so `resilience` below is the answer to OPEN-11,
    and the point of the whole phase is that it is a *knob*, not a conclusion.
    """

    enabled: bool = True

    # Episodes of experience the measure looks back over. Short windows make Phi
    # twitchy; long ones make it a fossil of a life the organism no longer lives.
    window: int = 24
    # Ticks between recomputations. The eigen-decomposition is the most
    # expensive thing in a tick, and a Phi that moves every tick makes an
    # organism's model tier oscillate -- throwing away its cached prompt prefix
    # every time it wobbles across a boundary.
    recompute_every: int = 4
    # Phi must clear a tier boundary by this much before the tier actually
    # changes. Hysteresis, for the same cache reason (OPEN-8).
    tier_margin: float = 0.25
    # Ceiling, so one pathological graph cannot route an organism to the most
    # expensive model in the table forever.
    maximum: float = 8.0

    # --- what Phi buys (OPEN-11) --------------------------------------------
    # How strongly integration protects perception and memory from disorder.
    # Damage is scaled by 1 / (1 + resilience * Phi), so an organism at Phi = 2
    # with resilience 0.5 takes half the corruption of one at Phi = 0.
    #
    # The trade-off this creates is the experiment: high Phi cooks an organism
    # faster (L3 is superlinear in Phi) while keeping its map usable longer. Set
    # `resilience` to 0.0 to restore the pure-cost regime and watch selection
    # drive Phi to the floor.
    # 0.25 from a five-seed sweep over {0, 0.25, 0.5, 1, 2}, where it gave the
    # best mean population and lifespan -- though the differences were small and
    # no setting made integration pay for itself. See docs/phi.md.
    resilience: float = 0.25

    # --- the other thing Phi could buy (OPEN-18) -----------------------------
    # Bits of internal entropy charged per bit of surprise: how badly the
    # organism's own memory predicted the moment it just lived through.
    #
    # Still, Sivak, Bell & Crooks (2012) prove that a system which models its
    # environment badly *necessarily* dissipates more energy -- the part of its
    # memory that is not predictive of the future has to be paid for
    # thermodynamically. That is this project's premise, already proven in the
    # literature, and it makes integration pay in joules rather than in
    # protection. Error correction was tried first and measured as insufficient.
    #
    # Zero by default: this is an experiment, and turning it on changes the
    # metabolism of every organism. See docs/phi.md.
    prediction_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class SelectionConfig:
    replication_energy_threshold: float = 220.0
    replication_cost: float = 90.0
    # Child starts with this fraction of the parent's post-cost energy.
    # Calibrated, not chosen: at 0.4 the newborn is too poor to pay the Landauer
    # bill on the disorder it inherits, and lineages run down within a few
    # generations while the world is still full of food.
    child_energy_fraction: float = 0.5
    # The share of the parent's accumulated disorder that goes to the child.
    # This is a *division*, not a copy: the parent keeps the remainder, and the
    # population's total entropy is unchanged by the split itself. Creating the
    # child's share instead ratchets disorder upward every birth and drives
    # every lineage extinct in a world still full of food -- which is what the
    # first implementation did.
    #
    # This is the epigenetic channel the plan asks for, and it cuts both ways:
    # reproducing sheds disorder onto the next generation, so a lineage can stay
    # viable by loading damage onto its children -- and is selected against
    # through those children when it does. Set to 0.0 and the child is born
    # clean while the parent keeps everything.
    child_entropy_fraction: float = 0.15
    # Fraction of the genome text that drifts per replication (plan says 5-10%).
    mutation_rate: float = 0.07
    # Hard cap on evolved genome text. Mutation can duplicate words, so without
    # a ceiling a lineage can grow its own prompt without limit -- which is a
    # budget problem before it is an evolutionary one.
    genome_max_chars: int = 600
    # Strategy prose the founding organism carries.
    ancestor_strategy: str = "Prioritise resource sharing."
    carrying_capacity: int = 40
    # Genomes are prompt text for now. Executable policy code is Phase 5b,
    # gated behind this flag because it requires the Docker/WASM sandbox.
    executable_genomes: bool = False


@dataclass(frozen=True, slots=True)
class PhiTier:
    """One band of the Phi -> capability/cost mapping."""

    min_phi: float
    # The Claude model this tier names. With brain="local" it is not called; it
    # still sets the *virtual* price, so the economics of consciousness stay
    # identical whether or not real money is involved.
    model: str
    effort: str | None = None
    max_context_items: int = 16
    entropy_multiplier: float = 1.0
    # The Ollama model actually run for this tier. Leave null to use
    # cognition.local_model for every tier; set it (3b / 7b / 14b) when you have
    # several sizes and want bigger minds to genuinely cost more compute.
    local_model: str | None = None


def _default_tiers() -> list[PhiTier]:
    """Bands of the Phi -> capability mapping.

    The thresholds were *measured*, not chosen. While Phi was a static 1.0 they
    were round numbers (0 / 2 / 5 / 8) and no organism ever reached the second
    one. With Phi computed from experience the observed distribution is median
    0.38, 90th percentile 1.36, 95th 1.93, maximum 3.61 -- so these boundaries
    put roughly the top quarter of organisms above the first tier, the top tenth
    above the second, and only a handful in the peak band.

    Re-measure them if the world's constants change. A tier no organism reaches
    is a routing rule that never fires, which is worse than no rule at all,
    because it looks like one.
    """
    return [
        PhiTier(min_phi=0.0, model="claude-haiku-4-5", effort=None,
                max_context_items=8, entropy_multiplier=1.0),
        PhiTier(min_phi=0.7, model="claude-sonnet-5", effort="low",
                max_context_items=16, entropy_multiplier=1.8),
        PhiTier(min_phi=1.4, model="claude-opus-5", effort="medium",
                max_context_items=32, entropy_multiplier=3.2),
        PhiTier(min_phi=2.5, model="claude-opus-5", effort="high",
                max_context_items=64, entropy_multiplier=6.0),
    ]


@dataclass(frozen=True, slots=True)
class CognitionConfig:
    # "mock" is a deterministic rule-based policy that costs nothing. The whole
    # simulation must be tunable and testable on it before "llm" is ever used.
    brain: str = "mock"

    # Phi-tiered routing: consciousness costs more, in-sim and on the invoice.
    phi_tiers: list[PhiTier] = field(default_factory=_default_tiers)

    max_output_tokens: int = 1024

    # --- spend ---------------------------------------------------------------
    # Hard ceiling on real spend for one run, in USD. The runner halts and
    # checkpoints when this is crossed.
    budget_usd: float = 10.0
    # Halt as soon as the ceiling is crossed. Set false to keep running on the
    # mock brain after the money runs out, which is occasionally what you want
    # for a long tail-end observation.
    halt_on_budget: bool = True

    # --- how a turn is spent -------------------------------------------------
    # Round-trips the model may take within one tick. The percept is already in
    # the prompt, so a decisive organism needs exactly one request; each extra
    # round-trip is a real API call whose tokens become real entropy. Curiosity
    # is therefore metabolically expensive, which is the intended physics.
    max_tool_rounds: int = 3

    # --- token-to-entropy weighting (OPEN-10) --------------------------------
    # Not every token is equally costly to think. Cache reads are ~10% of the
    # price of fresh input, and there is a real argument that re-reading a
    # cached prefix is less cognitive work than reading it afresh. These weights
    # are the knob for that argument; set cache_read to 1.0 to treat all input
    # as equally taxing and see whether cache-stable lineages stop winning.
    cache_read_entropy_weight: float = 0.1
    cache_write_entropy_weight: float = 1.0
    input_entropy_weight: float = 1.0
    # Generated tokens are the organism's own output -- thought made manifest.
    # Raise above 1.0 to make verbosity itself lethal.
    output_entropy_weight: float = 1.0

    # --- structure -----------------------------------------------------------
    # Route the turn through the explicit LangGraph Sense/Integrate/Reason/Act
    # cycle. Off by default because it is pure structure: a test asserts both
    # paths produce the same decision, so if it ever changes an outcome, one of
    # the two is wrong. Turn it on for checkpointed pause/resume of long runs.
    use_graph: bool = False

    # --- local models (brain="local") ----------------------------------------
    local_base_url: str = "http://localhost:11434"
    # Fallback for any tier that does not name its own local_model.
    local_model: str = "qwen2.5:3b"
    # Generous: a cold model can spend half a minute loading before it answers.
    local_timeout_seconds: float = 600.0
    # Small models routinely omit required tool arguments. Filling a sensible
    # default is kinder -- and more interesting -- than voiding the turn, since
    # a vague intention still expresses a choice.
    local_repair_arguments: bool = True

    # Ollama has no prompt cache, so it reports every token as fresh input --
    # and an organism re-reading the identical laws block each tick would be
    # charged full entropy for information it already holds. The discount in
    # D-9 was never really about billing; it is about *information stability*.
    # Re-reading an unchanged prefix is not new cognitive work whether or not a
    # provider metered it. With this on, the stable prefix is counted as a cache
    # read after its first sighting, exactly as the API path would.
    local_simulate_cache: bool = True

    # Charge local runs at the tier's Claude price even though nothing is spent.
    # This keeps the budget guard, the cost reports and the Phi economics
    # identical between free and paid runs, so an experiment developed locally
    # transfers to the API without changing a line.
    virtual_pricing: bool = True

    # --- compute as metabolism (local runs) ----------------------------------
    # Bits of internal entropy per second of real compute. Zero by default, so
    # accounting stays token-based. Raise it and a slow thought literally cooks
    # the organism -- the closest this project gets to charging real joules,
    # since the seconds are your own hardware actually working.
    compute_seconds_entropy_weight: float = 0.0

    # --- transport -----------------------------------------------------------
    request_timeout_seconds: float = 120.0
    max_retries: int = 2
    # Log every request and response. Model sampling is not reproducible, so
    # this is what makes an LLM run auditable and replayable after the fact.
    log_model_calls: bool = True

    def tier_for(self, phi: float) -> PhiTier:
        """Return the highest tier whose threshold this Phi has reached."""
        chosen = self.phi_tiers[0]
        for tier in self.phi_tiers:
            if phi >= tier.min_phi:
                chosen = tier
        return chosen


@dataclass(frozen=True, slots=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    phi: PhiConfig = field(default_factory=PhiConfig)
    cognition: CognitionConfig = field(default_factory=CognitionConfig)

    # ---- loading ---------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Config":
        data = data or {}
        _reject_unknown(data, {f.name for f in dataclasses.fields(cls)}, "config")
        return cls(
            run=_build(RunConfig, data.get("run"), "run"),
            world=_build(WorldConfig, data.get("world"), "world"),
            physics=_build(PhysicsConfig, data.get("physics"), "physics"),
            agent=_build_agent(data.get("agent")),
            selection=_build(SelectionConfig, data.get("selection"), "selection"),
            scenario=_build_scenario(data.get("scenario")),
            phi=_build(PhiConfig, data.get("phi"), "phi"),
            cognition=_build_cognition(data.get("cognition")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form, embedded in the event log so a run is self-describing."""
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Check invariants that a type annotation cannot express."""
        problems: list[str] = []

        if self.world.width < 1 or self.world.height < 1:
            problems.append("world dimensions must be positive")
        if not 0.0 <= self.world.thermal_diffusion <= 0.24:
            problems.append(
                "world.thermal_diffusion must be in [0, 0.24]; explicit 2D diffusion "
                "goes numerically unstable at 0.25"
            )
        if not 0.0 <= self.world.ambient_coupling <= 1.0:
            problems.append("world.ambient_coupling must be in [0, 1]")
        if self.world.cell_heat_capacity <= 0.0:
            problems.append("world.cell_heat_capacity must be > 0")
        if self.world.sense_radius < 1:
            problems.append("world.sense_radius must be >= 1")
        if self.physics.min_temperature <= 0.0:
            problems.append("physics.min_temperature must be > 0 (it divides)")
        if not 0.0 <= self.physics.waste_heat_fraction <= 1.0:
            problems.append("physics.waste_heat_fraction must be in [0, 1]")
        if self.agent.initial_population < 1:
            problems.append("agent.initial_population must be >= 1")
        if not 0.0 < self.physics.erase_efficiency <= 1.0:
            problems.append("physics.erase_efficiency must be in (0, 1]")
        if not 0.0 < self.physics.ingest_efficiency <= 1.0:
            problems.append("physics.ingest_efficiency must be in (0, 1]")
        if self.world.ambient_temperature <= 0.0:
            problems.append("world.ambient_temperature must be > 0 (it divides in the Carnot bound)")
        if self.world.node_temperature_mean <= self.world.ambient_temperature:
            problems.append(
                "world.node_temperature_mean must exceed ambient_temperature, "
                "or no node ever carries extractable free energy"
            )
        if not self.agent.s_warn < self.agent.s_confuse < self.agent.s_dement:
            problems.append("agent degradation thresholds must be strictly increasing")
        if not 0.0 <= self.selection.mutation_rate <= 1.0:
            problems.append("selection.mutation_rate must be in [0, 1]")
        if not 0.0 <= self.selection.child_energy_fraction < 1.0:
            problems.append("selection.child_energy_fraction must be in [0, 1)")
        if not 0.0 <= self.selection.child_entropy_fraction <= 1.0:
            problems.append("selection.child_entropy_fraction must be in [0, 1]")
        if self.selection.replication_cost <= 0.0:
            problems.append("selection.replication_cost must be positive")
        if self.selection.replication_energy_threshold <= self.selection.replication_cost:
            # Otherwise an organism can pass the gate and still be unable to pay,
            # which is a configuration mistake that looks like a physics bug.
            problems.append(
                "selection.replication_energy_threshold must exceed replication_cost"
            )
        if self.selection.carrying_capacity < 1:
            problems.append("selection.carrying_capacity must be at least 1")
        if self.selection.genome_max_chars < 32:
            problems.append("selection.genome_max_chars must be at least 32")
        if self.phi.window < 2:
            problems.append("phi.window must be at least 2; one episode has no structure")
        if self.phi.recompute_every < 1:
            problems.append("phi.recompute_every must be at least 1")
        if self.phi.resilience < 0.0:
            problems.append("phi.resilience must not be negative")
        if self.phi.prediction_weight < 0.0:
            problems.append("phi.prediction_weight must not be negative")
        if self.phi.maximum <= 0.0:
            problems.append("phi.maximum must be positive")

        # Scenario names and actions are checked at load, not at tick 300. A run
        # that dies four minutes in because of a typo has wasted four minutes.
        from btcs.sim.scenario import ACTIONS, PRESETS

        if not self.scenario.events and self.scenario.name not in PRESETS:
            problems.append(
                f"scenario.name must be one of {', '.join(sorted(PRESETS))}, "
                f"or supply scenario.events; got {self.scenario.name!r}"
            )
        for event in self.scenario.events:
            if event.at < 1:
                problems.append("scenario.events[].at must be at least 1")
            if event.do is not None and event.do not in ACTIONS:
                problems.append(
                    f"scenario event action {event.do!r} is unknown; "
                    f"known: {', '.join(sorted(ACTIONS))}"
                )
            if not event.set and event.do is None:
                problems.append(
                    f"scenario event at tick {event.at} does nothing; "
                    "give it a `set` or a `do`"
                )
        if self.cognition.brain not in {"mock", "local", "llm"}:
            problems.append(
                f"cognition.brain must be 'mock', 'local' or 'llm', got {self.cognition.brain!r}"
            )
        if not self.cognition.phi_tiers:
            problems.append("cognition.phi_tiers must not be empty")
        elif self.cognition.phi_tiers[0].min_phi > 0.0:
            problems.append("cognition.phi_tiers must include a tier starting at min_phi 0.0")
        if self.selection.executable_genomes:
            problems.append(
                "selection.executable_genomes is Phase 5b and requires the sandbox layer, "
                "which is not implemented yet"
            )

        if problems:
            raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))


def _reject_unknown(data: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) in [{where}]: {', '.join(sorted(unknown))}. "
            f"Known keys: {', '.join(sorted(allowed))}"
        )


def _build[T](kind: type[T], data: Any, where: str) -> T:
    if data is None:
        return kind()
    if not isinstance(data, dict):
        raise ConfigError(f"[{where}] must be a mapping, got {type(data).__name__}")
    _reject_unknown(data, {f.name for f in dataclasses.fields(kind)}, where)
    return kind(**data)


def _build_agent(data: Any) -> AgentConfig:
    """Agent config needs custom handling because `policy` is a nested dataclass."""
    if data is None:
        return AgentConfig()
    if not isinstance(data, dict):
        raise ConfigError(f"[agent] must be a mapping, got {type(data).__name__}")

    data = dict(data)
    raw_policy = data.pop("policy", None)
    _reject_unknown(
        data,
        {f.name for f in dataclasses.fields(AgentConfig)} - {"policy"},
        "agent",
    )
    policy = _build(PolicyParams, raw_policy, "agent.policy")
    return AgentConfig(policy=policy, **data)


def _build_scenario(data: Any) -> ScenarioConfig:
    """Custom handling because `events` is a list of dataclasses."""
    if data is None:
        return ScenarioConfig()
    if not isinstance(data, dict):
        raise ConfigError(f"[scenario] must be a mapping, got {type(data).__name__}")

    data = dict(data)
    raw_events = data.pop("events", None) or []
    _reject_unknown(
        data,
        {f.name for f in dataclasses.fields(ScenarioConfig)} - {"events"},
        "scenario",
    )
    if not isinstance(raw_events, list):
        raise ConfigError("[scenario.events] must be a list")

    events = [_build(ScenarioEventConfig, entry, "scenario.events[]")
              for entry in raw_events]
    events.sort(key=lambda e: e.at)
    return ScenarioConfig(events=events, **data)


def _build_cognition(data: Any) -> CognitionConfig:
    """Cognition needs custom handling because phi_tiers is a list of dataclasses."""
    if data is None:
        return CognitionConfig()
    if not isinstance(data, dict):
        raise ConfigError(f"[cognition] must be a mapping, got {type(data).__name__}")

    data = dict(data)
    raw_tiers = data.pop("phi_tiers", None)
    _reject_unknown(
        data,
        {f.name for f in dataclasses.fields(CognitionConfig)} - {"phi_tiers"},
        "cognition",
    )

    if raw_tiers is None:
        tiers = _default_tiers()
    else:
        if not isinstance(raw_tiers, list):
            raise ConfigError("[cognition.phi_tiers] must be a list")
        tiers = [_build(PhiTier, entry, "cognition.phi_tiers[]") for entry in raw_tiers]
        tiers.sort(key=lambda t: t.min_phi)

    return CognitionConfig(phi_tiers=tiers, **data)
