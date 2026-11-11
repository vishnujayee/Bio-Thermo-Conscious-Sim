"""The five laws.

This module is the only place in the project permitted to change an organism's
energy or entropy. Every such change is paired with an entry in the tick's
`EntropyLedger`, which is what makes the second-law assertion in the runner a
real check rather than a decoration.

The laws, as specified in construction.md section 3.2:

    L1  Erasing n bits costs n * k_B * T * ln2 / eta_erase joules.
    L2  Ingestion is capped by the Carnot bound on the node/cell gradient.
    L3  Cognition converts tokens into internal entropy, superlinearly in Phi.
    L4  dS_total >= 0 every tick. Enforced by the runner against the ledger.
    L5  Work costs energy and deposits waste heat where the work happened.

A note on how the second law is guaranteed rather than hoped for. Each
entropy-reducing process is written so that its own accounting covers the
deficit it creates:

  * dissipation reduces an organism by n bits but dumps n/eta_erase bits into
    the environment, so it nets positive exactly because eta_erase < 1;
  * ingestion rejects waste heat into the same reservoir whose temperature set
    the efficiency, so the Carnot case nets exactly zero and every real case
    nets positive;
  * erasure at death moves an organism's bits to the environment one-for-one.

There is no process that can drive the total negative, which is why the runner's
assertion has never had to be relaxed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from btcs.config import Config
from btcs.events import EventLog, EventType
from btcs.rng import RngService
from btcs.world.agents import Agent, AgentRegistry, Degradation, mutate
from btcs.world.grid import World
from btcs.world.memory import MemoryGraph
from btcs.world.ledger import EntropyLedger
from btcs.world.types import (
    Action,
    Dissipate,
    Ingest,
    Move,
    Percept,
    Replicate,
    Rest,
)

LN2 = math.log(2.0)


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """What actually happened, which is often not what was asked for."""

    kind: str
    ok: bool
    detail: str = ""
    energy_delta: float = 0.0
    entropy_delta: float = 0.0


class PhysicsEngine:
    """Applies the laws to organisms in a world."""

    __slots__ = ("_config", "_world", "_registry", "_rng", "_log")

    def __init__(
        self,
        config: Config,
        world: World,
        registry: AgentRegistry,
        rng: RngService,
        log: EventLog,
    ) -> None:
        # The world owns some physics of its own (the Carnot efficiency, the
        # thermal field), so it carries a config reference too. If the two ever
        # diverge, the engine charges one price while the world applies another
        # and the entropy ledger silently stops balancing. Catching it here
        # turns a subtle physics bug into an immediate, obvious failure.
        if world.config is not config:
            raise ValueError(
                "PhysicsEngine and World must share one Config instance; "
                "build the World from the same config object"
            )
        self._config = config
        self._world = world
        self._registry = registry
        self._rng = rng
        self._log = log

    # -- helpers ----------------------------------------------------------

    def local_temperature(self, agent: Agent) -> float:
        return max(
            self._config.physics.min_temperature,
            self._world.temperature_at(agent.y, agent.x),
        )

    def _spend(
        self,
        agent: Agent,
        joules: float,
        ledger: EntropyLedger,
        process: str,
        *,
        at: tuple[int, int] | None = None,
    ) -> float:
        """Take energy from an organism and release it as heat.

        Spending is clamped to what the organism actually has -- energy that does
        not exist cannot become heat, and letting the balance go negative would
        put entropy into the environment that nothing paid for. An organism that
        cannot afford an act simply performs less of it, and usually dies shortly
        afterwards for unrelated reasons.
        """
        spent = min(max(0.0, joules), max(0.0, agent.energy))
        if spent <= 0.0:
            return 0.0
        agent.energy -= spent
        y, x = at if at is not None else (agent.y, agent.x)
        self._world.deposit_heat(y, x, spent, ledger, process)
        return spent

    def landauer_cost(self, bits: float, temperature: float) -> float:
        """L1: the energy price of erasing `bits`.

        With `landauer_enabled` off this returns zero, which makes forgetting
        free and breaks the second law -- so the flag exists only for debugging
        and `assert_second_law` should be off whenever it is used.
        """
        physics = self._config.physics
        if not physics.landauer_enabled:
            return 0.0
        return (
            bits * physics.k_boltzmann * temperature * LN2 / physics.erase_efficiency
        )

    # -- L3: cognition ----------------------------------------------------

    def apply_cognition(
        self, agent: Agent, tokens: int, ledger: EntropyLedger, tick: int
    ) -> None:
        """Charge an organism for having thought.

        This is the hinge of the whole project: the token count is the same
        number the API bills for, and here it becomes physical disorder. An
        organism that reasons expensively heats up, and heat is what kills it.

        A demented organism pays a token surcharge for the junk occupying its
        context. That is the death spiral in one line -- disorder makes thinking
        more expensive, and expensive thinking makes more disorder.
        """
        physics = self._config.physics
        billed = tokens
        if agent.degradation >= Degradation.DEMENTED:
            billed += self._config.agent.dementia_injection_tokens

        g_phi = (1.0 + agent.phi) ** physics.phi_entropy_exponent
        bits = physics.entropy_per_token * billed * g_phi

        agent.entropy += bits
        agent.total_tokens += billed
        ledger.agent_entropy(bits, "cognition")

        self._spend(agent, physics.energy_per_token * billed, ledger, "cognition")

        self._log.emit(
            tick, EventType.COGNITION, agent.id,
            tokens=billed, entropy_bits=round(bits, 6), phi=round(agent.phi, 4),
        )

    # -- basal metabolism --------------------------------------------------

    def apply_basal(self, agent: Agent, ledger: EntropyLedger, tick: int) -> None:
        """The cost of merely existing."""
        physics = self._config.physics
        agent.entropy += physics.basal_entropy_gain
        ledger.agent_entropy(physics.basal_entropy_gain, "basal")
        self._spend(agent, physics.basal_metabolic_cost, ledger, "basal")

    # -- actions -----------------------------------------------------------

    def apply(
        self, agent: Agent, action: Action, ledger: EntropyLedger, tick: int
    ) -> ActionOutcome:
        match action:
            case Rest():
                return ActionOutcome("rest", True)
            case Move(dy=dy, dx=dx):
                return self._do_move(agent, dy, dx, ledger, tick)
            case Ingest(joules=joules):
                return self._do_ingest(agent, joules, ledger, tick)
            case Dissipate(bits=bits):
                return self._do_dissipate(agent, bits, ledger, tick)
            case Replicate():
                return self._do_replicate(agent, ledger, tick)
        return ActionOutcome("unknown", False, f"unhandled action: {action!r}")

    def _do_replicate(
        self, agent: Agent, ledger: EntropyLedger, tick: int
    ) -> ActionOutcome:
        """Spend energy to place a mutated copy of an organism in the world.

        Three costs, and each is a real one rather than a fee:

        * **Construction.** `replication_cost` joules are spent as work and
          become waste heat in the parent's own cell. Building a body is
          irreversible, so it warms the neighbourhood -- which means a lineage
          that breeds densely degrades the very gradients its children need.
          Nobody wrote that trade-off; it falls out of using `_spend`.
        * **Endowment.** A share of what remains is *transferred* to the child.
          Energy is conserved across that line, so this costs the parent
          everything it gives and costs the world nothing.
        * **Division of disorder.** The parent's accumulated entropy is *split*
          with the child, not copied: the child is built out of the parent, so
          the damage in it comes out of the parent too. Total organism entropy
          is unchanged across the split, which is the honest accounting -- an
          earlier version created the child's share out of nothing, and the
          resulting ratchet drove every lineage extinct within four generations
          while the world still held most of its food.

          That makes replication a way to shed disorder, which is not a loophole
          but a real and well-documented strategy: asymmetric division, where a
          lineage stays viable by loading its damage onto particular offspring.
          It is paid for honestly -- the child is less viable for it, and a
          parent that breeds filthy children is selected against through them.

        Refusals are ordinary outcomes here, not errors: an organism that cannot
        afford a child, or lives in a world already at carrying capacity, is
        told so and keeps its turn's other consequences.
        """
        selection = self._config.selection

        if len(self._registry.living()) >= selection.carrying_capacity:
            return ActionOutcome(
                "replicate", False,
                f"the world is full ({selection.carrying_capacity} living)",
            )
        if agent.energy < selection.replication_energy_threshold:
            return ActionOutcome(
                "replicate", False,
                f"needs {selection.replication_energy_threshold:.0f} energy to "
                f"replicate, has {agent.energy:.0f}",
            )

        before = agent.energy
        self._spend(agent, selection.replication_cost, ledger, "replication")
        endowment = agent.energy * selection.child_energy_fraction
        agent.energy -= endowment

        inherited_bits = agent.entropy * selection.child_entropy_fraction
        agent.entropy -= inherited_bits
        # Filed as two entries that sum to zero rather than as nothing at all,
        # so the per-process attribution shows the transfer happening and the
        # ledger audit still balances to the bit.
        ledger.agent_entropy(-inherited_bits, "replication_parent")
        ledger.agent_entropy(inherited_bits, "replication_child")

        stream = self._rng.agent_stream(agent.id)
        child = Agent(
            id=self._registry.next_id(agent.generation + 1),
            y=agent.y,
            x=agent.x,
            energy=endowment,
            entropy=inherited_bits,
            phi=agent.phi,
            genome=mutate(agent.genome, stream, self._config),
            generation=agent.generation + 1,
            parent_id=agent.id,
            birth_tick=tick,
        )
        self._registry.add(child)
        agent.offspring += 1

        self._log.emit(
            tick, EventType.REPLICATION, agent.id,
            child=child.id, generation=child.generation,
            parent_energy_before=round(before, 3),
            parent_energy_after=round(agent.energy, 3),
            child_energy=round(child.energy, 3),
            child_entropy=round(child.entropy, 4),
            mutated=child.genome.strategy != agent.genome.strategy,
            strategy=child.genome.strategy,
        )
        self._log.emit(
            tick, EventType.AGENT_BORN, child.id,
            parent=agent.id, generation=child.generation,
            energy=round(child.energy, 3), entropy=round(child.entropy, 4),
            y=child.y, x=child.x, genome=child.genome.to_dict(),
        )
        return ActionOutcome(
            "replicate", True, detail=f"child {child.id}",
            energy_delta=agent.energy - before, entropy_delta=0.0,
        )

    def _do_move(
        self, agent: Agent, dy: int, dx: int, ledger: EntropyLedger, tick: int
    ) -> ActionOutcome:
        """L5: work costs energy and leaves waste heat behind."""
        dy = max(-1, min(1, int(dy)))
        dx = max(-1, min(1, int(dx)))
        if dy == 0 and dx == 0:
            return ActionOutcome("move", False, "no displacement")

        cost = self._config.physics.work_cost_per_step
        if agent.energy < cost:
            return ActionOutcome("move", False, "insufficient energy")

        origin = (agent.y, agent.x)
        target = self._world.wrap(agent.y + dy, agent.x + dx)

        # Waste heat splits between where the work started and where it ended,
        # so a moving organism warms its own trail.
        waste = self._config.physics.waste_heat_fraction
        self._spend(agent, cost * waste, ledger, "work", at=origin)
        self._spend(agent, cost * (1.0 - waste), ledger, "work", at=target)

        agent.y, agent.x = target
        self._log.emit(tick, EventType.WORK, agent.id,
                       dy=dy, dx=dx, y=agent.y, x=agent.x, cost=round(cost, 4))
        return ActionOutcome("move", True, energy_delta=-cost)

    def _do_ingest(
        self, agent: Agent, joules: float, ledger: EntropyLedger, tick: int
    ) -> ActionOutcome:
        """L2: harvest a node, capped by the Carnot bound on the local gradient.

        The entropy accounting here is the subtle one. The node loses stock at
        its own temperature; the fraction that does not become usable work is
        rejected into the organism's cell. Because the efficiency was computed
        against that same cell temperature, the ideal case nets exactly zero
        entropy and every real case nets positive -- which is the Carnot
        theorem, arrived at by bookkeeping rather than by assertion.
        """
        index = self._world.node_index_at(agent.y, agent.x)
        if index < 0:
            return ActionOutcome("ingest", False, "no node here")

        cold = self.local_temperature(agent)
        node_temperature = float(self._world.node_temperature[index])
        efficiency = self._world.carnot_efficiency(node_temperature, cold)
        if efficiency <= 0.0:
            return ActionOutcome(
                "ingest", False,
                f"node at thermal equilibrium (T_node={node_temperature:.3f} <= T_cell={cold:.3f})",
            )

        requested = min(max(0.0, joules), self._config.physics.ingest_rate_limit)
        drawn = self._world.draw_from_node(index, requested)
        if drawn <= 0.0:
            return ActionOutcome("ingest", False, "node exhausted")

        work = drawn * efficiency
        waste = drawn - work

        # The node gives up `drawn` at its own temperature: an entropy decrease.
        ledger.environment_entropy(
            -drawn / (ledger.k_boltzmann * node_temperature * LN2), "ingest"
        )
        # The unusable remainder is rejected into the cell: an increase that,
        # by construction, at least covers it.
        self._world.deposit_heat(agent.y, agent.x, waste, ledger, "ingest")

        agent.energy += work
        agent.total_ingested += work

        self._log.emit(
            tick, EventType.INGEST, agent.id,
            drawn=round(drawn, 4), work=round(work, 4), waste=round(waste, 4),
            efficiency=round(efficiency, 4), node_temperature=round(node_temperature, 4),
        )
        return ActionOutcome("ingest", True, energy_delta=work)

    def _do_dissipate(
        self, agent: Agent, bits: float, ledger: EntropyLedger, tick: int
    ) -> ActionOutcome:
        """L1: buy back order at the Landauer price.

        Note the clamping order. The request is capped by the ceiling, by what
        the organism actually has to erase, and finally by what it can afford --
        an organism too poor to forget stays confused, which is the trap the
        whole selection story depends on.
        """
        temperature = self.local_temperature(agent)
        physics = self._config.physics

        wanted = min(max(0.0, bits), self._config.agent.max_dissipate_bits, agent.entropy)
        if wanted <= 0.0:
            return ActionOutcome("dissipate", False, "nothing to erase")

        cost = self.landauer_cost(wanted, temperature)
        if cost > agent.energy:
            # Erase only what is affordable rather than failing outright.
            affordable = agent.energy / self.landauer_cost(1.0, temperature) if cost > 0 else wanted
            wanted = max(0.0, min(wanted, affordable))
            cost = self.landauer_cost(wanted, temperature)
            if wanted <= 0.0:
                return ActionOutcome(
                    "dissipate", False,
                    f"cannot afford erasure at T={temperature:.3f} (energy={agent.energy:.2f})",
                )

        agent.entropy -= wanted
        agent.total_dissipated_bits += wanted
        ledger.agent_entropy(-wanted, "dissipate")

        # The Landauer heat. This is what makes the act net-positive in entropy:
        # cost / (k*T*ln2) == wanted / eta_erase >= wanted.
        self._spend(agent, cost, ledger, "dissipate")

        self._log.emit(
            tick, EventType.DISSIPATE, agent.id,
            bits=round(wanted, 4), cost=round(cost, 4), temperature=round(temperature, 4),
        )
        return ActionOutcome(
            "dissipate", True, energy_delta=-cost, entropy_delta=-wanted
        )

    # -- degradation and death --------------------------------------------

    def charge_surprise(
        self, agent: Agent, ledger: EntropyLedger, tick: int
    ) -> float:
        """Bill an organism for having been wrong about its own world.

        The Still et al. (2012) mechanism, and the closest this project gets to
        its own premise: a system that models its environment badly dissipates
        more, so the disorder an organism accrues depends on how surprised it
        was. Nothing here rewards a good model -- it merely stops charging for a
        bad one, which is the honest direction. Prediction is not a bonus; being
        wrong is a cost.
        """
        weight = self._config.phi.prediction_weight
        if weight <= 0.0 or agent.last_surprise <= 0.0:
            return 0.0

        bits = weight * agent.last_surprise
        agent.entropy += bits
        ledger.agent_entropy(bits, "surprise")
        self._log.emit(
            tick, EventType.SURPRISE, agent.id,
            surprise=round(agent.last_surprise, 4), entropy_bits=round(bits, 6),
            phi=round(agent.phi, 4),
        )
        agent.last_surprise = 0.0
        return bits

    def integration_shield(self, agent: Agent) -> float:
        """The factor by which integration divides incoming corruption.

        **This is the answer to OPEN-11: what Phi buys.** Until now Phi raised
        the price of thought and chose the model, and did nothing else -- and a
        trait that is pure cost is selected to zero, which would have settled the
        project's central question by default rather than by experiment.

        What it buys is *error correction*, which is what integration is for.
        A system whose parts carry information about each other can reconstruct a
        damaged part from the rest; a heap of unrelated fragments cannot. So an
        organism with an integrated memory keeps a usable map for longer under
        the same disorder.

        It is bought, not granted. L3 is superlinear in Phi, so the organism that
        thinks in an integrated way cooks itself faster while it does. Whether
        there is a Phi that repays its own heat is the experiment, and setting
        `phi.resilience` to zero restores the pure-cost regime for the control.
        """
        phi_cfg = self._config.phi
        if not phi_cfg.enabled or phi_cfg.resilience <= 0.0:
            return 1.0
        return 1.0 + phi_cfg.resilience * max(0.0, agent.phi)

    def corruption_probability(self, agent: Agent) -> float:
        """Chance that any one remembered thing is lost, after the shield."""
        return self._config.agent.edge_prune_probability / self.integration_shield(agent)

    def sense_noise(self, agent: Agent) -> float:
        """Scale of the noise on this organism's readings, after the shield."""
        return self._config.agent.sense_noise_scale / self.integration_shield(agent)

    def update_phi(self, agent: Agent, graph: MemoryGraph, tick: int) -> float:
        """Recompute integration from experience, with hysteresis on the tiers.

        Phi moves only when it has cleared a tier boundary by `tier_margin`.
        Without that, an organism sitting on a boundary swaps models every few
        ticks -- and because prompt caches are model-scoped, each swap throws
        away the cached prefix its whole lineage shares. This is the answer to
        OPEN-8: dynamic, but rate-limited and hysteretic.
        """
        phi_cfg = self._config.phi
        if not phi_cfg.enabled:
            return agent.phi

        measured = min(phi_cfg.maximum,
                       graph.phi(tick, recompute_every=phi_cfg.recompute_every))
        if abs(measured - agent.phi) < phi_cfg.tier_margin:
            return agent.phi

        previous, agent.phi = agent.phi, measured
        if abs(measured - previous) >= phi_cfg.tier_margin:
            self._log.emit(
                tick, EventType.PHI_CHANGED, agent.id,
                phi=round(measured, 4), previous=round(previous, 4),
                items=len(graph.items), episodes=len(graph.history),
            )
        return measured

    def resolve_degradation(self, agent: Agent, tick: int) -> None:
        """Place the organism on the ladder from lucid to demented."""
        cfg = self._config.agent
        if agent.entropy > cfg.s_dement:
            rung = Degradation.DEMENTED
        elif agent.entropy > cfg.s_confuse:
            rung = Degradation.CONFUSED
        elif agent.entropy > cfg.s_warn:
            rung = Degradation.NOISY
        else:
            rung = Degradation.LUCID

        if rung != agent.degradation:
            previous, agent.degradation = agent.degradation, rung
            self._log.emit(
                tick, EventType.DEGRADATION, agent.id,
                to=Degradation.NAMES[rung], **{"from": Degradation.NAMES[previous]},
                entropy=round(agent.entropy, 4),
            )

    def check_death(self, agent: Agent, ledger: EntropyLedger, tick: int) -> bool:
        """Erase an organism whose free energy has run out.

        Death releases the organism's accumulated disorder into the world
        one-for-one, so the books stay balanced across the transition: what left
        the population arrives in the environment.
        """
        temperature = self.local_temperature(agent)
        free_energy = agent.free_energy(temperature)
        if free_energy > 0.0:
            return False

        cause = "energy_exhausted" if agent.energy <= 0.0 else "entropy_overwhelmed"

        released = agent.entropy
        ledger.agent_entropy(-released, "erasure")
        ledger.environment_entropy(released, "erasure")

        # Whatever energy remains is released as heat rather than vanishing.
        remaining = max(0.0, agent.energy)
        if remaining > 0.0:
            agent.energy = 0.0
            self._world.deposit_heat(agent.y, agent.x, remaining, ledger, "erasure")

        agent.entropy = 0.0
        agent.alive = False
        agent.death_tick = tick
        agent.death_cause = cause

        self._log.emit(
            tick, EventType.AGENT_DIED, agent.id,
            cause=cause, free_energy=round(free_energy, 4),
            temperature=round(temperature, 4), generation=agent.generation,
            ticks_survived=agent.ticks_survived, released_bits=round(released, 4),
            total_tokens=agent.total_tokens,
        )
        return True

    # -- perception --------------------------------------------------------

    def perceive(self, agent: Agent, tick: int) -> Percept:
        """Build what the organism senses, degraded by its own disorder.

        Rungs of the ladder corrupt perception in escalating ways: first the
        readings acquire noise, then whole nodes go missing from the map. The
        organism is never told which of its beliefs are wrong -- an accurate
        error bar would be a form of the very integration that high entropy is
        supposed to have destroyed.
        """
        world_cfg = self._config.world
        agent_cfg = self._config.agent

        temperature = self.local_temperature(agent)
        nodes = self._world.nodes_near(
            agent.y, agent.x, world_cfg.sense_radius, agent_cfg.vision_node_limit
        )

        degraded = agent.degradation >= Degradation.NOISY
        # A confused organism loses the memory of what it just did, along with
        # the nodes it can no longer perceive. Disorder corrupts the map; there
        # is no reason it would spare the organism's memory of its own acts.
        remembered = (
            None if agent.degradation >= Degradation.CONFUSED else agent.last_action
        )
        energy = agent.energy
        entropy = agent.entropy
        reported_temperature = temperature

        if degraded:
            stream = self._rng.agent_stream(agent.id)
            scale = self.sense_noise(agent)

            if agent.degradation >= Degradation.CONFUSED:
                # Information loss: some nodes simply are not perceived.
                keep = [
                    node
                    for node in nodes
                    if stream.random() >= self.corruption_probability(agent)
                ]
                nodes = keep

            def blur(value: float) -> float:
                return float(value * (1.0 + stream.normal(0.0, scale)))

            energy = blur(energy)
            entropy = blur(entropy)
            reported_temperature = max(
                self._config.physics.min_temperature, blur(reported_temperature)
            )

        return Percept(
            tick=tick,
            agent_id=agent.id,
            energy=energy,
            entropy=entropy,
            phi=agent.phi,
            # Free energy is reported from the (possibly corrupted) readings, so
            # a degraded organism can be wrong about how close it is to death.
            free_energy=energy - reported_temperature * entropy * LN2,
            local_temperature=reported_temperature,
            ambient_temperature=world_cfg.ambient_temperature,
            degradation=agent.degradation,
            nodes=tuple(nodes),
            on_node=self._world.node_index_at(agent.y, agent.x) >= 0,
            last_action=remembered,
            degraded=degraded,
        )
