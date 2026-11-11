"""Reading a run back: metrics, lineage, replay, and the two renderers.

Every result this project reports was computed from a log. If the reader is
wrong, every result is wrong and nothing else in the test suite would notice --
so these tests check the analysis against a simulation whose truth is known,
rather than against a fixture that agrees with it by construction.
"""

from __future__ import annotations

import dataclasses

import pytest

from btcs.analysis import Run, compare, load
from btcs.config import Config
from btcs.events import EventLog
from btcs.runner import SimulationRunner


@pytest.fixture(scope="module")
def finished(tmp_path_factory) -> tuple[Run, "SimulationRunner"]:
    """One real run, and the runner that produced it, for cross-checking."""
    base = Config()
    config = dataclasses.replace(
        base,
        run=dataclasses.replace(base.run, max_ticks=140, name="analysis"),
        world=dataclasses.replace(base.world, width=20, height=20, node_count=14),
        agent=dataclasses.replace(base.agent, initial_population=10),
    )
    path = tmp_path_factory.mktemp("run") / "events.jsonl"
    with EventLog(path) as log:
        runner = SimulationRunner(config, log=log)
        runner.run()
    return load(path), runner


class TestAgainstTheSimulation:
    """The reader must agree with the thing it is reading."""

    def test_the_run_identifies_itself(self, finished):
        run, runner = finished
        assert run.name == runner.config.run.name
        assert run.seed == runner.config.run.seed
        assert run.brain == "mock"

    def test_final_population_matches(self, finished):
        run, runner = finished
        assert run.final_population == runner.registry.population

    def test_the_log_remembers_organisms_the_registry_has_forgotten(self, finished):
        """The registry drops an organism when it is erased -- it holds the
        living, not the history. The log holds everyone who ever lived, which is
        exactly why every result in this project is computed from the log and
        not from the simulation's own memory.
        """
        run, runner = finished
        living = {a.id for a in runner.registry.all()}
        assert living <= set(run.organisms)
        assert len(run.organisms) > len(living), "nothing died; widen the test run"
        assert all(run.organisms[a].died is None for a in living)

    def test_offspring_counts_match(self, finished):
        run, runner = finished
        for agent in runner.registry.all():
            assert run.organisms[agent.id].offspring == agent.offspring

    def test_generations_match(self, finished):
        run, runner = finished
        assert run.generations == max(a.generation for a in runner.registry.all())

    def test_positions_match_at_the_final_tick(self, finished):
        """Replay is reconstruction, not storage: birth gives a starting cell and
        every move gives the new one. If that is wrong, the animation lies."""
        run, runner = finished
        placed = run.positions_at(run.duration)
        for agent in runner.registry.living():
            assert placed[agent.id] == (agent.y, agent.x)


class TestMetrics:
    def test_the_series_covers_every_tick(self, finished):
        run, _ = finished
        assert [t.tick for t in run.ticks] == list(range(1, run.duration + 1))

    def test_cumulative_entropy_never_falls(self, finished):
        """The second law, re-checked from the log rather than from memory."""
        run, _ = finished
        cumulative = [t.cumulative for t in run.ticks]
        assert all(b >= a - 1e-9 for a, b in zip(cumulative, cumulative[1:], strict=False))

    def test_births_and_lineage_agree(self, finished):
        run, _ = finished
        assert run.births == sum(1 for o in run.organisms.values() if o.parent)

    def test_founders_have_no_parent(self, finished):
        run, _ = finished
        assert all(o.generation == 1 for o in run.founders)

    def test_every_child_names_a_parent_that_existed(self, finished):
        run, _ = finished
        for organism in run.organisms.values():
            if organism.parent:
                assert organism.parent in run.organisms

    def test_descendants_counts_the_whole_subtree(self, finished):
        run, _ = finished
        breeders = [o for o in run.organisms.values() if o.bred]
        if breeders:
            assert run.descendants(breeders[0].id) >= breeders[0].offspring

    def test_deaths_by_cause_matches_the_dead(self, finished):
        run, _ = finished
        dead = sum(1 for o in run.organisms.values() if o.died is not None)
        assert sum(run.deaths_by_cause.values()) == dead


class TestSelectionDifferential:
    """The instrument behind every evolutionary claim the project makes."""

    def test_it_splits_by_whether_an_organism_bred(self, finished):
        run, _ = finished
        _, _, n_bred, n_never = run.selection_differential("dissipate_at")
        assert n_bred == sum(1 for o in run.organisms.values() if o.bred)
        assert n_never == sum(1 for o in run.organisms.values() if not o.bred)

    def test_an_unknown_trait_yields_nothing(self, finished):
        run, _ = finished
        _, _, n_bred, n_never = run.selection_differential("charisma")
        assert n_bred == n_never == 0

    def test_trait_by_generation_covers_every_generation(self, finished):
        run, _ = finished
        by_generation = run.trait_by_generation("dissipate_at")
        assert set(by_generation) == {o.generation for o in run.organisms.values()}

    def test_compare_renders_a_row_per_run(self, finished):
        run, _ = finished
        assert len(compare([run, run], "phi").splitlines()) == 3


class TestRenderers:
    """A renderer that crashes on a real log is a renderer nobody will use."""

    def test_the_terminal_report_renders(self, finished, capsys):
        from rich.console import Console

        from btcs.report import render

        run, _ = finished
        render(run, Console(file=None, width=100, force_terminal=False))
        assert "selection differential" in capsys.readouterr().out

    def test_sparklines_survive_a_console_that_cannot_draw_them(self):
        """Windows consoles still default to cp1252. A report that raises
        UnicodeEncodeError halfway through a table is worse than an ASCII one."""
        from btcs.report import ASCII, sparkline

        drawn = sparkline([1.0, 5.0, 3.0], ramp=ASCII)
        assert drawn and all(character in ASCII for character in drawn)
        assert drawn.encode("cp1252")

    def test_a_flat_series_still_draws(self):
        from btcs.report import sparkline

        assert len(sparkline([2.0, 2.0, 2.0])) == 3

    def test_an_empty_series_draws_nothing(self):
        from btcs.report import sparkline

        assert sparkline([]) == ""

    def test_the_html_view_is_self_contained(self, finished):
        """No server, no dependencies, no network -- except fonts, which are
        allowed to fail to a real fallback stack."""
        from btcs.viz import build

        run, _ = finished
        html = build(run, run.path, stride=2)
        assert "<title>" in html and "const DATA" in html
        assert "http://" not in html
        for remote in ("cdn.", "unpkg", "jsdelivr"):
            assert remote not in html

    def test_the_html_view_carries_real_frames(self, finished):
        import json
        import re

        from btcs.viz import build

        run, _ = finished
        html = build(run, run.path, stride=1)
        data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
        assert len(data["frames"]) == run.duration
        assert data["frames"][0]["agents"]
        assert data["lineage"]

    def test_stride_thins_the_frames(self, finished):
        import json
        import re

        from btcs.viz import build

        run, _ = finished
        html = build(run, run.path, stride=5)
        data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
        assert len(data["frames"]) < run.duration

    def test_a_custom_title_is_used(self, finished):
        from btcs.viz import build

        run, _ = finished
        assert "<title>Sandbox</title>" in build(run, run.path, title="Sandbox")


class TestScenarioRuns:
    def test_interventions_are_read_back(self, small_config, tmp_path):
        config = dataclasses.replace(
            small_config,
            run=dataclasses.replace(small_config.run, max_ticks=80),
            scenario=dataclasses.replace(small_config.scenario, name="bottleneck"),
        )
        path = tmp_path / "s.jsonl"
        with EventLog(path) as log:
            SimulationRunner(config, log=log).run()
        run = load(path)
        assert run.scenario == "bottleneck"
        assert run.scenario_events
