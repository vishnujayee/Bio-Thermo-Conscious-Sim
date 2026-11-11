"""Terminal report for a finished run.

    python -m btcs.report runs/default_seed42/events.jsonl
    python -m btcs.report runs/*/events.jsonl --trait dissipate_at

Built last, on metrics that real runs proved worth watching (D-4). The panels
here are the ones that actually caught things: the population curve caught the
boom-bust, the death causes caught organisms cooking rather than starving, the
refusal list caught a language model trying to eat where there was no food
nineteen turns out of twenty-three, and the selection differential is the
instrument behind every evolutionary claim the project makes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from btcs.analysis import Run, load

BLOCKS = "▁▂▃▄▅▆▇█"
ASCII = ".:-=+*#@"


def _ramp(stream: Any = None) -> str:
    """Block characters where the console can render them, ASCII where it cannot.

    Windows consoles still default to cp1252, which cannot encode a block
    element -- and a report that raises UnicodeEncodeError halfway through a
    table is worse than one drawn in punctuation. Checked rather than assumed,
    because the answer differs between a terminal, a pipe and a CI log.
    """
    encoding = getattr(stream or sys.stdout, "encoding", None) or "ascii"
    try:
        BLOCKS.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return ASCII
    return BLOCKS


def sparkline(values: list[float], width: int = 60, ramp: str | None = None) -> str:
    """A curve in one line of text. Crude, and it has caught real problems.

    The population sparkline is what showed the boom-bust cycle at a glance, and
    the food curve next to it is what showed the crash was self-inflicted rather
    than imposed.
    """
    if not values:
        return ""
    ramp = ramp or _ramp()
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    low, high = min(values), max(values)
    if high - low < 1e-12:
        return ramp[0] * len(values)
    return "".join(
        ramp[min(len(ramp) - 1, int((v - low) / (high - low) * (len(ramp) - 1)))]
        for v in values
    )


def _headline(run: Run) -> Panel:
    outcome = "[red]EXTINCT[/red]" if run.extinct else f"[green]{run.final_population} alive[/green]"
    body = Text.from_markup(
        f"[bold]{run.name}[/bold]  seed {run.seed}  ·  brain [cyan]{run.brain}[/cyan]"
        f"  ·  scenario [magenta]{run.scenario}[/magenta]\n"
        f"{run.duration} ticks  ·  {outcome}  ·  peak population {run.peak_population}"
        f"  ·  mean {run.mean_population:.1f}\n"
        f"{run.births} births across {run.generations} generations"
        f"  ·  mean lifespan {run.mean_lifespan():.1f} ticks"
        + (f"  ·  spend ${run.spend_usd:.4f}" if run.spend_usd else "")
    )
    return Panel(body, title="run", border_style="blue")


def _curves(run: Run) -> Table:
    table = Table(title="the shape of the run", show_header=True, header_style="bold")
    table.add_column("series", style="cyan", no_wrap=True)
    table.add_column("curve", no_wrap=True)
    table.add_column("first", justify="right")
    table.add_column("last", justify="right")

    series = {
        "population": [float(t.population) for t in run.ticks],
        "agent energy": [t.agent_energy for t in run.ticks],
        "agent entropy": [t.agent_entropy for t in run.ticks],
        "food available": [t.extractable_energy for t in run.ticks],
        "mean temperature": [t.mean_temperature for t in run.ticks],
        "entropy per tick": [t.ds_total for t in run.ticks],
    }
    for name, values in series.items():
        if not values:
            continue
        table.add_row(name, sparkline(values), f"{values[0]:,.1f}", f"{values[-1]:,.1f}")
    return table


def _behaviour(run: Run) -> Table:
    table = Table(title="what they did, and what it cost them", header_style="bold")
    table.add_column("acts", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("deaths by cause", style="cyan")
    table.add_column("n", justify="right")

    acts = run.actions.most_common()
    causes = run.deaths_by_cause.most_common()
    for index in range(max(len(acts), len(causes), 1)):
        left = acts[index] if index < len(acts) else ("", "")
        right = causes[index] if index < len(causes) else ("", "")
        table.add_row(str(left[0]), str(left[1]), str(right[0]), str(right[1]))
    return table


def _refusals(run: Run) -> Panel | None:
    """What the world said no to.

    Worth its own panel because it is where the most surprising result came
    from: a language model spent nineteen turns out of twenty-three trying to
    eat where there was no food, which no aggregate metric would have shown.
    """
    if not run.refusals:
        return None
    lines = [f"{count:>5}  {reason}" for reason, count in run.refusals.most_common(6)]
    total = sum(run.refusals.values())
    return Panel("\n".join(lines), title=f"refused acts ({total})", border_style="yellow")


def _selection(run: Run, traits: list[str]) -> Table:
    table = Table(
        title="selection differential -- organisms that reproduced vs those that did not",
        header_style="bold",
    )
    table.add_column("trait", style="cyan")
    table.add_column("bred", justify="right")
    table.add_column("never bred", justify="right")
    table.add_column("difference", justify="right")
    table.add_column("n", justify="right")

    for trait in traits:
        bred, never, n_bred, n_never = run.selection_differential(trait)
        if bred != bred or never != never:  # NaN
            continue
        difference = bred - never
        colour = "green" if difference > 0 else "red" if difference < 0 else "white"
        table.add_row(
            trait, f"{bred:.3f}", f"{never:.3f}",
            f"[{colour}]{difference:+.3f}[/{colour}]", f"{n_bred}/{n_never}",
        )
    return table


def _generations(run: Run, trait: str) -> Table | None:
    by_generation = run.trait_by_generation(trait)
    if len(by_generation) < 2:
        return None
    table = Table(title=f"{trait} by generation", header_style="bold")
    table.add_column("gen", justify="right", style="cyan")
    table.add_column("mean", justify="right")
    table.add_column("", no_wrap=True)
    values = list(by_generation.values())
    low, high = min(values), max(values)
    span = max(high - low, 1e-9)
    for generation, value in by_generation.items():
        unit = _ramp()[-1]
        bar = unit * max(1, int((value - low) / span * 30))
        table.add_row(str(generation), f"{value:.2f}", bar)
    return table


def _scenario(run: Run) -> Table | None:
    if not run.scenario_events:
        return None
    table = Table(title=f"what happened to the world ({run.scenario})", header_style="bold")
    table.add_column("tick", justify="right", style="cyan")
    table.add_column("note")
    table.add_column("change")
    for event in run.scenario_events[:12]:
        change = ", ".join(f"{k}={v}" for k, v in (event.get("changed") or {}).items())
        if event.get("action"):
            change = (change + " · " if change else "") + str(event["action"])
        table.add_row(str(event["tick"]), event.get("note") or "", change)
    return table


def render(run: Run, console: Console, trait: str = "dissipate_at") -> None:
    console.print(_headline(run))
    console.print(_curves(run))
    console.print(_behaviour(run))
    refusals = _refusals(run)
    if refusals is not None:
        console.print(refusals)
    scenario = _scenario(run)
    if scenario is not None:
        console.print(scenario)
    console.print(_selection(run, ["phi", "dissipate_at", "replicate_at", "greed",
                                   "heat_aversion", "explore_bias", "lifespan"]))
    generations = _generations(run, trait)
    if generations is not None:
        console.print(generations)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="btcs.report", description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--trait", default="dissipate_at",
                        help="heritable trait to chart by generation")
    args = parser.parse_args(argv)

    # Ask for UTF-8 first; `_ramp` falls back to ASCII if the console refuses.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    console = Console()
    runs = []
    for path in args.logs:
        if not path.is_file():
            console.print(f"[red]no such log:[/red] {path}")
            return 2
        runs.append(load(path))

    for run in runs:
        render(run, console, args.trait)

    if len(runs) > 1:
        from btcs.analysis import compare

        console.print(Panel(compare(runs, "phi"), title="across runs",
                            border_style="blue"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
