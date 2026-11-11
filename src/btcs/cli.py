"""Headless run entry point.

    python -m btcs.cli --config configs/default.yaml
    python -m btcs.cli --ticks 200 --seed 7 --population 20

Prints a progress line and a summary; the authoritative record is the JSONL log
the run writes, not this output.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from btcs.config import Config, ConfigError
from btcs.runner import SimulationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="btcs", description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config; defaults are used when omitted")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--brain", choices=["mock", "local", "llm"], default=None,
                        help="mock (free, instant) | local (Ollama, free, slow) | llm (paid)")
    parser.add_argument("--local-model", type=str, default=None,
                        help="Ollama model to run, e.g. qwen2.5:3b")
    parser.add_argument("--scenario", type=str, default=None,
                        help="none | heat_wave | drought | gradient_collapse | "
                             "bottleneck | seasons")
    parser.add_argument("--budget", type=float, default=None,
                        help="override the run's USD ceiling")
    parser.add_argument("--estimate", action="store_true",
                        help="price a live run without making any API call, then exit")
    parser.add_argument("--quiet", action="store_true", help="suppress per-tick output")
    parser.add_argument("--every", type=int, default=25,
                        help="print a status line every N ticks")
    return parser


def apply_overrides(config: Config, args: argparse.Namespace) -> Config:
    """CLI flags beat the file, so a sweep can vary one knob without new files."""
    run = config.run
    if args.seed is not None:
        run = dataclasses.replace(run, seed=args.seed)
    if args.ticks is not None:
        run = dataclasses.replace(run, max_ticks=args.ticks)
    if args.name is not None:
        run = dataclasses.replace(run, name=args.name)
    if args.output_dir is not None:
        run = dataclasses.replace(run, output_dir=str(args.output_dir))

    agent = config.agent
    if args.population is not None:
        agent = dataclasses.replace(agent, initial_population=args.population)

    scenario = config.scenario
    if args.scenario is not None:
        scenario = dataclasses.replace(scenario, name=args.scenario, events=[])

    cognition = config.cognition
    if args.brain is not None:
        cognition = dataclasses.replace(cognition, brain=args.brain)
    if args.budget is not None:
        cognition = dataclasses.replace(cognition, budget_usd=args.budget)
    if args.local_model is not None:
        cognition = dataclasses.replace(cognition, local_model=args.local_model)

    return dataclasses.replace(config, run=run, agent=agent, scenario=scenario,
                               cognition=cognition)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = Config.load(args.config) if args.config else Config()
        config = apply_overrides(config, args)
        config.validate()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.estimate:
        # Prices the run from the real prompt assembly path, without a client,
        # a credential, or a single request.
        from btcs.cognition.estimate import estimate_run

        estimate, result = estimate_run(
            dataclasses.replace(
                config, cognition=dataclasses.replace(config.cognition, brain="llm")
            )
        )
        print(f"btcs estimate: {config.run.name} seed={config.run.seed} "
              f"pop={config.agent.initial_population} ticks={config.run.max_ticks}")
        print(f"  simulated: {result.ticks_completed} ticks, "
              f"{result.final_population} survivors")
        print(estimate.report())
        print()
        print(f"  configured ceiling: ${config.cognition.budget_usd:.2f}")
        if estimate.total_usd > config.cognition.budget_usd:
            print("  WARNING: the projected cost exceeds the ceiling; the run "
                  "would halt part-way.")
        print("  No API call was made. Output tokens are assumed, not measured -- "
              "they are the largest source of error here.")
        return 0

    runner = SimulationRunner(config)
    print(
        f"btcs: {config.run.name} seed={config.run.seed} "
        f"pop={config.agent.initial_population} ticks={config.run.max_ticks} "
        f"brain={runner.brain.name}"
        + (f" scenario={runner.scenario.name}" if runner.scenario.events else "")
    )

    def report_progress(report) -> None:
        if args.quiet:
            return
        if report.tick % args.every and report.population:
            return
        print(
            f"  t={report.tick:>5}  pop={report.population:>3}  "
            f"S_agents={report.total_agent_entropy:>8.1f}  "
            f"E_agents={report.total_agent_energy:>9.1f}  "
            f"dS={report.ds_total:>10.1f}  "
            f"free={report.world['extractable_energy']:>9.1f}"
        )

    result = runner.run(on_tick=report_progress)
    print(f"done: {result.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
