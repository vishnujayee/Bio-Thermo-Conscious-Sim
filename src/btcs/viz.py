"""Turn a run into something you can watch.

    python -m btcs.viz runs/default_seed42/events.jsonl -o run.html

Produces one self-contained HTML file: no server, no dependencies, no network.
Open it in a browser and press play.

**Why this is worth having.** Every result in this project was found in numbers,
and two of the most important ones -- the boom-bust cycle, and a language model
trying to eat where there was no food nineteen turns out of twenty-three -- were
things you would have seen in five seconds of watching and might never have
thought to compute.

**What is real in the picture.** Organism positions are exact: birth records a
starting cell and every move records the new one. Node positions are exact too:
the initial layout is a deterministic function of the seed, which the log
carries, and every later spawn and exhaustion is recorded. Per-node energy is
*not* logged and is not shown -- a node is drawn as present or absent, and the
legend says so rather than inventing a brightness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from btcs.analysis import Run, load
from btcs.events import read_events

TEMPLATE = """<title>{title}</title>
<link rel="stylesheet"
  href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@500;600&display=swap">
<style>
  /* Palette taken from the physics rather than from a UI kit: the cold
     reservoir is blue, waste heat is orange, living things are green, and
     erasure is the one red on the page. Neutrals carry a cool bias toward the
     cold reservoir. */
  :root {{
    --bg:#e9edef; --panel:#f8fafb; --sunk:#e2e7ea;
    --ink:#12171b; --muted:#5b676e; --line:#cfd7dc;
    --life:#1c7a58; --heat:#bc5620; --cold:#2a68ad; --erase:#9a2b2b;
    --grid:#dde3e7;
    --serif:"IBM Plex Serif",Georgia,serif;
    --sans:"IBM Plex Sans",ui-sans-serif,system-ui,sans-serif;
    --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --bg:#0e1215; --panel:#151b1f; --sunk:#0a0e11;
      --ink:#e4eaed; --muted:#8a959c; --line:#242c32;
      --life:#4dbf91; --heat:#dd8f4f; --cold:#69a6e2; --erase:#dd7a6a;
      --grid:#1c2429;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg:#0e1215; --panel:#151b1f; --sunk:#0a0e11;
    --ink:#e4eaed; --muted:#8a959c; --line:#242c32;
    --life:#4dbf91; --heat:#dd8f4f; --cold:#69a6e2; --erase:#dd7a6a;
    --grid:#1c2429;
  }}

  body {{ background:var(--bg); color:var(--ink); margin:0; font-family:var(--sans);
    font-size:15px; line-height:1.6; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:34px 22px 72px;
    display:flex; flex-direction:column; gap:34px; }}

  header h1 {{ font-family:var(--serif); font-weight:600; font-size:30px;
    letter-spacing:-0.015em; margin:0 0 6px; text-wrap:balance; }}
  .meta {{ font-family:var(--mono); font-size:12.5px; color:var(--muted); margin:0; }}

  .eyebrow {{ font-family:var(--mono); font-size:11px; letter-spacing:0.09em;
    text-transform:uppercase; color:var(--muted); margin:0 0 12px;
    padding-bottom:7px; border-bottom:1px solid var(--line); }}

  .instrument {{ display:grid; grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);
    gap:24px; align-items:start; }}
  @media (max-width:820px) {{ .instrument {{ grid-template-columns:1fr; }} }}

  canvas {{ width:100%; height:auto; display:block; }}
  #grid {{ background:var(--sunk); border:1px solid var(--line); }}

  .transport {{ display:flex; gap:10px; align-items:center; margin-top:12px; }}
  button {{ font-family:var(--mono); font-size:12px; padding:6px 13px;
    border:1px solid var(--line); background:var(--panel); color:var(--ink);
    cursor:pointer; }}
  button:hover {{ border-color:var(--life); color:var(--life); }}
  button:focus-visible, input:focus-visible {{ outline:2px solid var(--life);
    outline-offset:2px; }}
  input[type=range] {{ flex:1; min-width:110px; accent-color:var(--life); }}
  #tick {{ font-family:var(--mono); font-size:12px; color:var(--muted);
    min-width:76px; text-align:right; }}

  /* An axis, not decoration: marks sit where interventions actually landed. */
  .axis {{ position:relative; height:22px; margin-top:4px;
    border-top:1px solid var(--line); }}
  .axis b {{ position:absolute; top:0; width:1px; height:7px; background:var(--heat); }}
  .axis span {{ position:absolute; top:8px; transform:translateX(-50%);
    font-family:var(--mono); font-size:9.5px; color:var(--muted); white-space:nowrap; }}

  .readout {{ display:grid; grid-template-columns:1fr 1fr; gap:1px;
    background:var(--line); border:1px solid var(--line); }}
  .readout div {{ background:var(--panel); padding:10px 12px; }}
  .readout span {{ display:block; font-family:var(--mono); font-size:10.5px;
    letter-spacing:0.07em; text-transform:uppercase; color:var(--muted); }}
  .readout b {{ font-family:var(--mono); font-size:19px; font-weight:500;
    font-variant-numeric:tabular-nums; }}

  .legend {{ display:flex; gap:8px 15px; flex-wrap:wrap; color:var(--muted);
    font-size:12px; margin-top:11px; font-family:var(--mono); }}
  .legend i {{ display:inline-block; width:9px; height:9px; margin-right:5px; }}
  .legend i.dot {{ border-radius:50%; }}

  table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
  th, td {{ text-align:left; padding:6px 10px 6px 0;
    border-bottom:1px solid var(--line); }}
  th {{ font-family:var(--mono); font-size:10.5px; letter-spacing:0.07em;
    text-transform:uppercase; color:var(--muted); font-weight:500; }}
  td.num, th.num {{ text-align:right; font-family:var(--mono);
    font-variant-numeric:tabular-nums; }}
  .pos {{ color:var(--life); }} .neg {{ color:var(--erase); }}
  .scroll {{ overflow-x:auto; }}
  .note {{ color:var(--muted); font-size:13px; max-width:64ch; }}
  .cols {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:26px; }}
  @media (max-width:820px) {{ .cols {{ grid-template-columns:1fr; }} }}
  .prose p {{ max-width:66ch; }}
  .prose p:first-of-type {{ margin-top:0; }}
  @media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">
  <header>
    <h1>{name}</h1>
    <p class="meta">{subtitle}</p>
  </header>

  <section>
    <p class="eyebrow">the world, tick by tick</p>
    <div class="instrument">
      <div>
        <canvas id="grid" width="{cw}" height="{cw}"></canvas>
        <div class="transport">
          <button id="play">play</button>
          <input type="range" id="scrub" min="0" max="{last_frame}" value="0"
                 aria-label="scrub through the run">
          <button id="speed">1&times;</button>
          <span id="tick">tick 0</span>
        </div>
        <div class="axis" id="marks"></div>
        <div class="legend">
          <span><i style="background:var(--life)"></i>lucid</span>
          <span><i style="background:var(--heat)"></i>noisy</span>
          <span><i style="background:var(--erase)"></i>confused / demented</span>
          <span><i class="dot" style="background:var(--cold)"></i>free-energy node</span>
        </div>
      </div>
      <div>
        <div class="readout">
          <div><span>population</span><b id="s-pop">0</b></div>
          <div><span>nodes</span><b id="s-nodes">0</b></div>
          <div><span>energy E</span><b id="s-energy">0</b></div>
          <div><span>entropy S</span><b id="s-entropy">0</b></div>
          <div><span>mean T</span><b id="s-temp">0</b></div>
          <div><span>erased</span><b id="s-deaths">0</b></div>
        </div>
        <canvas id="curves" width="640" height="300" style="margin-top:16px"></canvas>
        <div class="legend">
          <span><i style="background:var(--life)"></i>population</span>
          <span><i style="background:var(--cold)"></i>extractable energy</span>
          <span><i style="background:var(--heat)"></i>agent entropy</span>
        </div>
        <p class="note" style="margin-top:12px">A node is drawn present or absent.
          Per-node energy is not written to the log, so it is not shown rather
          than guessed at.</p>
      </div>
    </div>
  </section>

  <section>
    <p class="eyebrow">lineage &mdash; every organism that lived</p>
    <div class="scroll"><canvas id="tree" width="1000" height="{tree_h}"></canvas></div>
    <p class="note" style="margin-top:12px">Each horizontal line is one life,
      drawn from birth to erasure on the row of its generation. Thin grey lines
      died childless; thick green ones reproduced. Diagonals join children to
      parents.</p>
  </section>

  <section>
    <p class="eyebrow">does anything evolve?</p>
    <div class="cols">
      <div>
        <table>
          <tr><th>trait</th><th class="num">bred</th><th class="num">never bred</th>
              <th class="num">difference</th></tr>
          {differentials}
        </table>
        <p class="note" style="margin-top:12px">The selection differential: a
          heritable trait compared between organisms that reproduced and those
          that never did. It says which organisms bred &mdash; not that the
          population mean moved and stayed moved. Traits with no signal are the
          control, and most of them should have none.</p>
      </div>
      <div>
        <canvas id="gens" width="620" height="260"></canvas>
        <p class="note" style="margin-top:12px">Mean {trait} by generation.</p>
      </div>
    </div>
  </section>

  {scenario_block}

  <section class="prose">
    <p class="eyebrow">what you are looking at</p>
    <p>Each square is an organism living on an energy and entropy budget. It
    harvests free energy from the blue nodes, and everything it does &mdash;
    moving, thinking, remembering &mdash; generates disorder it must spend energy
    to erase. When its free energy <b>F = E &minus; T&middot;S&middot;ln2</b>
    reaches zero, it is erased.</p>
    <p>Nothing here is scored, rewarded, or given a fitness function. The squares
    that survive are the ones whose behaviour happened to keep them solvent. The
    colours are the degradation ladder: as disorder accumulates, perception
    first acquires noise, then loses detail outright, then fills with junk.</p>
    <p>The population almost always overshoots and crashes, and nobody designed
    that. Waste heat from a crowd raises the local temperature, which lowers the
    Carnot efficiency of every meal nearby &mdash; so a crowd degrades its own
    metabolism by being a crowd.</p>
  </section>
</div>

<script>
const DATA = {data};

const rampC = getComputedStyle(document.documentElement);
const col = n => rampC.getPropertyValue(n).trim();

// ---- grid playback -------------------------------------------------------
const grid = document.getElementById('grid'), gctx = grid.getContext('2d');
const cell = grid.width / DATA.width;

function drawGrid(frame) {{
  const f = DATA.frames[frame];
  gctx.fillStyle = col('--panel'); gctx.fillRect(0, 0, grid.width, grid.height);
  gctx.strokeStyle = col('--grid'); gctx.lineWidth = 1;
  const step = Math.max(1, Math.round(DATA.width / 12));
  for (let i = 0; i <= DATA.width; i += step) {{
    gctx.beginPath(); gctx.moveTo(i * cell, 0); gctx.lineTo(i * cell, grid.height); gctx.stroke();
    gctx.beginPath(); gctx.moveTo(0, i * cell); gctx.lineTo(grid.width, i * cell); gctx.stroke();
  }}
  gctx.fillStyle = col('--cold');
  for (const [y, x] of f.nodes) {{
    gctx.beginPath();
    gctx.arc((x + 0.5) * cell, (y + 0.5) * cell, Math.max(1.6, cell * 0.32), 0, 6.284);
    gctx.fill();
  }}
  const shade = ['--accent', '--warm', '--dead', '--dead'];
  for (const [y, x, rung] of f.agents) {{
    gctx.fillStyle = col(shade[Math.min(3, rung)]);
    const r = Math.max(2.2, cell * 0.42);
    gctx.fillRect((x + 0.5) * cell - r, (y + 0.5) * cell - r, r * 2, r * 2);
  }}
}}

// ---- curves --------------------------------------------------------------
const curves = document.getElementById('curves'), cctx = curves.getContext('2d');
function series(key) {{ return DATA.ticks.map(t => t[key]); }}
function drawCurves(frame) {{
  const W = curves.width, H = curves.height, pad = 26;
  cctx.clearRect(0, 0, W, H);
  const sets = [['pop', '--accent'], ['food', '--cold'], ['entropy', '--warm']];
  for (const [key, colour] of sets) {{
    const vals = series(key);
    const hi = Math.max(...vals, 1e-9), lo = Math.min(...vals, 0);
    cctx.strokeStyle = col(colour); cctx.lineWidth = 1.8; cctx.beginPath();
    vals.forEach((v, i) => {{
      const x = pad + (i / Math.max(1, vals.length - 1)) * (W - pad * 2);
      const y = H - pad - ((v - lo) / Math.max(1e-9, hi - lo)) * (H - pad * 2);
      i ? cctx.lineTo(x, y) : cctx.moveTo(x, y);
    }});
    cctx.stroke();
  }}
  const t = DATA.frames[frame].tick;
  const idx = DATA.ticks.findIndex(r => r.tick >= t);
  if (idx >= 0) {{
    const x = pad + (idx / Math.max(1, DATA.ticks.length - 1)) * (W - pad * 2);
    cctx.strokeStyle = col('--muted'); cctx.lineWidth = 1;
    cctx.beginPath(); cctx.moveTo(x, pad * 0.4); cctx.lineTo(x, H - pad * 0.4); cctx.stroke();
  }}
}}

// ---- lineage -------------------------------------------------------------
const tree = document.getElementById('tree'), tctx = tree.getContext('2d');
(function drawTree() {{
  const W = tree.width, H = tree.height, pad = 22;
  const maxTick = Math.max(...DATA.lineage.map(o => o.died ?? DATA.duration), 1);
  const maxGen = Math.max(...DATA.lineage.map(o => o.gen), 1);
  const px = t => pad + (t / maxTick) * (W - pad * 2);
  const py = g => pad + ((g - 1) / Math.max(1, maxGen - 1)) * (H - pad * 2);
  const at = {{}};
  for (const o of DATA.lineage) at[o.id] = o;
  tctx.strokeStyle = col('--line'); tctx.lineWidth = 1;
  for (const o of DATA.lineage) {{
    if (!o.parent || !at[o.parent]) continue;
    tctx.beginPath();
    tctx.moveTo(px(at[o.parent].born), py(at[o.parent].gen));
    tctx.lineTo(px(o.born), py(o.gen));
    tctx.stroke();
  }}
  for (const o of DATA.lineage) {{
    tctx.strokeStyle = o.bred ? col('--accent') : col('--muted');
    tctx.lineWidth = o.bred ? 2.4 : 1.2;
    tctx.beginPath();
    tctx.moveTo(px(o.born), py(o.gen));
    tctx.lineTo(px(o.died ?? DATA.duration), py(o.gen));
    tctx.stroke();
  }}
}})();

// ---- trait by generation -------------------------------------------------
const gens = document.getElementById('gens'), xctx = gens.getContext('2d');
(function drawGens() {{
  const W = gens.width, H = gens.height, pad = 34;
  const rows = DATA.byGeneration;
  if (!rows.length) return;
  const vals = rows.map(r => r[1]);
  const hi = Math.max(...vals), lo = Math.min(...vals);
  const span = Math.max(1e-9, hi - lo);
  const bw = (W - pad * 2) / rows.length;
  xctx.font = '11px ui-sans-serif, system-ui, sans-serif';
  rows.forEach((r, i) => {{
    const h = ((r[1] - lo) / span) * (H - pad * 2) + 4;
    xctx.fillStyle = col('--accent');
    xctx.fillRect(pad + i * bw + bw * 0.18, H - pad - h, bw * 0.64, h);
    xctx.fillStyle = col('--muted');
    xctx.fillText('g' + r[0], pad + i * bw + bw * 0.2, H - pad + 13);
    xctx.fillText(r[1].toFixed(1), pad + i * bw + bw * 0.1, H - pad - h - 5);
  }});
}})();

// ---- season markers ------------------------------------------------------
(function marks() {{
  const host = document.getElementById('marks');
  for (const e of DATA.scenario) {{
    const b = document.createElement('b');
    b.style.left = (100 * e.tick / Math.max(1, DATA.duration)) + '%';
    b.title = e.note || e.action || 'intervention';
    host.appendChild(b);
  }}
}})();

// ---- transport -----------------------------------------------------------
let frame = 0, playing = false, speed = 1, timer = null;
const scrub = document.getElementById('scrub');
const label = document.getElementById('tick');
const fmt = n => n >= 1000 ? Math.round(n).toLocaleString() : n.toFixed(1);

function show(i) {{
  frame = Math.max(0, Math.min(DATA.frames.length - 1, i));
  const f = DATA.frames[frame];
  scrub.value = frame;
  label.textContent = 'tick ' + f.tick;
  drawGrid(frame); drawCurves(frame);
  document.getElementById('s-pop').textContent = f.agents.length;
  document.getElementById('s-nodes').textContent = f.nodes.length;
  document.getElementById('s-energy').textContent = fmt(f.energy);
  document.getElementById('s-entropy').textContent = fmt(f.entropy);
  document.getElementById('s-temp').textContent = f.temp.toFixed(2);
  document.getElementById('s-deaths').textContent = f.deaths;
}}
function tickOn() {{
  if (!playing) return;
  show(frame + 1 >= DATA.frames.length ? 0 : frame + 1);
  timer = setTimeout(tickOn, 90 / speed);
}}
document.getElementById('play').onclick = e => {{
  playing = !playing;
  e.target.textContent = playing ? 'pause' : 'play';
  clearTimeout(timer); if (playing) tickOn();
}};
document.getElementById('speed').onclick = e => {{
  speed = speed >= 8 ? 1 : speed * 2;
  e.target.textContent = speed + 'x';
}};
scrub.oninput = e => {{ playing = false;
  document.getElementById('play').textContent = 'play';
  clearTimeout(timer); show(+e.target.value); }};
show(0);
</script>
"""


def _frames(run: Run, path: Path, stride: int) -> tuple[list[dict], int]:
    """Reconstruct the world tick by tick, from the log alone."""
    from btcs.config import Config
    from btcs.rng import RngService
    from btcs.world.grid import World

    config = Config.from_dict(run.config) if run.config else Config()
    world = World(config, RngService(config.run.seed))
    nodes: dict[tuple[int, int], bool] = {
        (int(y), int(x)): True for y, x in zip(world.node_y, world.node_x, strict=True)
    }

    node_changes: dict[int, list[tuple[str, int, int]]] = {}
    for event in read_events(path):
        kind = event["type"]
        if kind in ("node_spawned", "node_exhausted"):
            node_changes.setdefault(event["tick"], []).append(
                ("+" if kind == "node_spawned" else "-", int(event["y"]), int(event["x"]))
            )

    rungs: dict[str, int] = {}
    rung_changes: dict[int, list[tuple[str, int]]] = {}
    names = {"lucid": 0, "noisy": 1, "confused": 2, "demented": 3}
    for event in read_events(path):
        if event["type"] == "degradation":
            rung_changes.setdefault(event["tick"], []).append(
                (event["agent"], names.get(event.get("to", "lucid"), 0))
            )

    frames, deaths = [], 0
    by_tick = {t.tick: t for t in run.ticks}
    for tick in range(1, run.duration + 1):
        for sign, y, x in node_changes.get(tick, []):
            if sign == "+":
                nodes[(y, x)] = True
            else:
                nodes.pop((y, x), None)
        for agent_id, rung in rung_changes.get(tick, []):
            rungs[agent_id] = rung
        snapshot = by_tick.get(tick)
        if snapshot is not None:
            deaths += snapshot.deaths
        if tick % stride and tick != run.duration:
            continue

        placed = run.positions_at(tick)
        frames.append({
            "tick": tick,
            "agents": [[y, x, rungs.get(a, 0)] for a, (y, x) in sorted(placed.items())],
            "nodes": [[y, x] for (y, x) in sorted(nodes)],
            "energy": round(snapshot.agent_energy, 1) if snapshot else 0.0,
            "entropy": round(snapshot.agent_entropy, 1) if snapshot else 0.0,
            "temp": round(snapshot.mean_temperature, 3) if snapshot else 0.0,
            "deaths": deaths,
        })
    return frames, config.world.width


def build(run: Run, path: Path, *, trait: str = "dissipate_at",
          stride: int = 1, title: str | None = None) -> str:
    frames, width = _frames(run, path, stride)

    rows = []
    for name in ("phi", "dissipate_at", "replicate_at", "greed", "heat_aversion",
                 "explore_bias", "lifespan"):
        bred, never, n_bred, n_never = run.selection_differential(name)
        if bred != bred or never != never:
            continue
        difference = bred - never
        css = "pos" if difference > 0 else "neg" if difference < 0 else ""
        rows.append(
            f'<tr><td>{name}</td><td class="num">{bred:.3f}</td>'
            f'<td class="num">{never:.3f}</td>'
            f'<td class="num {css}">{difference:+.3f}</td></tr>'
        )

    scenario_block = ""
    if run.scenario_events:
        items = "".join(
            f'<tr><td class="num">{e["tick"]}</td><td>{e.get("note") or ""}</td>'
            f'<td>{", ".join(f"{k}={v}" for k, v in (e.get("changed") or {}).items())}'
            f'{" · " + e["action"] if e.get("action") else ""}</td></tr>'
            for e in run.scenario_events[:20]
        )
        scenario_block = (
            f'<h2>What happened to the world</h2><div class="card scroll">'
            f'<table><tr><th class="num">tick</th><th>season</th><th>change</th></tr>'
            f'{items}</table><p class="note">Orange marks under the timeline show '
            f'when each intervention landed.</p></div>'
        )

    outcome = "went extinct" if run.extinct else f"{run.final_population} alive at the end"
    data = {
        "width": width,
        "duration": run.duration,
        "frames": frames,
        "ticks": [
            {"tick": t.tick, "pop": t.population, "food": round(t.extractable_energy, 1),
             "entropy": round(t.agent_entropy, 1)}
            for t in run.ticks
        ],
        "lineage": [
            {"id": o.id, "parent": o.parent, "gen": o.generation, "born": o.born,
             "died": o.died, "bred": o.bred}
            for o in sorted(run.organisms.values(), key=lambda o: (o.generation, o.born))
        ],
        "byGeneration": [[g, round(v, 3)] for g, v in run.trait_by_generation(trait).items()],
        "scenario": [
            {"tick": e["tick"], "note": e.get("note") or "", "action": e.get("action") or ""}
            for e in run.scenario_events
        ],
    }

    return TEMPLATE.format(
        title=title or f"{run.name} — a thermodynamic ecology",
        name=run.name,
        subtitle=(
            f"seed {run.seed} · {run.brain} brain · scenario {run.scenario} · "
            f"{run.duration} ticks · {outcome} · {run.births} births across "
            f"{run.generations} generations"
        ),
        cw=560,
        last_frame=max(0, len(frames) - 1),
        tree_h=max(240, min(520, 40 + run.generations * 26)),
        differentials="".join(rows) or '<tr><td colspan="4">no data</td></tr>',
        trait=trait,
        scenario_block=scenario_block,
        data=json.dumps(data, separators=(",", ":")),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="btcs.viz", description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--trait", default="dissipate_at")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth tick; raise it for very long runs")
    parser.add_argument("--title", default=None,
                        help="page title; defaults to the run name")
    args = parser.parse_args(argv)

    if not args.log.is_file():
        print(f"no such log: {args.log}", file=sys.stderr)
        return 2

    run = load(args.log)
    html = build(run, args.log, trait=args.trait, stride=max(1, args.stride),
                 title=args.title)
    output = args.output or args.log.with_suffix(".html")
    output.write_text(html, encoding="utf-8")
    print(f"wrote {output}  ({len(html) / 1024:.0f} KB, {run.duration} ticks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
