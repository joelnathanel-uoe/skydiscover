"""
extract_run.py  —  human-readable summary of an AdaEvolve run

Usage:
    python extract_run.py <run_dir> [output_file]

Output format is determined by the file extension:
    .txt  — plain text (default if no extension)
    .html — interactive HTML with collapsible sections

Examples:
    python extract_run.py /root/skydiscover/outputs/adaevolve/fjsp_twt_0608_1615
    python extract_run.py /root/skydiscover/outputs/adaevolve/fjsp_twt_0608_1615 run_summary.txt
    python extract_run.py /root/skydiscover/outputs/adaevolve/fjsp_twt_0608_1615 run_summary.html
"""

import json
import os
import sys
import html as html_mod
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_programs(checkpoint_dir):
    progs_dir = checkpoint_dir / "programs"
    programs = {}
    for p in progs_dir.glob("*.json"):
        with open(p) as f:
            prog = json.load(f)
        programs[prog["id"]] = prog
    return programs


def latest_checkpoint(run_dir):
    checkpoints = sorted(
        (run_dir / "checkpoints").iterdir(),
        key=lambda p: int(p.name.split("_")[1])
    )
    return checkpoints[-1]


def short_id(uid):
    return uid[:8] if uid else "unknown"


def load_run(run_dir_str):
    run_dir = Path(run_dir_str)
    jsonl_files = list(run_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL found in {run_dir}")
        sys.exit(1)
    iterations = load_jsonl(jsonl_files[0])
    ckpt = latest_checkpoint(run_dir)
    programs = load_programs(ckpt)
    return run_dir, iterations, programs, ckpt, jsonl_files[0]


# ---------------------------------------------------------------------------
# Plain-text output
# ---------------------------------------------------------------------------

def sep(char="=", width=80):
    return char * width


def fmt_metrics(metrics):
    if not metrics:
        return "  (no metrics)"
    return "\n".join(
        f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
        for k, v in metrics.items()
    )


def fmt_code(code):
    if not code:
        return "  (no code)"
    return "\n".join("    " + ln for ln in code.splitlines())


def extract_text(run_dir_str, output_path=None):
    run_dir, iterations, programs, ckpt, jsonl_path = load_run(run_dir_str)
    lines = []

    lines += [
        sep(), f"  AdaEvolve Run Summary",
        f"  Run:        {run_dir.name}",
        f"  JSONL:      {jsonl_path.name}",
        f"  Checkpoint: {ckpt.name}  ({len(programs)} programs)",
        f"  Iterations: {len(iterations)}", sep(),
    ]

    best_prog = max(programs.values(), key=lambda p: p.get("metrics", {}).get("combined_score", 0))
    lines += [
        "", "OVERALL BEST PROGRAM", sep("-"),
        f"  ID:        {best_prog['id']}",
        f"  Iteration: {best_prog.get('iteration_found', '?')}",
        f"  Parent:    {best_prog.get('parent_id', 'none')}",
        "  Metrics:", fmt_metrics(best_prog.get("metrics")), "",
    ]

    lines += [sep(), "  ITERATION LOG", sep()]

    for record in iterations:
        it_num   = record["iteration"]
        island   = record["global"]["current_island_idx"]
        n_isl    = record["global"]["num_islands"]
        result   = record["iteration_result"]
        success  = result["success"]
        child_info = result.get("child_program") or {}
        child_id   = child_info.get("id")
        parent_id  = child_info.get("parent_id")
        metrics    = child_info.get("metrics")

        status = "OK" if success else "FAILED"
        lines += [
            "", sep("─"),
            f"  ITERATION {it_num:>3}  [{status}]  Island {island+1}/{n_isl}"
            f"  child={short_id(child_id)}  parent={short_id(parent_id)}",
            sep("─"),
        ]

        if not success:
            lines.append(f"  Error: {result.get('error','unknown')}")
            continue

        lines += ["", "  Metrics:", fmt_metrics(metrics)]

        prog   = programs.get(child_id)
        parent = programs.get(parent_id)

        lines += ["", "  PARENT PROGRAM", f"  ID: {parent_id}"]
        if parent:
            lines += ["  Metrics:", fmt_metrics(parent.get("metrics")), "", "  Code:", fmt_code(parent.get("solution",""))]
        else:
            lines.append("  (parent not in checkpoint)")

        if prog:
            prompts   = prog.get("prompts", {}).get("diff_user_message", {})
            system    = prompts.get("system", "")
            user      = prompts.get("user", "")
            responses = prompts.get("responses", [])

            lines += ["", "  LLM SYSTEM PROMPT", "  " + sep("-", 76)]
            lines += ["  " + ln for ln in system.splitlines()]
            lines += ["", "  LLM USER PROMPT", "  " + sep("-", 76)]
            lines += ["  " + ln for ln in user.splitlines()]
            if responses:
                lines += ["", "  LLM RESPONSE", "  " + sep("-", 76)]
                lines += ["  " + ln for ln in responses[0].splitlines()]

            lines += [
                "", "  RESULTING CHILD PROGRAM", "  " + sep("-", 76),
                f"  ID:         {child_id}",
                f"  Generation: {prog.get('generation','?')}",
            ]
            if prog.get("metadata", {}).get("changes"):
                lines.append(f"  Changes:    {prog['metadata']['changes']}")
            lines += ["", "  Code:", fmt_code(prog.get("solution",""))]

    lines += ["", sep(), "  END OF RUN", sep()]
    output = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print(f"Written to {output_path}")
    else:
        print(output)


# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

HTML_STYLE = """
<style>
  :root {
    --bg:      #0f1117;
    --surface: #1a1f2e;
    --border:  #2d3748;
    --text:    #e2e8f0;
    --muted:   #94a3b8;
    --green:   #34d399;
    --red:     #f87171;
    --blue:    #60a5fa;
    --purple:  #a78bfa;
    --yellow:  #fbbf24;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: system-ui, sans-serif;
         font-size: 14px; line-height: 1.6; padding: 24px; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  h2 { font-size: 1.1rem; color: var(--blue); margin: 20px 0 8px; }
  .meta-table { border-collapse: collapse; margin-bottom: 20px; }
  .meta-table td { padding: 2px 16px 2px 0; color: var(--muted); }
  .meta-table td:first-child { color: var(--text); font-weight: 600; min-width: 120px; }

  .best-box { background: var(--surface); border: 1px solid var(--border);
              border-left: 3px solid var(--yellow); border-radius: 6px;
              padding: 14px 18px; margin-bottom: 24px; }
  .best-box .label { font-size: 0.75rem; text-transform: uppercase;
                     letter-spacing: .08em; color: var(--yellow); margin-bottom: 6px; }

  details { background: var(--surface); border: 1px solid var(--border);
            border-radius: 6px; margin-bottom: 8px; overflow: hidden; }
  details[open] { border-color: var(--blue); }
  summary { padding: 10px 16px; cursor: pointer; list-style: none;
            display: flex; align-items: center; gap: 12px; user-select: none; }
  summary::-webkit-details-marker { display: none; }
  summary .arrow { color: var(--muted); transition: transform .15s; font-size: 12px; }
  details[open] summary .arrow { transform: rotate(90deg); }
  summary .it-num { font-weight: 700; font-size: 1rem; min-width: 32px; }
  summary .badge { padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
  summary .ok   { background: #14532d; color: var(--green); }
  summary .fail { background: #7f1d1d; color: var(--red); }
  summary .island { color: var(--muted); font-size: 0.82rem; }
  summary .score { margin-left: auto; font-size: 0.85rem; }
  summary .score .val { font-weight: 700; color: var(--green); }
  summary .ids { color: var(--muted); font-size: 0.78rem; font-family: monospace; }

  .iter-body { padding: 0 16px 16px; }

  .section-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .08em;
                   color: var(--muted); margin: 14px 0 6px; border-bottom: 1px solid var(--border);
                   padding-bottom: 4px; }
  .metrics-grid { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
  .metric-chip { background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
                 padding: 3px 10px; font-size: 0.82rem; }
  .metric-chip .mk { color: var(--muted); margin-right: 4px; }
  .metric-chip .mv { font-weight: 600; }
  .mv.good { color: var(--green); }
  .mv.bad  { color: var(--red); }
  .mv.mid  { color: var(--yellow); }

  details.sub { background: var(--bg); border-color: var(--border); margin: 8px 0 0; }
  details.sub summary { padding: 7px 12px; font-size: 0.85rem; color: var(--muted); }
  details.sub[open] { border-color: var(--purple); }

  pre { background: #0a0d14; border: 1px solid var(--border); border-radius: 4px;
        padding: 12px 14px; overflow-x: auto; font-size: 0.8rem; line-height: 1.5;
        font-family: 'Fira Code', 'Cascadia Code', 'Consolas', monospace;
        white-space: pre; margin: 6px 0; }
  .prompt-text { background: #0a0d14; border: 1px solid var(--border); border-radius: 4px;
                 padding: 12px 14px; font-size: 0.82rem; line-height: 1.55;
                 font-family: inherit; white-space: pre-wrap; word-break: break-word;
                 margin: 6px 0; max-height: 400px; overflow-y: auto; }

  .tag-parent { color: var(--purple); }
  .tag-child  { color: var(--blue); }
  .improvement { color: var(--green); font-weight: 600; }
  .regression  { color: var(--red);   font-weight: 600; }
</style>
"""

def score_class(score):
    if score is None:
        return ""
    if score >= 0.55:
        return "good"
    if score >= 0.5:
        return "mid"
    return "bad"


def metric_chips(metrics):
    if not metrics:
        return "<span style='color:var(--muted)'>no metrics</span>"
    chips = []
    for k, v in metrics.items():
        if isinstance(v, float):
            cls = score_class(v) if "score" in k else ""
            chips.append(
                f'<span class="metric-chip"><span class="mk">{html_mod.escape(k)}</span>'
                f'<span class="mv {cls}">{v:.4f}</span></span>'
            )
        else:
            chips.append(
                f'<span class="metric-chip"><span class="mk">{html_mod.escape(k)}</span>'
                f'<span class="mv">{html_mod.escape(str(v))}</span></span>'
            )
    return '<div class="metrics-grid">' + "".join(chips) + "</div>"


def code_block(code):
    if not code:
        return "<pre>(no code)</pre>"
    return f"<pre>{html_mod.escape(code)}</pre>"


def prompt_block(text):
    if not text:
        return ""
    return f'<div class="prompt-text">{html_mod.escape(text)}</div>'


def extract_html(run_dir_str, output_path, watch_interval=None):
    from datetime import datetime
    run_dir, iterations, programs, ckpt, jsonl_path = load_run(run_dir_str)

    best_prog = max(programs.values(), key=lambda p: p.get("metrics", {}).get("combined_score", 0))
    best_score = best_prog.get("metrics", {}).get("combined_score", 0)

    auto_refresh = f'<meta http-equiv="refresh" content="{watch_interval}">' if watch_interval else ""
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {auto_refresh}
  <title>AdaEvolve Run — {html_mod.escape(run_dir.name)}</title>
  {HTML_STYLE}
</head>
<body>
""")

    # Header
    parts.append(f"""
<h1>AdaEvolve Run — {html_mod.escape(run_dir.name)}</h1>
<table class="meta-table">
  <tr><td>JSONL</td><td>{html_mod.escape(jsonl_path.name)}</td></tr>
  <tr><td>Checkpoint</td><td>{html_mod.escape(ckpt.name)} &nbsp;({len(programs)} programs)</td></tr>
  <tr><td>Iterations</td><td>{len(iterations)}</td></tr>
  <tr><td>Last updated</td><td>{last_updated}{' &nbsp;(auto-refresh ' + str(watch_interval) + 's)' if watch_interval else ''}</td></tr>
</table>
""")

    # Best program
    bm = best_prog.get("metrics", {})
    parts.append(f"""
<h2>Overall Best Program</h2>
<div class="best-box">
  <div class="label">&#9733; Best found at iteration {best_prog.get('iteration_found','?')}</div>
  <div style="font-family:monospace;font-size:0.82rem;color:var(--muted);margin-bottom:8px">
    ID: {html_mod.escape(best_prog['id'])} &nbsp;|&nbsp;
    Parent: {html_mod.escape(best_prog.get('parent_id','none'))}
  </div>
  {metric_chips(bm)}
  <details class="sub" style="margin-top:10px">
    <summary><span class="arrow">&#9654;</span> View code</summary>
    {code_block(best_prog.get('solution',''))}
  </details>
</div>
""")

    # Iterations
    parts.append("<h2>Iteration Log</h2>")

    for record in iterations:
        it_num    = record["iteration"]
        island    = record["global"]["current_island_idx"]
        n_isl     = record["global"]["num_islands"]
        result    = record["iteration_result"]
        success   = result["success"]
        child_info = result.get("child_program") or {}
        child_id   = child_info.get("id")
        parent_id  = child_info.get("parent_id")
        metrics    = child_info.get("metrics") or {}

        combined = metrics.get("combined_score")
        score_str = f'<span class="val">{combined:.4f}</span>' if combined is not None else "—"

        # Delta vs parent
        prog   = programs.get(child_id)
        parent = programs.get(parent_id)
        delta_str = ""
        if combined is not None and parent:
            parent_score = parent.get("metrics", {}).get("combined_score")
            if parent_score is not None:
                delta = combined - parent_score
                cls   = "improvement" if delta >= 0 else "regression"
                sign  = "+" if delta >= 0 else ""
                delta_str = f' <span class="{cls}">({sign}{delta:.4f})</span>'

        status_badge = f'<span class="badge {"ok" if success else "fail"}">{"OK" if success else "FAILED"}</span>'

        parts.append(f"""
<details>
  <summary>
    <span class="arrow">&#9654;</span>
    <span class="it-num">#{it_num}</span>
    {status_badge}
    <span class="island">Island {island+1}/{n_isl}</span>
    <span class="ids">child={html_mod.escape(short_id(child_id))} &nbsp; parent={html_mod.escape(short_id(parent_id))}</span>
    <span class="score">combined {score_str}{delta_str}</span>
  </summary>
  <div class="iter-body">
""")

        if not success:
            parts.append(f'<p style="color:var(--red)">Error: {html_mod.escape(result.get("error","unknown"))}</p>')
            parts.append("</div></details>")
            continue

        # Metrics
        parts.append('<div class="section-title">Metrics</div>')
        parts.append(metric_chips(metrics))

        # Parent
        parts.append('<div class="section-title">Parent Program</div>')
        parts.append(f'<div style="font-family:monospace;font-size:0.78rem;color:var(--muted);margin-bottom:6px">'
                     f'<span class="tag-parent">ID: {html_mod.escape(parent_id or "")}</span></div>')
        if parent:
            parts.append(metric_chips(parent.get("metrics")))
            parts.append(f"""
  <details class="sub">
    <summary><span class="arrow">&#9654;</span> View parent code</summary>
    {code_block(parent.get('solution',''))}
  </details>""")
        else:
            parts.append('<p style="color:var(--muted);font-size:0.82rem">Parent not retained in checkpoint.</p>')

        # Prompts & response
        if prog:
            prompts   = prog.get("prompts", {}).get("diff_user_message", {})
            system    = prompts.get("system", "")
            user      = prompts.get("user", "")
            responses = prompts.get("responses", [])

            parts.append('<div class="section-title">LLM Interaction</div>')
            parts.append(f"""
  <details class="sub">
    <summary><span class="arrow">&#9654;</span> System prompt</summary>
    {prompt_block(system)}
  </details>
  <details class="sub">
    <summary><span class="arrow">&#9654;</span> User prompt</summary>
    {prompt_block(user)}
  </details>""")
            if responses:
                parts.append(f"""
  <details class="sub">
    <summary><span class="arrow">&#9654;</span> LLM response</summary>
    {prompt_block(responses[0])}
  </details>""")

            # Child
            parts.append('<div class="section-title">Resulting Child Program</div>')
            parts.append(f'<div style="font-family:monospace;font-size:0.78rem;color:var(--muted);margin-bottom:6px">'
                         f'<span class="tag-child">ID: {html_mod.escape(child_id or "")}</span> &nbsp; '
                         f'generation {prog.get("generation","?")} </div>')
            if prog.get("metadata", {}).get("changes"):
                parts.append(f'<p style="font-size:0.82rem;color:var(--muted);margin-bottom:6px">'
                              f'Changes: {html_mod.escape(prog["metadata"]["changes"])}</p>')
            parts.append(code_block(prog.get("solution", "")))

        parts.append("</div></details>")

    parts.append("</body></html>")

    with open(output_path, "w") as f:
        f.write("\n".join(parts))
    print(f"Written to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Extract AdaEvolve run summary")
    parser.add_argument("run_dir", help="Path to the run output directory")
    parser.add_argument("output", nargs="?", help="Output file (.html or .txt)")
    parser.add_argument("--watch", type=int, metavar="SECONDS", default=None,
                        help="Regenerate every N seconds (HTML only)")
    args = parser.parse_args()

    if args.watch and (not args.output or not args.output.endswith(".html")):
        print("--watch requires an .html output file")
        sys.exit(1)

    if args.watch:
        print(f"Watching {args.run_dir} — regenerating every {args.watch}s. Ctrl+C to stop.")
        while True:
            try:
                extract_html(args.run_dir, args.output, watch_interval=args.watch)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Error: {e}")
            time.sleep(args.watch)
    elif args.output and args.output.endswith(".html"):
        extract_html(args.run_dir, args.output)
    else:
        extract_text(args.run_dir, args.output)
