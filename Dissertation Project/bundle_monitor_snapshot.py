"""
bundle_monitor_snapshot.py  —  Save a self-contained monitor snapshot for a completed run.

Reads checkpoint data from a run directory and produces a standalone HTML file
(dashboard + data injected) that works without any server. Open directly in a browser.

Usage:
    python bundle_monitor_snapshot.py <run_dir> [output_file]

Examples:
    python bundle_monitor_snapshot.py /root/skydiscover/outputs/adaevolve/fjsp_twt_0611_1506
    python bundle_monitor_snapshot.py /root/skydiscover/outputs/adaevolve/fjsp_twt_0611_1506 run_summaries/fjsp_twt_0611_1506_monitor.html
"""

import json
import sys
from pathlib import Path


DASHBOARD_PATH = Path(__file__).parent.parent / "skydiscover/extras/monitor/dashboard.html"
PLOTLY_PATH = Path(__file__).parent.parent / ".venv/lib/python3.12/site-packages/plotly/package_data/plotly.min.js"
PLOTLY_CDN_TAG = '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'

# Injected just before </script> — checks for __INIT_STATE__ before opening WebSocket
STATIC_PATCH = """
  // ── Static snapshot patch ─────────────────────────────────────
  if (window.__INIT_STATE__) {
    const badge = document.getElementById('status-badge');
    if (badge) { badge.className = 'status-badge connected'; badge.textContent = 'Replay'; }
    handleMessage(window.__INIT_STATE__);
  } else {
    connect();
  }
"""


def find_last_checkpoint(run_dir: Path) -> Path:
    ckpts_dir = run_dir / "checkpoints"
    if not ckpts_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints/ found in {run_dir}")
    ckpts = sorted(ckpts_dir.glob("checkpoint_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint_N dirs found in {ckpts_dir}")
    return ckpts[-1]


def load_programs(ckpt_dir: Path):
    programs = {}
    best_program_id = None
    last_iteration = 0

    meta_path = ckpt_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        best_program_id = meta.get("best_program_id")
        last_iteration = meta.get("last_iteration", 0)

    programs_dir = ckpt_dir / "programs"
    for jf in programs_dir.glob("*.json"):
        try:
            data = json.loads(jf.read_text())
            programs[data["id"]] = data
        except Exception:
            continue

    if not best_program_id and programs:
        best_score = -float("inf")
        for pid, prog in programs.items():
            s = (prog.get("metrics") or {}).get("combined_score", 0)
            if isinstance(s, (int, float)) and s > best_score:
                best_score = s
                best_program_id = pid

    return programs, best_program_id, last_iteration


def to_monitor_format(prog: dict, all_progs: dict) -> dict:
    metrics = prog.get("metrics") or {}
    score = metrics.get("combined_score", 0.0)
    if not isinstance(score, (int, float)):
        score = 0.0

    parent_id = prog.get("parent_id")
    parent_score = None
    parent_iter = None
    if parent_id and parent_id in all_progs:
        pm = all_progs[parent_id].get("metrics") or {}
        parent_score = pm.get("combined_score")
        parent_iter = all_progs[parent_id].get("iteration_found")

    context_ids = prog.get("other_context_ids") or []
    context_scores = []
    for cid in context_ids:
        if cid in all_progs:
            cm = all_progs[cid].get("metrics") or {}
            context_scores.append(cm.get("combined_score"))
        else:
            context_scores.append(None)

    label_type = "unknown"
    pi = prog.get("parent_info")
    if pi and isinstance(pi, (list, tuple)) and len(pi) >= 1:
        ls = str(pi[0]).lower()
        if "diverge" in ls:
            label_type = "diverge"
        elif "refine" in ls:
            label_type = "refine"
        elif "crossover" in ls:
            label_type = "crossover"
    if label_type == "unknown":
        label_type = (prog.get("metadata") or {}).get("label_type", "unknown")

    safe_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            safe_metrics[k] = v

    return {
        "id": prog["id"],
        "iteration": prog.get("iteration_found", 0),
        "score": score,
        "metrics": safe_metrics,
        "parent_id": parent_id,
        "parent_score": parent_score,
        "parent_iter": parent_iter,
        "context_ids": context_ids,
        "context_scores": context_scores,
        "label_type": label_type,
        "island": (prog.get("metadata") or {}).get("island"),
        "generation": prog.get("generation", 0),
        "image_path": (prog.get("metadata") or {}).get("image_path"),
        "solution_snippet": (prog.get("solution") or "")[:500],
        "full_solution": (prog.get("solution") or ""),
    }


def build_init_state(programs: dict, best_program_id: str, last_iteration: int) -> dict:
    monitor_programs = [
        to_monitor_format(p, programs)
        for p in sorted(programs.values(), key=lambda x: x.get("iteration_found", 0))
    ]

    best_score = max(
        (p["score"] for p in monitor_programs if isinstance(p["score"], (int, float))),
        default=0.0,
    )

    return {
        "type": "init_state",
        "programs": monitor_programs,
        "best_program_id": best_program_id,
        "stats": {
            "total_programs": len(monitor_programs),
            "current_iteration": last_iteration,
            "best_score": best_score,
            "iterations_since_improvement": 0,
            "programs_per_min": 0,
            "elapsed_seconds": 0,
        },
        "summary_enabled": False,
        "summary_model": "",
        "summary_text": "",
        "summary_generating": False,
        "human_feedback_enabled": False,
        "feedback_active": False,
        "feedback_text": "",
        "human_feedback_mode": "off",
        "human_feedback_current_prompt": "",
        "human_feedback_history": [],
    }


def bundle(run_dir: Path, output_path: Path) -> None:
    ckpt_dir = find_last_checkpoint(run_dir)
    print(f"Loading from: {ckpt_dir}")

    programs, best_program_id, last_iteration = load_programs(ckpt_dir)
    if not programs:
        raise RuntimeError(f"No programs found in {ckpt_dir}")
    print(f"Loaded {len(programs)} programs (best={best_program_id}, last_iter={last_iteration})")

    init_state = build_init_state(programs, best_program_id, last_iteration)

    dashboard_html = DASHBOARD_PATH.read_text(encoding="utf-8")

    # Inline Plotly so the file opens instantly without any network requests
    if PLOTLY_PATH.exists():
        plotly_js = PLOTLY_PATH.read_text(encoding="utf-8")
        dashboard_html = dashboard_html.replace(
            PLOTLY_CDN_TAG,
            f"<script>\n{plotly_js}\n</script>",
            1,
        )
        print("Inlined Plotly from local package")
    else:
        print("Warning: local Plotly not found, keeping CDN link")

    # Inject __INIT_STATE__ as a script block before the main <script>
    state_json = json.dumps(init_state)
    state_script = f"<script>\nwindow.__INIT_STATE__ = {state_json};\n</script>\n"
    dashboard_html = dashboard_html.replace("<script>\n(function() {", state_script + "<script>\n(function() {", 1)

    # Replace `connect();` at the bottom with the static patch (skip the WS call)
    dashboard_html = dashboard_html.replace("  rebuildPlot();\n  connect();\n", f"  rebuildPlot();\n{STATIC_PATCH}\n", 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dashboard_html, encoding="utf-8")
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        run_name = run_dir.name
        output_path = Path(__file__).parent / "run_summaries" / f"{run_name}_monitor.html"

    bundle(run_dir, output_path)


if __name__ == "__main__":
    main()
