"""
build_comparison.py — Build the multi-run AdaEvolve comparison dashboard.

Usage:
    python build_comparison.py [run_dir ...] [--output PATH] [--include-full-code] [--no-cache]

With no run_dir arguments, defaults to the 5 Jul 14 batch runs
(fjsp_twt_0714_{1339,1341,1342,1406,1418}).

Per-run bundles are cached under run_summaries/.cache/, keyed by a fingerprint
of the run's log file, JSONL file, and checkpoint-directory listing -- an
unchanged run is not re-parsed on rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compare_dashboard import bundle as bundle_mod
from compare_dashboard import html_utils

DEFAULT_RUN_NAMES = [
    "fjsp_twt_0714_1339",
    "fjsp_twt_0714_1341",
    "fjsp_twt_0714_1342",
    "fjsp_twt_0714_1406",
    "fjsp_twt_0714_1418",
]
OUTPUTS_ROOT = Path("/root/skydiscover/outputs/adaevolve")
# build_comparison.py -> compare_dashboard -> tools -> skydiscover (parents[2])
RUN_SUMMARIES_DIR = Path(__file__).resolve().parents[2] / "run_summaries"
CACHE_DIR = RUN_SUMMARIES_DIR / ".cache"


def _fingerprint(run_dir: Path) -> str:
    """Cheap fingerprint of a run's inputs: log/JSONL sizes+mtimes plus the
    sorted checkpoint directory listing (catches new checkpoints appearing,
    e.g. if a run was still in progress on a previous build)."""
    parts = []
    for pattern in ("logs/adaevolve_*.log", "adaevolve_iteration_stats_*.jsonl"):
        for p in sorted(run_dir.glob(pattern)):
            st = p.stat()
            parts.append(f"{p.name}:{st.st_size}:{st.st_mtime}")
    ckpts_dir = run_dir / "checkpoints"
    if ckpts_dir.is_dir():
        parts.append(",".join(sorted(p.name for p in ckpts_dir.glob("checkpoint_*"))))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def build_run_bundle_cached(run_dir: Path, include_full_code: bool, use_cache: bool) -> dict:
    run_name = run_dir.name
    fp = _fingerprint(run_dir)
    cache_path = CACHE_DIR / f"{run_name}_bundle_v{bundle_mod.SCHEMA_VERSION}.json"

    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("_fingerprint") == fp and cached.get("_include_full_code") == include_full_code:
                print(f"  [cache hit] {run_name}")
                cached.pop("_fingerprint", None)
                cached.pop("_include_full_code", None)
                return cached
        except (json.JSONDecodeError, OSError):
            pass

    print(f"  [building] {run_name} ...")
    result = bundle_mod.build_run_bundle(run_dir, include_full_code=include_full_code)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        to_store = dict(result)
        to_store["_fingerprint"] = fp
        to_store["_include_full_code"] = include_full_code
        cache_path.write_text(json.dumps(to_store))

    return result


def build(run_dirs, output_path: Path, include_full_code: bool, use_cache: bool) -> None:
    runs = {}
    for run_dir in run_dirs:
        runs[run_dir.name] = build_run_bundle_cached(run_dir, include_full_code, use_cache)

    data = {
        "schema_version": bundle_mod.SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = html_utils.build_dashboard_html(data, output_path.parent)
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path} ({len(html) / 1e6:.2f} MB)")


def resolve_run_dir(token: str) -> Path:
    """A bare run name (e.g. "fjsp_twt_0714_1339", no slash) resolves against
    OUTPUTS_ROOT. Anything that looks like a path (has a slash, or already
    exists relative to cwd) is treated as a literal path, so both
    `build_comparison.py fjsp_twt_0714_1339` and
    `build_comparison.py outputs/adaevolve/fjsp_twt_0714_1339` work."""
    p = Path(token)
    if "/" in token or p.exists():
        return p.resolve()
    return (OUTPUTS_ROOT / token).resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dirs", nargs="*",
        help="Run names or directories to include (default: Jul 14 batch). "
        "Bare names (no slash) resolve against outputs/adaevolve/. "
        "To add/remove a run from a comparison, just add/remove it from this "
        "list and rerun -- unaffected runs are served from cache, so only "
        "new/changed runs are reparsed.",
    )
    parser.add_argument("--output", type=Path, default=RUN_SUMMARIES_DIR / "comparison_0714_batch.html")
    parser.add_argument(
        "--code-preview-only", action="store_true",
        help="Store only a 500-char code_preview per program instead of full source "
        "(smaller build, but code viewing/downloading/batch-export will be truncated). "
        "Full code is included by default -- in practice it adds only ~1-2MB even for "
        "a few hundred programs, negligible next to the ~6MB of vendored Plotly.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--list", action="store_true", help="List available run directories under outputs/adaevolve/ (newest first) and exit")
    args = parser.parse_args()

    if args.list:
        candidates = sorted(
            (p for p in OUTPUTS_ROOT.iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        for p in candidates:
            print(p.name)
        return

    if args.run_dirs:
        run_dirs = [resolve_run_dir(r) for r in args.run_dirs]
    else:
        run_dirs = [OUTPUTS_ROOT / name for name in DEFAULT_RUN_NAMES]

    missing = [r for r in run_dirs if not r.is_dir()]
    if missing:
        print("Error: missing run directories:", *missing, sep="\n  ")
        sys.exit(1)

    build(run_dirs, args.output, not args.code_preview_only, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
