"""
extract_champion.py — pull the archive-max program (by combined_score) from
an AdaEvolve run's latest checkpoint and write it out as a standalone .py.

Deliberately does NOT trust best_program_info.json: for multiobjective runs
that file names a Pareto/fitness-proxy pick, not the highest combined_score
in the archive (see memory project-champion-programs, verified 2026-07-29 —
p1B's best_program_info undershot the true archive-max by 0.0061).

Usage:
    python extract_champion.py <run_dir> <output_dir> [--prefix NAME]
"""

import argparse
import json
from pathlib import Path


def latest_checkpoint(run_dir: Path) -> Path:
    """Sorted by mtime, NOT the numeric suffix -- a run that gets relaunched
    reuses the same checkpoints/ dir, and an orphaned checkpoint from an
    earlier, superseded attempt can carry a higher number than the true
    final state (verified 2026-08-11: p1mech04_homogeneous_r2's
    checkpoint_200 is an 8 Aug leftover; the real final state, matching the
    logged FINISHED-at-144, is checkpoint_140 from 10 Aug)."""
    ckpt_dir = run_dir / "checkpoints"
    checkpoints = sorted(
        (p for p in ckpt_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint_")),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {ckpt_dir}")
    return checkpoints[-1]


def champion(ckpt_dir: Path) -> dict:
    best = None
    for prog_path in (ckpt_dir / "programs").glob("*.json"):
        prog = json.loads(prog_path.read_text())
        score = prog.get("metrics", {}).get("combined_score")
        if score is None:
            continue
        if best is None or score > best["metrics"]["combined_score"]:
            best = prog
    if best is None:
        raise ValueError(f"No scored programs found under {ckpt_dir}/programs")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--prefix", default=None, help="Defaults to the run dir's name")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    prefix = args.prefix or run_dir.name
    ckpt = latest_checkpoint(run_dir)
    champ = champion(ckpt)
    m = champ["metrics"]
    short_id = champ["id"][:8]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{prefix}_champ_{short_id}.py"
    out_path.write_text(champ["solution"])

    print(f"{prefix}: checkpoint={ckpt.name} id={short_id} "
          f"combined_score={m['combined_score']:.4f} q10={m.get('q10_score')} "
          f"fraction_positive={m.get('fraction_positive')} -> {out_path}")


if __name__ == "__main__":
    main()
