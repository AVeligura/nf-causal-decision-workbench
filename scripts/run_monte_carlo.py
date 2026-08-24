from __future__ import annotations

import argparse
from pathlib import Path

from study.monte_carlo import FULL_DESIGN, DesignCell, MonteCarloRunner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--experiment-id", default="r3-20260820")
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="prepare 500 repeats × 10 cells")
    parser.add_argument(
        "--authorize-full",
        action="store_true",
        help="required after a separate coordinator authorization",
    )
    parser.add_argument("--compute-cate", action="store_true")
    parser.add_argument("--cate-trees", type=int, default=300)
    parser.add_argument("--single-cell", choices=[cell.scenario for cell in FULL_DESIGN])
    args = parser.parse_args()
    if args.full and not args.authorize_full:
        parser.error(
            "Полный эксперимент заблокирован до отдельного решения после R3; "
            "после разрешения передайте --authorize-full"
        )
    design = FULL_DESIGN
    if args.single_cell:
        design = (DesignCell(args.single_cell, 1500),)
    runner = MonteCarloRunner(args.root, args.experiment_id)
    total = (
        runner.prepare(500, design)
        if args.full
        else runner.prepare_r3(design, repetitions_per_subcell=args.repetitions)
    )
    print(f"Prepared {total} design rows; current status {runner.status_counts()}")

    def progress(counts, eta):
        completed = counts.get("completed", 0) + counts.get("failed", 0)
        if completed % 50 == 0 or completed == total:
            print(f"{counts}; ETA {eta / 60:.1f} min", flush=True)

    print(
        runner.run(
            workers=args.workers,
            compute_cate=args.compute_cate,
            cate_trees=args.cate_trees,
            progress=progress,
        )
    )
    print(runner.export())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
