"""
Run a sweep of HPO methods over the continuous benchmark with a pool of workers.

Usage (from the repository root):

    python run_sweep.py --config configs/sweep_5971.json --out sweep_runs/my_run --workers 4

Add --resume to continue a sweep whose output directory already has finished runs.
"""

import argparse

from sweep.config import load_config
from sweep.coordinator import run_sweep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--config", required=True, help="Path to the sweep configuration (JSON)")
    parser.add_argument("--out", required=True, help="Output directory of the sweep")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--resume", action="store_true", help="Continue a sweep that was interrupted")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be at least 1")

    config = load_config(args.config)
    run_sweep(config, args.out, n_workers=args.workers, resume=args.resume)


if __name__ == "__main__":
    main()
