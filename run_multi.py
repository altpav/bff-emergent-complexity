"""
https://arxiv.org/abs/2406.19108
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import time


def run_one_seed(seed: int, args_dict: dict) -> tuple[int, int, str]:
    cmd = [
        sys.executable,
        "run_fast.py",
        "--epochs",
        str(args_dict["epochs"]),
        "--population",
        str(args_dict["population"]),
        "--sample-every",
        str(args_dict["sample_every"]),
        "--max-steps",
        str(args_dict["max_steps"]),
        "--batch",
        str(args_dict["batch"]),
        "--output-dir",
        str(args_dict["output_dir"]),
        "--seed",
        str(seed),
    ]

    if args_dict.get("quiet"):
        cmd.append("--quiet")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args_dict["threads_per_run"])

    result = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return seed, result.returncode, result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple BFF seeds in parallel")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=16000)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--start-seed", type=int, default=1)
    parser.add_argument("--parallel", type=int, default=None)
    parser.add_argument("--quiet", action="store_true", help="Suppress file output")
    args = parser.parse_args()

    n_cpus = os.cpu_count() or 1
    parallel = args.parallel or n_cpus
    threads_per_run = max(1, n_cpus // parallel)

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_dir = Path("results") / ts
        out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BFF EMERGENT COMPLEXITY - MULTI-RUN")
    print("=" * 70)
    print(f"Runs:          {args.runs}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Population:    {args.population}")
    print(f"Max steps:     {args.max_steps}")
    print(f"Seeds:         {args.start_seed} to {args.start_seed + args.runs - 1}")
    print(f"Parallel:      {parallel} processes")
    print(f"Output dir:    {out_dir}")
    print("=" * 70)
    print()

    seeds_to_run = []
    for i in range(args.runs):
        seed = args.start_seed + i
        npz = out_dir / f"run_s{seed}.npz"
        if npz.exists() and not args.quiet:
            print(f"  Seed {seed}: already exists, skipping")
        else:
            seeds_to_run.append(seed)

    if not seeds_to_run:
        print("All runs already exist. Use --quiet to force re-run.")
        return

    print(f"Running {len(seeds_to_run)} seed(s)...")
    print("-" * 70)

    args_dict = {
        "epochs": args.epochs,
        "population": args.population,
        "sample_every": args.sample_every,
        "max_steps": args.max_steps,
        "batch": args.batch,
        "output_dir": str(out_dir),
        "threads_per_run": threads_per_run,
        "quiet": args.quiet,
    }

    t_start = time.time()
    completed = 0
    phase_transitions = 0

    with ProcessPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(run_one_seed, seed, args_dict): seed for seed in seeds_to_run
        }

        for future in as_completed(futures):
            seed, rc, output = future.result()
            completed += 1

            if rc == 0:
                if "PHASE TRANSITION DETECTED" in output:
                    phase_transitions += 1
                    status = "done (PHASE TRANSITION!)"
                else:
                    status = "done"
            else:
                status = f"FAILED (exit {rc})"

            print(f"\n[Seed {seed}] {status}")

            lines = output.strip().split("\n")
            for line in lines:
                if (
                    "PHASE TRANSITION" in line
                    or "SUMMARY" in line
                    or "Final tokens:" in line
                ):
                    print(f"    {line}")

            elapsed = time.time() - t_start
            rate = completed / elapsed if elapsed > 0 else 0
            print(f"    Progress: {completed}/{len(seeds_to_run)} ({rate:.2f} runs/s)")

    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("MULTI-RUN COMPLETE")
    print("=" * 70)
    print(f"Total runs:        {completed}")
    print(
        f"Phase transitions: {phase_transitions} ({phase_transitions / completed * 100:.0f}%)"
    )
    print(f"Total time:        {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)

    if not args.quiet:
        print(f"\nPlot with: python plot_results.py --input-dir {out_dir}")


if __name__ == "__main__":
    main()
