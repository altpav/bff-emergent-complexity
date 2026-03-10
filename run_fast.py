"""
https://arxiv.org/abs/2406.19108
"""

import argparse
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
from engine import Population


def main() -> None:
    parser = argparse.ArgumentParser(description="BFF Emergent Complexity")
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=16384)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--quiet", action="store_true", help="Suppress the file output")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2**31)

    if args.output_dir and not args.quiet:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    pop = Population(size=args.population, seed=seed, max_steps=args.max_steps)

    print("=" * 70)
    print("BFF EMERGENT COMPLEXITY EXPERIMENT")
    print("=" * 70)
    print(f"Seed:        {seed}")
    print(f"Population:  {pop.size}")
    print(f"Epochs:      {args.epochs}")
    print(f"Max steps:   {args.max_steps}")
    print(f"Total interactions: {pop.size * args.epochs:,}")
    if out_dir:
        print(f"Output dir:  {out_dir}")
    print("=" * 70)

    if out_dir:
        init_dump = pop.dump_programs(top_n=30)
        init_path = out_dir / f"tape_initial_s{seed}.txt"
        init_path.write_text(init_dump)

    print("\n---- Initial Soup (random programs) ------")
    print(pop.dump_programs(top_n=20))
    print()

    sample_epochs = []
    hoe_list = []
    tokens_list = []
    compressibility_list = []
    all_steps_chunks = []

    print("\nRunning - (Epoch | HOE | Tokens | Compress | Mean ops/s)")
    print("-" * 70)
    sys.stdout.flush()

    def sample_metrics(epoch_num: int, mean_ops: float = 0):
        data = pop.get_values()
        compressed = zlib.compress(data, level=9)
        compress_ratio = len(compressed) / len(data)
        hoe = pop.higher_order_entropy()
        tokens = pop.unique_tokens()
        sample_epochs.append(epoch_num)
        hoe_list.append(hoe)
        tokens_list.append(tokens)
        compressibility_list.append(compress_ratio)

        now = time.monotonic()
        rate = epoch_num / (now - t_start) if epoch_num > 0 else 0

        print(
            f"  {epoch_num:5d}  | {hoe:.4f} | {tokens:5d} | {compress_ratio:.4f}  | {mean_ops:.0f} ({rate:.0f} ep/s)"
        )
        sys.stdout.flush()

        return hoe, tokens, compress_ratio

    epoch = 0
    t_start = time.monotonic()

    hoe, tokens, cr = sample_metrics(0)

    while epoch < args.epochs:
        chunk = min(args.batch, args.epochs - epoch)
        steps_block = pop.run_epochs(chunk)
        all_steps_chunks.append(steps_block)
        epoch += chunk

        if epoch % args.sample_every == 0 or epoch >= args.epochs:
            hoe, tokens, cr = sample_metrics(epoch, steps_block.mean())

    elapsed = time.monotonic() - t_start
    all_steps = np.concatenate(all_steps_chunks, axis=0)

    npz_path = None
    if out_dir:
        npz_path = out_dir / f"run_s{seed}.npz"
        np.savez_compressed(
            npz_path,
            seed=seed,
            population=args.population,
            max_steps=args.max_steps,
            steps=all_steps,
            sample_epochs=np.array(sample_epochs),
            hoe=np.array(hoe_list),
            tokens=np.array(tokens_list, dtype=np.int64),
            compressibility=np.array(compressibility_list),
        )

    print("-" * 70)
    print(f"Completed in {elapsed:.1f}s ({args.epochs / elapsed:.1f} epochs/s)")

    if npz_path:
        print(f"Data saved {npz_path}")

    print("\n" + "=" * 70)
    print("FINAL - Programs after evolution")
    print("=" * 70)
    dump = pop.dump_programs(top_n=40)
    print(dump)

    if out_dir:
        dump_path = out_dir / f"tape_final_s{seed}.txt"
        dump_path.write_text(dump)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Initial tokens: 65536")
    print(f"Final tokens:   {tokens}")
    print(f"HOE change:     {hoe:.4f}")
    print(f"Compress:       {cr:.4f}")
    if tokens < 1000:
        print("\n*** PHASE TRANSITION DETECTED ***")
        print(f"Population collapsed from 65536 to {tokens} unique tokens!")
    print("=" * 70)


if __name__ == "__main__":
    main()
