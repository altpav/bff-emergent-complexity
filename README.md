# BFF Emergent Complexity

A replicable implementation of the **Computational Life** experiment by Blaise Agüera y Arcas et al., where self-replicating programs spontaneously emerge from random noise with no fitness function.

For the blog post referring to this repo, please visit <https://jonamiki.com/posts/bff-emergent-complexity-experiment/>

**Paper:** [Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction](https://arxiv.org/abs/2406.19108) (arXiv:2406.19108)

## What This Does

A population of 1,024 small programs (64 bytes each) is initialized with random bytes. Each "interaction" picks two programs, concatenates them into a 128-byte tape, and runs them as Extended Brainfuck (BFF). The tape is then split back into two programs which replace the originals.

After enough interactions, **self-replicating programs emerge** — programs that copy themselves onto their interaction partner. This creates a sharp phase transition visible as:

- A sudden spike in **computational intensity** (operations per interaction)
- A collapse in **unique tokens** (program ancestry diversity)
- A drop in **compressibility** (the soup becomes structured rather than random)

## Setup

Requires Python 3.10+, a C compiler with OpenMP support, and the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Compile the C engine:

```bash
gcc -O3 -march=native -fopenmp -shared -fPIC -o bff_engine.so bff_engine.c
```

## Quick Start

Run a single experiment:

```bash
python run_fast.py --epochs 6000 --seed 42
```

Run multiple experiments in parallel:

```bash
python run_multi.py --runs 5 --epochs 50000
```

Results are saved to timestamped folders under `results/`.

## Plotting

Generate static plots (matplotlib):

```bash
python plot_results.py --input-dir results/YYYY-MM-DD_HH-MM
```


## Files

| File | Description |
|------|-------------|
| `bff_engine.c` | C implementation of the BFF interpreter and simulation loop (OpenMP parallelized) |
| `engine.py` | Python ctypes wrapper around the C engine |
| `run_fast.py` | Run a single experiment with one seed |
| `run_multi.py` | Run multiple experiments in parallel with different seeds |
| `plot_results.py` | Generate static matplotlib plots from saved run data |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 6,000 (single) / 16,000 (multi) | Number of epochs (1 epoch = population_size interactions) |
| `--population` | 1,024 | Number of programs in the soup |
| `--max-steps` | 16,384 | Max BFF execution steps per interaction (2^14) |
| `--seed` | random | Random seed for reproducibility |
| `--runs` | 10 | Number of independent runs (multi mode) |
| `--parallel` | CPU count | Max concurrent processes |

For a clear phase transition, use at least 30,000–50,000 epochs. The paper reports ~12% of runs transition by 2,000 epochs, ~32% between 2,000–16,000, and ~56% haven't transitioned by 16,000.

## References

- Agüera y Arcas et al., *Computational Life: How Well-formed, Self-replicating Programs Emerge from Simple Interaction*, arXiv:2406.19108 (2024)
- [Reproducing the results of a computational life experiment](https://blog.ricky0123.com/blog/complexity_v2/) (ricky0123)
