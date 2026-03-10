"""
https://arxiv.org/abs/2406.19108
"""

import cupy as cp
from cupy import RawKernel
import numpy as np

PROG_LEN = 64
TAPE_LEN = 128
TAPE_MOD = 256
DEFAULT_MAX_STEPS = 16384


BFF_KERNEL_SRC = """\
typedef unsigned char uint8_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define PROG_LEN 64
#define TAPE_LEN 128

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ __forceinline__ uint32_t rng_next(uint32_t* s) {
    uint32_t result = rotl32(s[1] * 5u, 7u) * 9u;
    uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1];
    s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl32(s[3], 11u);
    return result;
}

__device__ __forceinline__ void rng_seed(uint32_t* s, uint64_t seed) {
    s[0] = (uint32_t)(seed);
    s[1] = (uint32_t)(seed >> 16) ^ 0x9E3779B9u;
    s[2] = (uint32_t)(seed >> 32) ^ 0x6C62272Eu;
    s[3] = (uint32_t)(seed >> 48) ^ 0x61C88647u;
    for (int i = 0; i < 8; i++) rng_next(s);
}

__device__ int run_bff(
    uint8_t* tape_val, 
    int64_t* tape_tok, 
    int n, 
    int max_steps
) {
    int ip = 0, rh = 0, wh = 0, steps = 0;

    while (ip >= 0 && ip < n && steps < max_steps) {
        steps++;
        uint8_t cmd = tape_val[ip];

        switch (cmd) {
            case '<': rh = (rh - 1 + n) % n; ip++; break;
            case '>': rh = (rh + 1) % n; ip++; break;
            case '{': wh = (wh - 1 + n) % n; ip++; break;
            case '}': wh = (wh + 1) % n; ip++; break;
            case '-': tape_val[rh] = (tape_val[rh] - 1) & 0xFF; ip++; break;
            case '+': tape_val[rh] = (tape_val[rh] + 1) & 0xFF; ip++; break;
            case '.': tape_val[wh] = tape_val[rh];
                      tape_tok[wh] = tape_tok[rh]; ip++; break;
            case ',': tape_val[rh] = tape_val[wh];
                      tape_tok[rh] = tape_tok[wh]; ip++; break;
            case '[':
                if (tape_val[rh] == 0) {
                    int depth = 1;
                    ip++;
                    while (depth && ip < n) {
                        uint8_t c = tape_val[ip];
                        if (c == '[') depth++;
                        else if (c == ']') depth--;
                        ip++;
                    }
                    if (depth) return steps;
                } else {
                    ip++;
                }
                break;
            case ']':
                if (tape_val[rh] != 0) {
                    int depth = 1;
                    ip--;
                    while (depth && ip >= 0) {
                        uint8_t c = tape_val[ip];
                        if (c == ']') depth++;
                        else if (c == '[') depth--;
                        ip--;
                    }
                    if (depth) return steps;
                    ip++;
                } else {
                    ip++;
                }
                break;
            default: ip++; break;
        }
    }
    return steps;
}

extern "C" __global__ void run_epoch_kernel(
    uint8_t* soup_val,
    int64_t* soup_tok,
    int* steps_out,
    int pop_size,
    int max_steps,
    const uint32_t* __restrict__ pair_i,
    const uint32_t* __restrict__ pair_j,
    uint64_t epoch_seed
) {
    extern __shared__ uint8_t tape_shared[];
    int64_t* tape_tok = (int64_t*)(tape_shared + TAPE_LEN * blockDim.x);
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    uint8_t* tape = tape_shared + (threadIdx.x * TAPE_LEN);
    
    uint32_t i = pair_i[idx];
    uint32_t j = pair_j[idx];
    
    // Copy programs to tape
    int src_i = i * PROG_LEN;
    int src_j = j * PROG_LEN;
    
    for (int k = 0; k < PROG_LEN; k++) {
        tape[k] = soup_val[src_i + k];
        tape_tok[k] = soup_tok[src_i + k];
        tape[k + PROG_LEN] = soup_val[src_j + k];
        tape_tok[k + PROG_LEN] = soup_tok[src_j + k];
    }
    
    int steps = run_bff(tape, tape_tok, TAPE_LEN, max_steps);
    
    // Copy back
    int dst_i = i * PROG_LEN;
    int dst_j = j * PROG_LEN;
    
    for (int k = 0; k < PROG_LEN; k++) {
        soup_val[dst_i + k] = tape[k];
        soup_tok[dst_i + k] = tape_tok[k];
        soup_val[dst_j + k] = tape[k + PROG_LEN];
        soup_tok[dst_j + k] = tape_tok[k + PROG_LEN];
    }
    
    steps_out[idx] = steps;
}
"""

INIT_SOUP_KERNEL_SRC = """\
typedef unsigned char uint8_t;
typedef long long int64_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

#define PROG_LEN 64

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ __forceinline__ uint32_t rng_next(uint32_t* s) {
    uint32_t result = rotl32(s[1] * 5u, 7u) * 9u;
    uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1];
    s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl32(s[3], 11u);
    return result;
}

__device__ __forceinline__ void rng_seed(uint32_t* s, uint64_t seed) {
    s[0] = (uint32_t)(seed);
    s[1] = (uint32_t)(seed >> 16) ^ 0x9E3779B9u;
    s[2] = (uint32_t)(seed >> 32) ^ 0x6C62272Eu;
    s[3] = (uint32_t)(seed >> 48) ^ 0x61C88647u;
    for (int i = 0; i < 8; i++) rng_next(s);
}

extern "C" __global__ void init_soup_kernel(
    uint8_t* soup_val,
    int64_t* soup_tok,
    int pop_size,
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = pop_size * PROG_LEN;
    if (idx >= total) return;

    uint32_t rng[4];
    rng_seed(rng, seed + idx * 73856093ull);

    soup_val[idx] = (uint8_t)(rng_next(rng) & 0xFF);
    soup_tok[idx] = idx;
}
"""

GET_VALUES_KERNEL_SRC = """\
typedef unsigned char uint8_t;

extern "C" __global__ void get_values_kernel(
    const uint8_t* soup_val,
    uint8_t* values_out,
    int total_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_cells) return;
    values_out[idx] = soup_val[idx];
}
"""

GENERATE_PAIRS_KERNEL_SRC = """\
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

__device__ __forceinline__ uint32_t rng_next(uint32_t* s) {
    uint32_t result = rotl32(s[1] * 5u, 7u) * 9u;
    uint32_t t = s[1] << 9;
    s[2] ^= s[0]; s[3] ^= s[1];
    s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl32(s[3], 11u);
    return result;
}

__device__ __forceinline__ void rng_seed(uint32_t* s, uint64_t seed) {
    s[0] = (uint32_t)(seed);
    s[1] = (uint32_t)(seed >> 16) ^ 0x9E3779B9u;
    s[2] = (uint32_t)(seed >> 32) ^ 0x6C62272Eu;
    s[3] = (uint32_t)(seed >> 48) ^ 0x61C88647u;
    for (int i = 0; i < 8; i++) rng_next(s);
}

extern "C" __global__ void generate_pairs_kernel(
    uint32_t* pair_i,
    uint32_t* pair_j,
    int pop_size,
    uint64_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    uint32_t rng[4];
    rng_seed(rng, seed + idx * 73856093ull + idx * 19349663ull);

    uint32_t i = rng_next(rng) % pop_size;
    uint32_t j = rng_next(rng) % pop_size;

    // Ensure i != j
    while (j == i) {
        j = rng_next(rng) % pop_size;
    }

    pair_i[idx] = i;
    pair_j[idx] = j;
}
"""

BFF_KERNEL = RawKernel(BFF_KERNEL_SRC, "run_epoch_kernel")
INIT_SOUP_KERNEL = RawKernel(INIT_SOUP_KERNEL_SRC, "init_soup_kernel")
GET_VALUES_KERNEL = RawKernel(GET_VALUES_KERNEL_SRC, "get_values_kernel")
GENERATE_PAIRS_KERNEL = RawKernel(GENERATE_PAIRS_KERNEL_SRC, "generate_pairs_kernel")


class CUDAEngine:
    def __init__(
        self, pop_size: int, seed: int | None = None, max_steps: int = DEFAULT_MAX_STEPS
    ):
        self.pop_size = pop_size
        self.max_steps = max_steps
        self.prog_len = PROG_LEN
        self.tape_len = TAPE_LEN
        self.total_cells = pop_size * PROG_LEN

        self.block_size = 32
        self.grid_size = (pop_size + self.block_size - 1) // self.block_size

        import random

        self._rng = random.Random(seed)
        self._epoch_counter = 0

        # Separate arrays for values and token_ids
        self._soup_val = cp.zeros(self.total_cells, dtype=np.uint8)
        self._soup_tok = cp.zeros(self.total_cells, dtype=np.int64)

        init_seed = seed if seed is not None else self._rng.randint(0, 2**63)
        self._init_soup_gpu(init_seed)

    def _init_soup_gpu(self, seed: int):
        threads = 256
        blocks = (self.total_cells + threads - 1) // threads

        INIT_SOUP_KERNEL(
            (blocks,), (threads,), (self._soup_val, self._soup_tok, self.pop_size, seed)
        )
        cp.cuda.Stream.null.synchronize()

    def run_epoch(self, seed: int | None = None):
        if seed is None:
            seed = self._rng.randint(0, 2**63)

        # Generate pairs on GPU for better performance
        pair_i_gpu = cp.zeros(self.pop_size, dtype=np.uint32)
        pair_j_gpu = cp.zeros(self.pop_size, dtype=np.uint32)

        threads = min(256, self.pop_size)
        blocks = (self.pop_size + threads - 1) // threads

        GENERATE_PAIRS_KERNEL(
            (blocks,), (threads,), (pair_i_gpu, pair_j_gpu, self.pop_size, seed)
        )
        cp.cuda.Stream.null.synchronize()

        steps_out = cp.zeros(self.pop_size, dtype=np.int32)

        # Shared memory: tape values + token_ids
        # Each block has block_size threads, each thread needs TAPE_LEN bytes for values
        # Plus TAPE_LEN * 8 bytes for token_ids
        shared_mem = self.block_size * (TAPE_LEN + TAPE_LEN * 8)

        BFF_KERNEL(
            (self.grid_size,),
            (self.block_size,),
            (
                self._soup_val,
                self._soup_tok,
                steps_out,
                self.pop_size,
                self.max_steps,
                pair_i_gpu,
                pair_j_gpu,
                seed,
            ),
            shared_mem=shared_mem,
        )

        cp.cuda.Stream.null.synchronize()

        self._epoch_counter += 1
        return cp.asnumpy(steps_out)

    def run_epochs(self, n: int) -> np.ndarray:
        total = n * self.pop_size
        all_steps = np.zeros(total, dtype=np.int32)

        for e in range(n):
            seed = self._rng.randint(0, 2**63)
            steps = self.run_epoch(seed)
            all_steps[e * self.pop_size : (e + 1) * self.pop_size] = steps

        return all_steps.reshape(n, self.pop_size)

    def unique_tokens(self) -> int:
        soup_tok_cpu = cp.asnumpy(self._soup_tok)
        return len(set(soup_tok_cpu.tolist()))

    def get_values(self) -> bytes:
        values = cp.zeros(self.total_cells, dtype=np.uint8)

        threads = 256
        blocks = (self.total_cells + threads - 1) // threads

        GET_VALUES_KERNEL(
            (blocks,), (threads,), (self._soup_val, values, self.total_cells)
        )

        cp.cuda.Stream.null.synchronize()
        return bytes(cp.asnumpy(values))

    def dump_programs(self, top_n: int = 40) -> str:
        soup_val_cpu = cp.asnumpy(self._soup_val)
        soup_tok_cpu = cp.asnumpy(self._soup_tok)

        from collections import Counter

        BFF_CHARS = set(b"<>{}-+.,[]")

        total_interactions = self._epoch_counter * self.pop_size
        n_unique = self.unique_tokens()

        all_tids = Counter()
        programs = []

        for p in range(self.pop_size):
            offset = p * PROG_LEN
            vals = bytes(soup_val_cpu[offset : offset + PROG_LEN])
            tids = soup_tok_cpu[offset : offset + PROG_LEN].tolist()
            programs.append((vals, tids))
            all_tids.update(tids)

        top_tokens = {tid for tid, _ in all_tids.most_common(50)}
        lineages = {}

        for vals, tids in programs:
            prog_tids = Counter(tids)
            best_tid = max(
                top_tokens & set(prog_tids.keys()),
                key=lambda t: prog_tids[t],
                default=tids[0],
            )
            lineages.setdefault(best_tid, []).append(vals)

        ranked = sorted(lineages.items(), key=lambda kv: -len(kv[1]))

        lines = [
            f"{total_interactions:,} interactions",
            f"{n_unique} unique tokens, {self.pop_size} programs",
        ]

        shown = 0
        for tid, members in ranked:
            if shown >= top_n:
                break
            tid_hex = f"{tid & 0xFFFF:04X}"
            for prog_bytes in members[: min(3, top_n - shown)]:
                rendered = []
                for b in prog_bytes:
                    if b in BFF_CHARS:
                        rendered.append(chr(b))
                    else:
                        rendered.append(" ")
                tape_str = "".join(rendered).rstrip()
                lines.append(f"{len(members):5d}: {tid_hex} {tape_str}")
                shown += 1

        return "\n".join(lines)


def create_population(
    pop_size: int, seed: int | None = None, max_steps: int = DEFAULT_MAX_STEPS
) -> CUDAEngine:
    return CUDAEngine(pop_size, seed, max_steps)
