"""
https://arxiv.org/abs/2406.19108
"""

import os
import math
import random
import zlib
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

PROG_LEN = 64


def _get_backend():
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return "cuda"
    except Exception:
        return "c"


if _get_backend() == "cuda":
    from bff_cuda import CUDAEngine as CPopulation

    class Population:
        def __init__(self, size: int, seed: int | None = None, max_steps: int = 16384):
            self._impl = CPopulation(size, seed, max_steps)
            self.size = size
            self.max_steps = max_steps

        def run_epoch(self) -> List[int]:
            return self._impl.run_epoch().tolist()

        def run_epochs(self, n: int) -> np.ndarray:
            return self._impl.run_epochs(n)

        def unique_tokens(self) -> int:
            return self._impl.unique_tokens()

        def get_values(self) -> bytes:
            return self._impl.get_values()

        def dump_programs(self, top_n: int = 40) -> str:
            return self._impl.dump_programs(top_n)

        def higher_order_entropy(self) -> float:
            data = self._impl.get_values()
            n = len(data)
            counts = Counter(data)
            h = 0.0
            for c in counts.values():
                p = c / n
                if p > 0:
                    h -= p * math.log2(p)
            compressed = zlib.compress(data, level=9)
            C = len(compressed) * 8
            return h - (C / n)
else:
    import ctypes

    _dir = Path(__file__).parent
    _lib = ctypes.CDLL(str(_dir / "bff_engine.so"))

    class CellStruct(ctypes.Structure):
        _fields_ = [("value", ctypes.c_uint8), ("token_id", ctypes.c_int64)]

    CellArray = ctypes.POINTER(CellStruct)

    _lib.init_soup.argtypes = [CellArray, ctypes.c_int, ctypes.c_uint64]
    _lib.init_soup.restype = None

    _lib.run_epoch.argtypes = [
        CellArray,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    _lib.run_epoch.restype = None

    _lib.run_epochs.argtypes = [
        CellArray,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
    ]
    _lib.run_epochs.restype = None

    _lib.count_unique_tokens.argtypes = [CellArray, ctypes.c_int]
    _lib.count_unique_tokens.restype = ctypes.c_int

    _lib.get_values.argtypes = [CellArray, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int]
    _lib.get_values.restype = None

    class Population:
        """C-backed population (CPU)."""

        def __init__(self, size: int, seed: int | None = None, max_steps: int = 16384):
            self.size = size
            self.max_steps = max_steps
            self.total_cells = size * PROG_LEN
            self._rng = random.Random(seed)
            self._epoch_counter = 0

            self._soup = (CellStruct * self.total_cells)()
            init_seed = seed if seed is not None else self._rng.randint(0, 2**63)
            _lib.init_soup(self._soup, size, ctypes.c_uint64(init_seed))

        def run_epoch(self) -> List[int]:
            steps = (ctypes.c_int * self.size)()
            epoch_seed = self._rng.randint(0, 2**63)
            _lib.run_epoch(
                self._soup,
                steps,
                self.size,
                self.max_steps,
                ctypes.c_uint64(epoch_seed),
            )
            self._epoch_counter += 1
            return list(steps)

        def run_epochs(self, n: int) -> np.ndarray:
            total = n * self.size
            steps = (ctypes.c_int * total)()
            base_seed = self._rng.randint(0, 2**63)
            _lib.run_epochs(
                self._soup,
                steps,
                self.size,
                self.max_steps,
                n,
                ctypes.c_uint64(base_seed),
            )
            self._epoch_counter += n
            return np.frombuffer(steps, dtype=np.int32).reshape(n, self.size)

        def unique_tokens(self) -> int:
            return _lib.count_unique_tokens(self._soup, self.total_cells)

        def get_values(self) -> bytes:
            buf = (ctypes.c_uint8 * self.total_cells)()
            _lib.get_values(self._soup, buf, self.total_cells)
            return bytes(buf)

        def dump_programs(self, top_n: int = 40) -> str:
            BFF_CHARS = set(b"<>{}-+.,[]")

            total_interactions = self._epoch_counter * self.size
            n_unique = self.unique_tokens()

            all_tids = Counter()
            programs = []
            for p in range(self.size):
                offset = p * PROG_LEN
                vals = bytes(self._soup[offset + i].value for i in range(PROG_LEN))
                tids = [self._soup[offset + i].token_id for i in range(PROG_LEN)]
                programs.append((vals, tids))
                all_tids.update(tids)

            top_tokens = {tid for tid, _ in all_tids.most_common(50)}
            lineages: dict[int, list[bytes]] = {}
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
                f"{n_unique} unique tokens, {self.size} programs",
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

        def higher_order_entropy(self) -> float:
            data = self.get_values()
            n = len(data)
            counts = Counter(data)
            h = 0.0
            for c in counts.values():
                p = c / n
                if p > 0:
                    h -= p * math.log2(p)
            compressed = zlib.compress(data, level=9)
            C = len(compressed) * 8
            return h - (C / n)
