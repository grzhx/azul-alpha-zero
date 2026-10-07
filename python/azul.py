"""Dependency-free ctypes interface; optional NumPy views are zero-copy.

Arrays belong to Batch and are overwritten by the next observe()/step().
Copy observations if storing them in a replay buffer.
"""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import sys

ACTION_COUNT = 180
OBSERVATION_SIZE = 172


class StepResult(C.Structure):
    _fields_ = [("reward", C.c_float), ("scores", C.c_uint16 * 2),
                ("actor", C.c_uint8), ("terminated", C.c_uint8),
                ("round_finished", C.c_uint8), ("invalid_action", C.c_uint8)]


def _load(path: str | Path | None) -> C.CDLL:
    if path is None:
        root = Path(__file__).resolve().parents[1]
        name = "azul.dll" if sys.platform == "win32" else ("libazul.dylib" if sys.platform == "darwin" else "libazul.so")
        candidates = [root / "build-release" / name, root / "build" / "Release" / name,
                      root / "build" / name]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError("Build the C++ library first, or supply library=...")
    lib = C.CDLL(str(Path(path).resolve()))  # CDLL releases the GIL during C calls
    signatures = {
        "azul_abi_version": (C.c_int, []),
        "azul_batch_create": (C.c_void_p, [C.c_size_t, C.c_uint, C.c_uint64]),
        "azul_batch_destroy": (None, [C.c_void_p]),
        "azul_batch_reset": (C.c_int, [C.c_void_p, C.c_uint64]),
        "azul_batch_reset_at": (C.c_int, [C.c_void_p, C.c_size_t, C.c_uint64, C.c_uint]),
        "azul_batch_observe": (C.c_int, [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_uint8), C.POINTER(C.c_uint8)]),
        "azul_batch_step": (C.c_int, [C.c_void_p, C.POINTER(C.c_uint16), C.POINTER(StepResult)]),
        "azul_snapshot_size": (C.c_size_t, []),
        "azul_batch_snapshot": (C.c_int, [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]),
        "azul_batch_restore": (C.c_int, [C.c_void_p, C.c_size_t, C.c_void_p, C.c_size_t]),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, args
    # Public-state import is used by external-board tooling, not normal play.
    # Keep older compatible game DLLs loadable and fail only when this feature
    # is actually requested.
    if hasattr(lib, "azul_batch_import_public"):
        lib.azul_batch_import_public.restype = C.c_int
        lib.azul_batch_import_public.argtypes = [C.c_void_p, C.c_size_t,
            C.POINTER(C.c_uint8), C.POINTER(C.c_uint32), C.POINTER(C.c_uint16),
            C.POINTER(C.c_uint8), C.POINTER(C.c_uint8), C.POINTER(C.c_uint8),
            C.POINTER(C.c_uint8), C.POINTER(C.c_uint8), C.POINTER(C.c_uint8),
            C.c_uint32, C.c_uint8, C.c_uint8, C.c_uint8]
    if lib.azul_abi_version() != 1:
        raise RuntimeError("Unsupported Azul ABI version")
    return lib


class Batch:
    def __init__(self, n: int, threads: int = 1, seed: int = 42, library=None):
        if n < 1 or threads < 0 or threads > 1024:
            raise ValueError("n must be positive and threads in [0, 1024]")
        self._lib = _load(library)
        self._handle = None
        self.n = n
        # Allocate buffers before starting native workers, so allocation failure cannot leak a handle.
        self.observations = (C.c_float * (n * OBSERVATION_SIZE))()
        self.masks = (C.c_uint8 * (n * ACTION_COUNT))()
        self.players = (C.c_uint8 * n)()
        self.actions = (C.c_uint16 * n)()
        self.results = (StepResult * n)()
        self._handle = self._lib.azul_batch_create(n, threads, seed)
        if not self._handle:
            raise RuntimeError("Failed to allocate batch or start worker threads")

    def _check(self, status):
        if status:
            raise ValueError("Azul call failed: invalid argument, snapshot, or native error")

    def _open(self):
        if not self._handle:
            raise RuntimeError("Batch is closed")

    def observe(self):
        self._open()
        self._check(self._lib.azul_batch_observe(self._handle, self.observations, self.masks, self.players))
        return self.observations, self.masks, self.players

    def step(self, actions=None):
        self._open()
        if actions is not None:
            if len(actions) != self.n:
                raise ValueError("Expected one action per environment")
            for i, action in enumerate(actions):
                value = int(action)
                if value < 0 or value > 65535:
                    raise ValueError("Action ID out of uint16 range")
                self.actions[i] = value
        self._check(self._lib.azul_batch_step(self._handle, self.actions, self.results))
        return self.results

    def manual_deal(self, colors):
        if len(colors) != 20 or any(type(c) is not int or c not in (*range(5),255) for c in colors):
            raise ValueError("手动发牌须提供20个0到4的颜色")
        self._lib.azul_batch_manual_deal.argtypes=[C.c_void_p,C.POINTER(C.c_uint8)]
        self._lib.azul_batch_manual_deal.restype=C.c_int
        values=(C.c_uint8*20)(*[int(c) for c in colors])
        self._check(self._lib.azul_batch_manual_deal(self._handle,values))

    def manual_reset(self,seed):
        self._lib.azul_manual_reset.argtypes=[C.c_void_p,C.c_uint64]
        self._check(self._lib.azul_manual_reset(self._handle,seed))

    def manual_step(self,action):
        self._lib.azul_manual_step.argtypes=[C.c_void_p,C.c_uint16,C.POINTER(StepResult)]
        self._check(self._lib.azul_manual_step(self._handle,action,self.results))
        return self.results[0]

    def reset(self, seed=42):
        self._open()
        self._check(self._lib.azul_batch_reset(self._handle, seed))

    def reset_at(self, index, seed, starting_player=0):
        self._open()
        if not 0 <= index < self.n or starting_player not in (0, 1):
            raise ValueError("Invalid index or starting player")
        self._check(self._lib.azul_batch_reset_at(self._handle, index, seed, starting_player))

    def snapshot(self, index):
        self._open()
        if not 0 <= index < self.n:
            raise ValueError("Invalid index")
        data = C.create_string_buffer(self._lib.azul_snapshot_size())
        self._check(self._lib.azul_batch_snapshot(self._handle, index, data, len(data)))
        return data.raw

    def restore(self, index, snapshot: bytes):
        self._open()
        if not 0 <= index < self.n:
            raise ValueError("Invalid index")
        data = C.create_string_buffer(snapshot, len(snapshot))
        self._check(self._lib.azul_batch_restore(self._handle, index, data, len(data)))

    def import_public(self, index, *, sources, walls, scores, pattern_colors,
                      pattern_counts, floor_tiles, floor_counts, bag, discard,
                      round, current, next_start, token_available):
        self._open()
        if not hasattr(self._lib, "azul_batch_import_public"):
            raise RuntimeError("This Azul library does not support public-state import; rebuild azul.dll")
        if not 0 <= index < self.n:
            raise ValueError("Invalid index")

        def array(ctype, values, size):
            flat = ([int(value) for row in values for value in row]
                    if values and isinstance(values[0], (list, tuple))
                    else [int(value) for value in values])
            if len(flat) != size:
                raise ValueError(f"Expected {size} public-state values")
            return (ctype * size)(*flat)

        args = (array(C.c_uint8, sources, 30), array(C.c_uint32, walls, 2),
                array(C.c_uint16, scores, 2), array(C.c_uint8, pattern_colors, 10),
                array(C.c_uint8, pattern_counts, 10), array(C.c_uint8, floor_tiles, 10),
                array(C.c_uint8, floor_counts, 2), array(C.c_uint8, bag, 5),
                array(C.c_uint8, discard, 5))
        self._check(self._lib.azul_batch_import_public(
            self._handle, index, *args, int(round), int(current), int(next_start),
            int(token_available)))

    def numpy_views(self):
        """Optional dependency: pip install numpy. Writes to actions go directly to C buffers."""
        import numpy as np
        return {
            "observations": np.ctypeslib.as_array(self.observations).reshape(self.n, OBSERVATION_SIZE),
            "masks": np.ctypeslib.as_array(self.masks).reshape(self.n, ACTION_COUNT),
            "players": np.ctypeslib.as_array(self.players),
            "actions": np.ctypeslib.as_array(self.actions),
        }

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.azul_batch_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
