from dataclasses import dataclass, asdict
import ctypes as C
import numpy as np
import torch
from azul import Batch


@dataclass
class SearchConfig:
    simulations: int = 128
    candidates: int = 16
    max_depth: int = 96
    chance_initial: int = 2
    chance_cap: int = 64
    gumbel_scale: float = 1.0
    value_scale: float = 0.1
    maxvisit_init: float = 50.0


class NativeConfig(C.Structure):
    _fields_ = [(name, C.c_int32) for name in ("simulations", "candidates", "max_depth", "chance_initial", "chance_cap")] + [
        (name, C.c_float) for name in ("gumbel_scale", "value_scale", "maxvisit_init")]


class Stats(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in ("simulations", "evaluations", "chance_draws", "chance_reuses", "chance_duplicates", "nodes")]


def pointer(a, kind=C.c_float):
    return a.ctypes.data_as(C.POINTER(kind))


class Evaluator:
    """Persistent pinned CPU buffers + fixed GPU tensors; optional CUDA graph.

    C++ writes directly into observations. GPU output is packed into a single
    transfer (180 logits + one value). Fixed-size batches allow graph replay.
    No CPU/GPU scalar round trips inside the network.
    """
    def __init__(self, model, n, device, use_graph=True, bf16=True):
        self.model, self.device = model, torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.bf16 = bf16 and self.cuda
        self.input_cpu = torch.empty((n, 172), dtype=torch.float32, pin_memory=self.cuda)
        self.output_cpu = torch.empty((n, 181), dtype=torch.float32, pin_memory=self.cuda)
        self.observations = self.input_cpu.numpy()
        self.output = self.output_cpu.numpy()
        self.logits = np.empty((n, 180), dtype=np.float32)
        self.values = np.empty(n, dtype=np.float32)
        self.input_gpu = torch.zeros((n, 172), device=self.device)
        self.graph = None
        self.calls = 0
        model.eval()
        if self.cuda and use_graph:
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(3):
                    self._forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(self.graph):
                self.graph_output = self._forward()

    def _forward(self):
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.bf16, cache_enabled=False):
            logits, wdl = self.model(self.input_gpu)
        probabilities = wdl.float().softmax(-1)
        return torch.cat((logits.float(), (probabilities[:, 0] - probabilities[:, 2])[:, None]), 1)

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()
        self.input_gpu.copy_(self.input_cpu, non_blocking=self.cuda)
        if self.graph is not None:
            self.graph.replay()
            out = self.graph_output
        else:
            out = self._forward()
        self.output_cpu.copy_(out, non_blocking=self.cuda)
        if self.cuda:
            torch.cuda.current_stream(self.device).synchronize()
        # Native submit takes two contiguous arrays. These are reused, never allocated per call.
        np.copyto(self.logits, self.output[:, :180])
        np.copyto(self.values, self.output[:, 180])
        self.calls += 1
        return self.logits, self.values


def bind_search_api(lib):
    signatures = {
            "azul_search_abi_version": (C.c_int, []),
            "azul_search_create": (C.c_void_p, [C.c_size_t, C.c_uint, C.POINTER(NativeConfig)]),
            "azul_search_destroy": (None, [C.c_void_p]),
            "azul_search_begin": (C.c_int, [C.c_void_p, C.c_void_p, C.c_uint64]),
            "azul_search_request": (C.c_int, [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_uint8)]),
            "azul_search_submit": (C.c_int, [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_float)]),
            "azul_search_results": (C.c_int, [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_uint16), C.POINTER(C.c_float), C.POINTER(Stats)]),
    }
    for name, (result, args) in signatures.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = result, args
    if lib.azul_search_abi_version() != 1:
        raise RuntimeError("Unsupported search ABI")


class Search:
    def __init__(self, batch: Batch, config=None, threads=8):
        self.batch, self.config = batch, config or SearchConfig()
        self.lib, self.n = batch._lib, batch.n
        self.handle = None
        bind_search_api(self.lib)
        native = NativeConfig(**asdict(self.config))
        self.handle = self.lib.azul_search_create(self.n, threads, C.byref(native))
        if not self.handle:
            raise ValueError("Invalid search configuration or allocation failed")
        self.ready = np.zeros(self.n, dtype=np.uint8)
        self.policies = np.zeros((self.n, 180), dtype=np.float32)
        self.actions = np.zeros(self.n, dtype=np.uint16)
        self.values = np.zeros(self.n, dtype=np.float32)
        self.stats = Stats()

    def run(self, evaluator, seed):
        if not self.handle:
            raise RuntimeError("Search is closed")
        self.batch._open()
        self._check(self.lib.azul_search_begin(self.handle, self.batch._handle, seed))
        for _ in range(self.config.simulations + 2):
            count = self.lib.azul_search_request(self.handle, pointer(evaluator.observations), pointer(self.ready, C.c_uint8))
            self._check(count)
            if count == 0:
                break
            logits, values = evaluator.evaluate()
            self._check(self.lib.azul_search_submit(self.handle, pointer(logits), pointer(values)))
        else:
            raise RuntimeError("Search exceeded its evaluation budget")
        self._check(self.lib.azul_search_results(self.handle, pointer(self.policies), pointer(self.actions, C.c_uint16), pointer(self.values), C.byref(self.stats)))
        return self.actions, self.policies, self.values

    @staticmethod
    def _check(code):
        if code < 0:
            raise RuntimeError("Native search failed: invalid response, arguments, or allocation")

    def close(self):
        if getattr(self, "handle", None):
            self.lib.azul_search_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
