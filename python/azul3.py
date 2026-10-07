"""High-throughput three-player Azul batch ABI."""
from __future__ import annotations
import ctypes as C
from pathlib import Path
import sys
ACTION_COUNT=240; OBSERVATION_SIZE=244
class StepResult(C.Structure):
    _fields_=[("reward",C.c_float),("scores",C.c_uint16*3),("actor",C.c_uint8),("terminated",C.c_uint8),("round_finished",C.c_uint8),("invalid_action",C.c_uint8)]
def _load(path=None):
    if path is None:
        root=Path(__file__).resolve().parents[1];name="azul.dll" if sys.platform=="win32" else ("libazul.dylib" if sys.platform=="darwin" else "libazul.so")
        path=next((p for p in [root/"build-release"/name,root/"build"/"Release"/name,root/"build"/name] if p.is_file()),None)
        if path is None: raise FileNotFoundError("Build azul.dll first")
    lib=C.CDLL(str(Path(path).resolve()))
    sig={"azul3_abi_version":(C.c_int,[]),"azul3_batch_create":(C.c_void_p,[C.c_size_t,C.c_uint,C.c_uint64]),"azul3_batch_destroy":(None,[C.c_void_p]),"azul3_batch_reset":(C.c_int,[C.c_void_p,C.c_uint64]),"azul3_batch_reset_at":(C.c_int,[C.c_void_p,C.c_size_t,C.c_uint64,C.c_uint]),"azul3_batch_observe":(C.c_int,[C.c_void_p,C.POINTER(C.c_float),C.POINTER(C.c_uint8),C.POINTER(C.c_uint8)]),"azul3_batch_step":(C.c_int,[C.c_void_p,C.POINTER(C.c_uint16),C.POINTER(StepResult)])}
    for n,(r,a) in sig.items():f=getattr(lib,n);f.restype=r;f.argtypes=a
    if hasattr(lib,"azul3_batch_import_public"):
        lib.azul3_batch_import_public.restype=C.c_int
        lib.azul3_batch_import_public.argtypes=[C.c_void_p,C.c_size_t,C.POINTER(C.c_uint8),C.POINTER(C.c_uint32),C.POINTER(C.c_uint16),C.POINTER(C.c_uint8),C.POINTER(C.c_uint8),C.POINTER(C.c_uint8),C.POINTER(C.c_uint8),C.POINTER(C.c_uint8),C.POINTER(C.c_uint8),C.c_uint32,C.c_uint8,C.c_uint8,C.c_uint8]
    if lib.azul3_abi_version()!=1: raise RuntimeError("Unsupported Azul3 ABI")
    return lib
class Batch3:
    def __init__(self,n,threads=1,seed=42,library=None):
        if n<1 or threads<0:raise ValueError("invalid batch size/threads")
        self._lib=_load(library);self.n=n
        self.observations=(C.c_float*(n*OBSERVATION_SIZE))();self.masks=(C.c_uint8*(n*ACTION_COUNT))();self.players=(C.c_uint8*n)();self.actions=(C.c_uint16*n)();self.results=(StepResult*n)()
        self._handle=self._lib.azul3_batch_create(n,threads,seed)
        if not self._handle:raise RuntimeError("failed to create 3P batch")
    def _check(self,x):
        if x:raise RuntimeError("Azul3 native call failed")
    def observe(self):self._check(self._lib.azul3_batch_observe(self._handle,self.observations,self.masks,self.players));return self.observations,self.masks,self.players
    def step(self,actions=None):
        if actions is not None:
            for i,a in enumerate(actions):self.actions[i]=int(a)
        self._check(self._lib.azul3_batch_step(self._handle,self.actions,self.results));return self.results
    def reset(self,seed=42):self._check(self._lib.azul3_batch_reset(self._handle,seed))
    def reset_at(self,index,seed,starting_player=0):self._check(self._lib.azul3_batch_reset_at(self._handle,index,seed,starting_player))
    def import_public(self,index,*,sources,walls,scores,pattern_colors,pattern_counts,floor_tiles,floor_counts,bag,discard,round,current,next_start,token_available):
        if not self._handle:raise RuntimeError("Batch3 is closed")
        if not 0<=index<self.n:raise ValueError("Invalid index")
        if not hasattr(self._lib,"azul3_batch_import_public"):
            raise RuntimeError("Rebuild azul.dll for three-player public-state import")
        def arr(values,kind,size):
            flat=([int(x) for row in values for x in row] if len(values) and isinstance(values[0],(list,tuple)) else [int(x) for x in values])
            if len(flat)!=size:raise ValueError(f"Expected {size} public-state values")
            if any(x<0 or x>(1<<(8*C.sizeof(kind)))-1 for x in flat):raise ValueError("Public-state value out of range")
            return (kind*size)(*flat)
        if not 0<=current<3 or not 0<=next_start<3 or token_available not in (0,1) or not 0<=round<=0xffffffff:
            raise ValueError("Invalid public-state counters")
        args=(arr(sources,C.c_uint8,40),arr(walls,C.c_uint32,3),arr(scores,C.c_uint16,3),arr(pattern_colors,C.c_uint8,15),arr(pattern_counts,C.c_uint8,15),arr(floor_tiles,C.c_uint8,15),arr(floor_counts,C.c_uint8,3),arr(bag,C.c_uint8,5),arr(discard,C.c_uint8,5))
        self._check(self._lib.azul3_batch_import_public(self._handle,index,*args,round,current,next_start,token_available))
    def numpy_views(self):
        import numpy as np
        return {"observations":np.ctypeslib.as_array(self.observations).reshape(self.n,OBSERVATION_SIZE),"masks":np.ctypeslib.as_array(self.masks).reshape(self.n,ACTION_COUNT),"players":np.ctypeslib.as_array(self.players),"actions":np.ctypeslib.as_array(self.actions)}
    def close(self):
        if getattr(self,"_handle",None):self._lib.azul3_batch_destroy(self._handle);self._handle=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
    def __del__(self):self.close()
