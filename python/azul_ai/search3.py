from dataclasses import dataclass,asdict
import ctypes as C, numpy as np, torch
from azul3 import Batch3,OBSERVATION_SIZE,ACTION_COUNT
@dataclass
class SearchConfig3:
    simulations:int=128; candidates:int=32; max_depth:int=128; chance_initial:int=4; chance_cap:int=128; gumbel_scale:float=1.; value_scale:float=.1; maxvisit_init:float=50.
class NativeConfig(C.Structure):
    _fields_=[(n,C.c_int32) for n in ("simulations","candidates","max_depth","chance_initial","chance_cap")]+[(n,C.c_float) for n in ("gumbel_scale","value_scale","maxvisit_init")]
class Stats(C.Structure):_fields_=[(n,C.c_uint64) for n in ("simulations","evaluations","chance_draws","chance_reuses","chance_duplicates","nodes")]
def ptr(a,kind=C.c_float):return a.ctypes.data_as(C.POINTER(kind))
class Evaluator3:
    def __init__(self,model,n,device,use_graph=True):
        self.model=model;self.device=torch.device(device);self.cuda=self.device.type=="cuda";self.input_cpu=torch.empty((n,OBSERVATION_SIZE),dtype=torch.float32,pin_memory=self.cuda);self.observations=self.input_cpu.numpy();self.input_gpu=torch.zeros((n,OBSERVATION_SIZE),device=self.device);self.logits=np.empty((n,ACTION_COUNT),np.float32);self.values=np.empty((n,3),np.float32);self.out_cpu=torch.empty((n,ACTION_COUNT+3),dtype=torch.float32,pin_memory=self.cuda);self.graph=None;model.eval()
        if self.cuda and use_graph:
            with torch.inference_mode():
                for _ in range(3):self._forward()
            torch.cuda.synchronize();self.graph=torch.cuda.CUDAGraph()
            with torch.inference_mode(),torch.cuda.graph(self.graph):self.graph_out=self._forward()
    def _forward(self):
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.cuda,cache_enabled=False):p,v=self.model(self.input_gpu)
        return torch.cat((p.float(),v.float()),1)
    @torch.inference_mode()
    def evaluate(self):
        self.input_gpu.copy_(self.input_cpu,non_blocking=self.cuda)
        if self.graph is not None:
            self.graph.replay(); out=self.graph_out
        else: out=self._forward()
        self.out_cpu.copy_(out,non_blocking=self.cuda)
        if self.cuda:torch.cuda.current_stream().synchronize()
        a=self.out_cpu.numpy();np.copyto(self.logits,a[:,:ACTION_COUNT]);np.copyto(self.values,a[:,ACTION_COUNT:ACTION_COUNT+3]);return self.logits,self.values
class Search3:
    def __init__(self,batch,config=None,threads=8):
        self.batch=batch;self.config=config or SearchConfig3();self.lib=batch._lib;self.n=batch.n;self.lib.azul3_search_create.restype=C.c_void_p;self.lib.azul3_search_create.argtypes=[C.c_size_t,C.c_uint,C.POINTER(NativeConfig)];self.lib.azul3_search_destroy.argtypes=[C.c_void_p];self.lib.azul3_search_begin.argtypes=[C.c_void_p,C.c_void_p,C.c_uint64];self.lib.azul3_search_request.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.POINTER(C.c_uint8)];self.lib.azul3_search_submit.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.POINTER(C.c_float)];self.lib.azul3_search_results.argtypes=[C.c_void_p,C.POINTER(C.c_float),C.POINTER(C.c_uint16),C.POINTER(C.c_float),C.POINTER(Stats)]
        self.native=NativeConfig(**asdict(self.config));self.handle=self.lib.azul3_search_create(self.n,threads,C.byref(self.native));
        if not self.handle:raise RuntimeError("invalid 3P search config")
        self.ready=np.zeros(self.n,np.uint8);self.policies=np.zeros((self.n,ACTION_COUNT),np.float32);self.actions=np.zeros(self.n,np.uint16);self.values=np.zeros(self.n,np.float32);self.stats=Stats()
    def run(self,evaluator,seed,cancel=None):
        if self.lib.azul3_search_begin(self.handle,self.batch._handle,seed)<0:raise RuntimeError("3P search begin failed")
        for _ in range(self.config.simulations+3):
            if cancel is not None and cancel.is_set():return None
            count=self.lib.azul3_search_request(self.handle,ptr(evaluator.observations),ptr(self.ready,C.c_uint8))
            if count<0:raise RuntimeError("3P search request failed")
            if count==0:break
            p,v=evaluator.evaluate()
            if self.lib.azul3_search_submit(self.handle,ptr(p),ptr(v))<0:raise RuntimeError("3P search submit failed")
        else:raise RuntimeError("3P search budget exceeded")
        if self.lib.azul3_search_results(self.handle,ptr(self.policies),ptr(self.actions,C.c_uint16),ptr(self.values),C.byref(self.stats))<0:raise RuntimeError("3P search results failed")
        return self.actions,self.policies,self.values
    def close(self):
        if getattr(self,"handle",None):self.lib.azul3_search_destroy(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*_):self.close()
