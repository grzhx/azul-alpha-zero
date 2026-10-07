"""Packed/bucketed CUDA graph inference and independent-tree CPU/GPU pipeline."""
import copy
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import time
import numpy as np
import torch
from azul import Batch
from .search import Search, SearchConfig, NativeConfig, Stats, pointer, bind_search_api


@dataclass
class Tuning:
    q_mode: int = 1
    subtree_reuse: int = 1
    cache_capacity: int = 32768
    symmetric_cache: int = 1
    q_floor: float = 0.25
    chance_coefficient: float = 2.0
    chance_exponent: float = 0.5
    chance_sensitivity: float = 1.0
    afterstate_prior: float = 0.25


class NativeTuning(C.Structure):
    _fields_ = [(name, C.c_int32) for name in ("q_mode", "subtree_reuse", "cache_capacity", "symmetric_cache")] + [
        (name, C.c_float) for name in ("q_floor", "chance_coefficient", "chance_exponent", "chance_sensitivity", "afterstate_prior")]

class StableNorm(torch.nn.LayerNorm):
    def forward(self,x):
        return torch.nn.functional.layer_norm(x.float(),self.normalized_shape,self.weight,self.bias,self.eps).to(x.dtype)


class FrozenActor:
    def __init__(self, learner, device, bf16=True):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if bf16 and self.device.type == "cuda" else torch.float32
        self.model = copy.deepcopy(learner).to(device=self.device, dtype=self.dtype).eval()
        # Native mixed-precision LayerNorm accepts FP32 affine parameters.
        for module in self.model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.__class__ = StableNorm
                module.float()
        self.model.requires_grad_(False)
        self.version = 0

    @torch.no_grad()
    def publish(self, learner, version):
        if version <= self.version:
            raise ValueError("Actor version must increase when publishing weights")
        self.model.load_state_dict(learner.state_dict())  # copies, preserves graph data pointers
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.version = version

    def predict(self, x):
        logits, wdl = self.model(x)
        wdl = wdl.float().softmax(-1)
        return torch.cat((logits.float(), (wdl[:, 0]-wdl[:, 2])[:, None]), -1)


class BucketEvaluator:
    def __init__(self, actor, capacity, graphs=True):
        self.actor, self.capacity = actor, capacity
        self.cuda = actor.device.type == "cuda"
        self.input_cpu = torch.empty((capacity,172),dtype=torch.float32,pin_memory=self.cuda)
        self.packed_cpu = torch.empty_like(self.input_cpu,pin_memory=self.cuda)
        self.output_cpu = torch.empty((capacity,181),dtype=torch.float32,pin_memory=self.cuda)
        self.observations = self.input_cpu.numpy()
        self.packed = self.packed_cpu.numpy()
        self.output = self.output_cpu.numpy()
        self.logits = np.empty((capacity,180),np.float32)
        self.values = np.empty(capacity,np.float32)
        sizes=[]; size=16
        while size < capacity:
            sizes.append(size);size*=2
        sizes.append(capacity)
        self.sizes=sorted(set(sizes))
        self.buckets={}
        self.stream = torch.cuda.Stream(device=actor.device) if self.cuda else None
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if self.cuda else None
        self.metrics=dict(gpu_forward_ms=0.0,h2d_ms=0.0,d2h_ms=0.0,pack_seconds=0.0,
                          valid_leaves=0,padded_leaves=0,gpu_batches=0)
        for size in self.sizes:
            x=torch.zeros((size,172),device=actor.device,dtype=actor.dtype)
            graph=None;output=None
            if self.cuda and graphs:
                with torch.cuda.stream(self.stream), torch.inference_mode():
                    for _ in range(3): actor.predict(x)
                self.stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.inference_mode(),torch.cuda.graph(graph,stream=self.stream):
                    output=actor.predict(x)
            self.buckets[size]=(x,graph,output)

    @torch.inference_mode()
    def launch(self, ready):
        start=time.perf_counter()
        self.indices=np.flatnonzero(ready)
        count=len(self.indices)
        size=next(s for s in self.sizes if s>=count)
        np.take(self.observations,self.indices,axis=0,out=self.packed[:count],mode="clip")
        self.packed[count:size].fill(0)
        self.metrics["pack_seconds"]+=time.perf_counter()-start
        self.metrics["valid_leaves"]+=count;self.metrics["padded_leaves"]+=size-count;self.metrics["gpu_batches"]+=1
        x,graph,output=self.buckets[size]
        if self.cuda:
            with torch.cuda.stream(self.stream):
                self.events[0].record()
                x.copy_(self.packed_cpu[:size],non_blocking=True)
                self.events[1].record()
                if graph is None: output=self.actor.predict(x)
                else: graph.replay()
                self.events[2].record()
                self.output_cpu[:size].copy_(output,non_blocking=True)
                self.events[3].record()
        else:
            x.copy_(self.packed_cpu[:size]);self.output_cpu[:size].copy_(self.actor.predict(x))

    def ready(self):
        return not self.cuda or self.events[3].query()

    def finish(self):
        if self.cuda:
            self.events[3].synchronize()
            self.metrics["h2d_ms"]+=self.events[0].elapsed_time(self.events[1])
            self.metrics["gpu_forward_ms"]+=self.events[1].elapsed_time(self.events[2])
            self.metrics["d2h_ms"]+=self.events[2].elapsed_time(self.events[3])
        self.logits[self.indices]=self.output[:len(self.indices),:180]
        self.values[self.indices]=self.output[:len(self.indices),180]


class Slot:
    def __init__(self, batch, offset, n, threads, config, tuning, actor, graphs):
        # No per-root Python state copies; begin_range reads the native game batch.
        self.offset,self.n,self.lib=offset,n,batch._lib
        self.native_cfg=NativeConfig(**asdict(config))
        self.handle=self.lib.azul_search_create(n,threads,C.byref(self.native_cfg))
        if not self.handle: raise RuntimeError("Native search allocation failed")
        self.ready_flags=np.zeros(n,np.uint8)
        try:self.evaluator=BucketEvaluator(actor,n,graphs)
        except BaseException:
            self.close();raise
        self.policy=np.zeros((n,180),np.float32);self.actions=np.empty(n,np.uint16);self.values=np.empty(n,np.float32)
        self.stats=Stats();self.diag=np.zeros(4,np.uint64)
        self.native_tuning=NativeTuning(**asdict(tuning))
        if self.lib.azul_search_configure(self.handle,C.byref(self.native_tuning))<0:
            self.close();raise ValueError("Invalid native search tuning")
        self.future=None;self.flight=False;self.done=False
        self.request_seconds=self.submit_seconds=0.0

    def advance(self, submit=False):
        if submit:
            start=time.perf_counter()
            Search._check(self.lib.azul_search_submit(self.handle,pointer(self.evaluator.logits),pointer(self.evaluator.values)))
            self.submit_seconds+=time.perf_counter()-start
        start=time.perf_counter()
        count=self.lib.azul_search_request(self.handle,pointer(self.evaluator.observations),pointer(self.ready_flags,C.c_uint8))
        self.request_seconds+=time.perf_counter()-start
        Search._check(count)
        return count

    def close(self):
        if getattr(self,"evaluator",None) is not None and self.evaluator.cuda:
            self.evaluator.stream.synchronize()
        if self.handle:
            self.lib.azul_search_destroy(self.handle);self.handle=None


class PipelineSearch:
    def __init__(self,batch,actor,config=None,tuning=None,threads=16,queues=4,graphs=True):
        self.batch,self.actor=batch,actor
        self.config=config or SearchConfig();self.tuning=tuning or Tuning()
        if self.tuning.symmetric_cache and actor.model.kind!="equivariant":
            raise ValueError("Factory-canonical caching requires an equivariant actor")
        self.n=batch.n;self.slots=[];self.metrics={}
        if queues<1 or threads<1:raise ValueError("queues and threads must be positive")
        lib=batch._lib
        bind_search_api(lib)
        lib.azul_search_configure.argtypes=[C.c_void_p,C.POINTER(NativeTuning)];lib.azul_search_configure.restype=C.c_int
        lib.azul_search_begin_range.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_uint64,C.c_uint64,C.POINTER(C.c_uint8)]
        lib.azul_search_begin_range.restype=C.c_int
        lib.azul_search_diagnostics.argtypes=[C.c_void_p,C.POINTER(C.c_uint64)];lib.azul_search_diagnostics.restype=C.c_int
        queues=min(queues,self.n);self.executor=ThreadPoolExecutor(max_workers=queues)
        try:
            for q in range(queues):
                start=self.n*q//queues;end=self.n*(q+1)//queues
                local=copy.copy(self.tuning);local.cache_capacity=max(0,self.tuning.cache_capacity//queues)
                self.slots.append(Slot(batch,start,end-start,max(1,threads//queues),self.config,local,actor,graphs))
        except BaseException:
            self.close();raise
        self.actions=np.empty(self.n,np.uint16);self.policies=np.empty((self.n,180),np.float32);self.values=np.empty(self.n,np.float32)
        self.active=np.ones(self.n,np.uint8)

    def set_afterstate_prior(self,weight):
        for s in self.slots:
            s.native_tuning.afterstate_prior=weight
            Search._check(s.lib.azul_search_configure(s.handle,C.byref(s.native_tuning)))

    def run(self,seed,active=None):
        if active is None:self.active.fill(1)
        else:self.active[:]=active
        wall=time.perf_counter()
        for s in self.slots:
            Search._check(s.lib.azul_search_begin_range(s.handle,self.batch._handle,s.offset,seed,self.actor.version,pointer(self.active,C.c_uint8)))
            s.done=s.flight=False
            s.future=self.executor.submit(s.advance) if len(self.slots)>1 else None
        remaining=len(self.slots)
        if remaining==1:
            s=self.slots[0];count=s.advance()
            while count:
                s.evaluator.launch(s.ready_flags);s.evaluator.finish();count=s.advance(True)
            s.done=True;remaining=0
        while remaining:
            progress=False
            for s in self.slots:
                if s.done:continue
                if s.flight and s.evaluator.ready():
                    s.evaluator.finish();s.flight=False;s.future=self.executor.submit(s.advance,True);progress=True
                if s.future is not None and s.future.done():
                    count=s.future.result();s.future=None;progress=True
                    if count:
                        s.evaluator.launch(s.ready_flags);s.flight=True
                    else:
                        s.done=True;remaining-=1
            if not progress:
                # Yield the GIL rather than spinning while native workers/GPU run.
                time.sleep(0)
        metrics={"wall_seconds":time.perf_counter()-wall}
        for s in self.slots:
            Search._check(s.lib.azul_search_results(s.handle,pointer(s.policy),pointer(s.actions,C.c_uint16),pointer(s.values),C.byref(s.stats)))
            Search._check(s.lib.azul_search_diagnostics(s.handle,pointer(s.diag,C.c_uint64)))
            dst=slice(s.offset,s.offset+s.n)
            self.actions[dst]=s.actions;self.policies[dst]=s.policy;self.values[dst]=s.values
            for k,_ in s.stats._fields_:metrics[k]=metrics.get(k,0)+getattr(s.stats,k)
            for k,v in zip(("cache_hits","cache_misses","reused_nodes","afterstate_evaluations"),s.diag):metrics[k]=metrics.get(k,0)+int(v)
            for k,v in s.evaluator.metrics.items():metrics[k]=metrics.get(k,0)+v;s.evaluator.metrics[k]=0
            metrics["request_worker_seconds"]=metrics.get("request_worker_seconds",0)+s.request_seconds
            metrics["submit_worker_seconds"]=metrics.get("submit_worker_seconds",0)+s.submit_seconds
            s.request_seconds=s.submit_seconds=0.0
        self.metrics=metrics
        return self.actions,self.policies,self.values

    def close(self):
        if getattr(self,"executor",None):self.executor.shutdown(wait=True);self.executor=None
        for s in self.slots:s.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()
