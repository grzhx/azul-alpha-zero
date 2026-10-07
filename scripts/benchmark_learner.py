"""Matched synthetic replay benchmark: same model, loss, sampler and update count."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
import numpy as np
import torch
from azul_ai.model import PolicyValue
from azul_ai.optimized_learning import DeviceReplay,GraphLearner
from azul_ai.train import device_setup
p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=300);p.add_argument("--output",type=Path,default=Path("reports/learner-benchmark.json"))
args=p.parse_args();device=device_setup("cuda",2)
rng=np.random.default_rng(42);obs=rng.random((10000,172)).astype(np.float16);obs[:,168]=0;obs[::10,168]=1
mask=np.ones((10000,180),np.uint8);mask[::10]=0
data=(obs,mask.astype(np.float16)/180,mask,rng.integers(0,3,10000,dtype=np.uint8));rows=[]
for captured in (False,True):
    for rep in range(2):
        torch.manual_seed(42);model=PolicyValue(kind="equivariant").cuda()
        replay=DeviceReplay(10000,"cuda");replay.add(data,1)
        opt=torch.optim.AdamW(model.parameters(),lr=2e-4,fused=True,capturable=True)
        learner=GraphLearner(model,opt,replay,1024,graphs=captured)
        # Warmup equally before timing; synthetic replay is only for performance.
        learner.run(5);torch.cuda.synchronize();start=time.perf_counter();metrics=learner.run(args.steps)
        seconds=time.perf_counter()-start
        row=dict(graph=captured,repeat=rep,updates=args.steps,seconds=seconds,updates_per_s=args.steps/seconds,**metrics)
        print(json.dumps(row),flush=True);rows.append(row)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(rows,indent=2),encoding="utf-8")
