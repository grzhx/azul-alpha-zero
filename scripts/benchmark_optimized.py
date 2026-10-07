import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
import numpy as np
import torch
from azul import Batch
from azul_ai.model import PolicyValue
from azul_ai.search import SearchConfig
from azul_ai.optimized_search import FrozenActor,PipelineSearch,Tuning
from azul_ai.selfplay import split_seed,results_view
from azul_ai.train import device_setup

p=argparse.ArgumentParser();p.add_argument("--batches",type=int,nargs="+",default=[1024,2048])
p.add_argument("--queues",type=int,nargs="+",default=[1,2,4]);p.add_argument("--threads",type=int,default=16)
p.add_argument("--moves",type=int,default=20);p.add_argument("--simulations",type=int,default=128)
p.add_argument("--repeats",type=int,default=2);p.add_argument("--output",type=Path,default=Path("reports/optimized-benchmark.json"))
p.add_argument("--disable-reuse",action="store_true");p.add_argument("--legacy-network",action="store_true")
args=p.parse_args();device=device_setup("cuda",2);torch.manual_seed(42)
model=PolicyValue(kind="mlp" if args.legacy_network else "equivariant").to(device)
actor=FrozenActor(model,device);rows=[]
for n in args.batches:
    for queues in args.queues:
        tuning=Tuning(afterstate_prior=0,subtree_reuse=int(not args.disable_reuse),
            cache_capacity=0 if args.disable_reuse else 32768,symmetric_cache=int(not args.legacy_network))
        with Batch(n,args.threads,42) as env,PipelineSearch(env,actor,SearchConfig(simulations=args.simulations),tuning,args.threads,queues) as search:
            for rep in range(args.repeats):
                env.reset(42);actor.publish(model,actor.version+1)
                v=env.numpy_views();rr=results_view(env);metrics={};start=time.perf_counter();checksum=0
                for tick in range(args.moves):
                    a,_,_=search.run(split_seed(15,tick));v["actions"][:]=a;env.step();checksum+=int(a.astype(np.uint64).sum())
                    for k,x in search.metrics.items():metrics[k]=metrics.get(k,0)+x
                    for i in np.flatnonzero(rr["terminated"]):env.reset_at(int(i),split_seed(71,tick*n+int(i)),int(i%2))
                seconds=time.perf_counter()-start
                row=dict(batch=n,queues=queues,threads=args.threads,repeat=rep,moves=args.moves,
                    simulations_per_move=args.simulations,seconds=seconds,decisions_per_s=n*args.moves/seconds,checksum=checksum,**metrics)
                print(json.dumps(row),flush=True);rows.append(row)
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(rows,indent=2),encoding="utf-8")
