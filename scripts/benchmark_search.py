"""Sequential end-to-end sweeps. Fixed real games and all search work included."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
import numpy as np
import torch
from azul import Batch
from azul_ai.model import PolicyValue
from azul_ai.search import Search,SearchConfig,Evaluator
from azul_ai.selfplay import split_seed,results_view
from azul_ai.train import device_setup

p=argparse.ArgumentParser()
p.add_argument("--batches",type=int,nargs="+",default=[128,256,512,1024])
p.add_argument("--threads",type=int,nargs="+",default=[4,8])
p.add_argument("--moves",type=int,default=32)
p.add_argument("--repeats",type=int,default=2)
p.add_argument("--simulations",type=int,default=128)
p.add_argument("--eager",action="store_true")
p.add_argument("--output",type=Path,default=Path("reports/search-benchmark.json"))
args=p.parse_args()
device=device_setup("cuda",2)
torch.manual_seed(42);model=PolicyValue().to(device);results=[]
for n in args.batches:
    evaluator=Evaluator(model,n,device,not args.eager)
    evaluator.observations.fill(0);evaluator.evaluate()
    for threads in args.threads:
        for rep in range(args.repeats):
            with Batch(n,threads,42) as env,Search(env,SearchConfig(simulations=args.simulations),threads) as search:
                views=env.numpy_views();rr=results_view(env)
                start=time.perf_counter();leaves=draws=reuses=decisions=0;checksum=0
                for tick in range(args.moves):
                    actions,policy,_=search.run(evaluator,split_seed(15,tick))
                    decisions+=int((actions!=65535).sum())
                    checksum+=int(actions[actions!=65535].astype(np.uint64).sum())
                    leaves+=search.stats.evaluations;draws+=search.stats.chance_draws;reuses+=search.stats.chance_reuses
                    np.copyto(views["actions"],actions);env.step()
                    for i in np.flatnonzero(rr["terminated"]):env.reset_at(int(i),split_seed(771,tick*n+int(i)),int(i%2))
                seconds=time.perf_counter()-start
                row=dict(batch=n,threads=threads,repeat=rep,simulations=args.simulations,cuda_graph=not args.eager,
                         decisions=decisions,seconds=seconds,decisions_per_s=decisions/seconds,leaves_per_s=leaves/seconds,
                         chance_draws=draws,chance_reuses=reuses,checksum=checksum)
                print(json.dumps(row),flush=True);results.append(row)
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(results,indent=2),encoding="utf-8")
