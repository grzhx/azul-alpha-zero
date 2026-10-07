from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from azul3 import Batch3
from azul_ai.model3 import PolicyValue3
from azul_ai.search3 import Search3,SearchConfig3,Evaluator3
class MixedEvaluator:
 def __init__(self,cand,opp,n,dev,seat):
  self.c=Evaluator3(cand,n,dev,False); self.o=Evaluator3(opp,n,dev,False); self.observations=self.c.observations; self.mask=np.zeros(n,dtype=bool)
 def evaluate(self):
  np.copyto(self.o.observations,self.observations);cp,cv=self.c.evaluate();op,ov=self.o.evaluate();sel=~self.mask;cp[sel]=op[sel];cv[sel]=ov[sel];return cp,cv

def load(path,dev):
 ck=torch.load(path,map_location="cpu",weights_only=False);m=PolicyValue3(width=ck["architecture"]["width"],blocks=ck["architecture"]["blocks"]);m.load_state_dict(ck["model"]);return m.to(dev).eval()
def seed(base,i):
 x=(base+0x9e3779b97f4a7c15*(i+1))&((1<<64)-1);x^=x>>30;x=(x*0xbf58476d1ce4e5b9)&((1<<64)-1);x^=x>>27;return (x^(x>>31))&((1<<64)-1)
def bootstrap(x,rng,n=10000):
 if len(x)==0:return None
 z=np.empty(n,np.float64)
 for i in range(n):z[i]=x[rng.integers(len(x),size=len(x))].mean()
 return [float(np.quantile(z,.025)),float(np.quantile(z,.975))]
def main(a):
 dev=torch.device("cuda" if a.device=="auto" and torch.cuda.is_available() else a.device if a.device!="auto" else "cpu");cand=load(a.candidate,dev);opp=load(a.opponent,dev) if a.opponent else None;cfg=__import__('azul_ai.search3',fromlist=['SearchConfig3']).SearchConfig3(simulations=a.simulations,candidates=min(a.candidates,a.simulations),max_depth=a.max_depth);records=[];rng=np.random.default_rng(a.seed);start=time.perf_counter()
 for off in range(0,a.pairs,a.batch_pairs):
  pairs=min(a.batch_pairs,a.pairs-off);n=3*pairs
  with Batch3(n,a.threads,a.seed+off) as env,Search3(env,cfg,a.threads) as search:
   views=env.numpy_views();seat=np.tile(np.arange(3,dtype=np.int8),pairs);active=np.ones(n,bool);score=np.full(n,np.nan)
   for i in range(n): env.reset_at(i,seed(a.seed,off+i),int((off+i+1)%3))
   evc=Evaluator3(cand,n,dev,not a.no_graphs) if opp is None else MixedEvaluator(cand,opp,n,dev,seat)
   for tick in range(a.max_steps):
    env.observe();
    if opp is not None: evc.mask=(views["players"]==seat)
    actions,pi,_=search.run(evc,seed(a.seed^0x1234,off*a.max_steps+tick));
    if opp is None:
     rnd=np.where(views["masks"],rng.random((n,240)),-1).argmax(1);actions[views["players"]!=seat]=rnd[views["players"]!=seat]
    views["actions"][:]=actions;env.step();done=np.array([bool(x.terminated) for x in env.results]);
    for i in np.flatnonzero(done&active):
     sc=np.array(env.results[i].scores);best=sc.max();score[i]=1. if sc[seat[i]]==best and (sc==best).sum()==1 else .5 if sc[seat[i]]==best else 0.;
    active[done]=False
    if not active.any():break
   for i in range(n):records.append({"pair":off+i//3,"seat":int(seat[i]),"score":None if np.isnan(score[i]) else float(score[i]),"scores":list(map(int,env.results[i].scores))})
  print(json.dumps({"event":"evaluation_progress","games":len(records),"seconds":time.perf_counter()-start}),flush=True)
 pair_scores=[]
 for p in range(a.pairs):
  rs=[r["score"] for r in records if r["pair"]==p and r["score"] is not None]
  if len(rs)==3:pair_scores.append(float(np.mean(rs)))
 x=np.array(pair_scores);ci=bootstrap(x,rng);report={"candidate":str(a.candidate),"opponent":str(a.opponent) if a.opponent else "random","pairs":a.pairs,"complete_pairs":len(x),"mean_score":float(x.mean()) if len(x) else None,"bootstrap_95":ci,"promotion_eligible":bool(len(x)>=a.pairs and len(x)>=100 and ci and ci[0]>.5),"records":records,"search":{k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2),encoding="utf-8");print(json.dumps({k:v for k,v in report.items() if k!="records"}),flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--candidate",type=Path,required=True);p.add_argument("--opponent",type=Path);p.add_argument("--pairs",type=int,default=300);p.add_argument("--batch-pairs",type=int,default=32);p.add_argument("--simulations",type=int,default=128);p.add_argument("--candidates",type=int,default=32);p.add_argument("--threads",type=int,default=8);p.add_argument("--max-steps",type=int,default=4096);p.add_argument("--max-depth",type=int,default=128);p.add_argument("--seed",type=int,default=39003);p.add_argument("--device",choices=["auto","cuda","cpu"],default="auto");p.add_argument("--output",type=Path,default=Path("runs/three_player/evaluation.json"));p.add_argument("--no-graphs",action="store_true");main(p.parse_args())
