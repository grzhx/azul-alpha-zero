"""Read-only checkpoint audit: paired-seat v2 matches and search budget comparison."""
import argparse,json,sys,time
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
import numpy as np
import torch
from azul import Batch
from azul_ai.evaluate import load_actor,paired_interval
from azul_ai.optimized_search import FrozenActor,PipelineSearch,Tuning
from azul_ai.search import SearchConfig
from azul_ai.selfplay import split_seed,results_view
from azul_ai.train import device_setup

def match(candidate,opponent,budget_a,budget_b,pairs,batch_pairs,seed,output,disable_a=None):
    device=device_setup("cuda",2)
    actors=[FrozenActor(load_actor(p,device),device) for p in (candidate,opponent)]
    saved=[torch.load(p,map_location="cpu",weights_only=False) for p in (candidate,opponent)]
    records=[];start=time.perf_counter();totals=[0,0];search_time=[0.,0.];calibration=[]
    for offset in range(0,pairs,batch_pairs):
        count=min(batch_pairs,pairs-offset);n=count*2
        with Batch(n,8) as env:
            searches=[]
            try:
                for j,(actor,checkpoint,budget) in enumerate(zip(actors,saved,(budget_a,budget_b))):
                    cfg=SearchConfig(**checkpoint["search"]);cfg.simulations=budget;cfg.gumbel_scale=0
                    tuning=Tuning(**checkpoint["tuning"])
                    if j==0 and disable_a=="afterstate":tuning.afterstate_prior=0
                    if j==0 and disable_a=="reuse":tuning.subtree_reuse=0
                    searches.append(PipelineSearch(env,actor,cfg,tuning,threads=8,queues=1))
                for i in range(n):env.reset_at(i,split_seed(seed,offset+i//2),0)
                views=env.numpy_views();result=results_view(env);seat=np.arange(n)%2
                active=np.ones(n,bool);winners=np.full(n,-2,np.int8);history=[]
                for tick in range(4096):
                    env.observe();current=views["players"].copy();use_a=current==seat
                    actions=np.full(n,65535,np.uint16)
                    # Each actor searches only its actual turns. Same protocol for both.
                    for j,mask in enumerate((active&use_a,active&~use_a)):
                        if not mask.any():continue
                        t=time.perf_counter()
                        a,_,_=searches[j].run(split_seed(seed^0xA35F,offset*4096+tick),mask)
                        search_time[j]+=time.perf_counter()-t;totals[j]+=int(mask.sum());actions[mask]=a[mask]
                    # Out-of-training outcomes; report network calibration separately from loss.
                    if tick%8==0:
                        ids=np.flatnonzero(active&use_a)
                        if len(ids):
                            with torch.inference_mode():
                                x=torch.from_numpy(views["observations"][ids].copy()).to(device,dtype=actors[0].dtype)
                                _,v=actors[0].model(x);probs=v.float().softmax(-1).cpu().numpy()
                            history.extend((int(i),int(current[i]),p) for i,p in zip(ids,probs))
                    views["actions"][:]=actions;env.step()
                    if result["invalid_action"][active].any():raise RuntimeError("illegal audit action")
                    done=active&result["terminated"].astype(bool);r=result["reward"]
                    absolute=np.where(r>0,result["actor"],1-result["actor"]).astype(np.int8)
                    winners[done]=np.where(r[done]==0,-1,absolute[done]);active[done]=False
                    if not active.any():break
                for i in range(n):
                    records.append(dict(pair=offset+i//2,seat=int(seat[i]),
                        score=None if active[i] else 0.5 if winners[i]==-1 else float(winners[i]==seat[i]),
                        scores=result["scores"][i].astype(int).tolist()))
                for i,player,probs in history:
                    if winners[i]!=-2:calibration.append((probs,1 if winners[i]==-1 else 0 if winners[i]==player else 2))
            finally:
                for search in searches:search.close()
        print(json.dumps(dict(event="audit_progress",games=len(records),seconds=time.perf_counter()-start)),flush=True)
    pair_scores=np.array([(records[i]["score"]+records[i+1]["score"])/2 for i in range(0,len(records),2)
                          if records[i]["score"] is not None and records[i+1]["score"] is not None])
    outcomes=[r["score"] for r in records if r["score"] is not None]
    probability=np.array([r[0] for r in calibration]);label=np.array([r[1] for r in calibration])
    report=dict(candidate=str(candidate),opponent=str(opponent),simulations=[budget_a,budget_b],seed=seed,
        games=len(records),wins=outcomes.count(1.),draws=outcomes.count(.5),losses=outcomes.count(0.),
        truncated=len(records)-len(outcomes),score_rate=float(pair_scores.mean()),ci95=paired_interval(pair_scores,seed),
        search_decisions=totals,search_seconds=search_time,seconds=time.perf_counter()-start,
        heldout_positions=len(label),wdl_brier=float(((probability-np.eye(3)[label])**2).sum(1).mean()),
        wdl_cross_entropy=float(-np.log(probability[np.arange(len(label)),label].clip(1e-8)).mean()),
        candidate_config=asdict(searches[0].config),candidate_tuning=asdict(searches[0].tuning),records=records)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="records"}),flush=True)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--candidate",type=Path,required=True);p.add_argument("--opponent",type=Path,required=True)
    p.add_argument("--budget-a",type=int,default=128);p.add_argument("--budget-b",type=int,default=128)
    p.add_argument("--pairs",type=int,default=500);p.add_argument("--batch-pairs",type=int,default=128)
    p.add_argument("--seed",type=int,default=7301901);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--disable-a",choices=["afterstate","reuse"])
    a=p.parse_args();match(a.candidate,a.opponent,a.budget_a,a.budget_b,a.pairs,a.batch_pairs,a.seed,a.output,a.disable_a)
