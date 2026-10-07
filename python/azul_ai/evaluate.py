import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from azul import Batch
from . import ENGINE_VERSION, OPTIMIZED_VERSION
from .model import PolicyValue
from .search import Search, SearchConfig, Evaluator
from .selfplay import results_view, split_seed
from .train import device_setup


def load_actor(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["engine_version"] not in (ENGINE_VERSION, OPTIMIZED_VERSION):
        raise ValueError("Engine version mismatch")
    model = PolicyValue(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


class MatchEvaluator:
    """Each root uses the actor choosing the REAL move, for its whole tree.

    At hypothetical opponent turns that actor's network still evaluates from
    the requested current-player perspective; native search handles the sign.
    Do not dispatch hypothetical leaves by their side-to-move to another model.
    """
    def __init__(self, candidate, opponent, n, device, graph=True):
        self.candidate = Evaluator(candidate,n,device,graph)
        self.opponent = Evaluator(opponent,n,device,graph) if opponent is not None else None
        self.observations = self.candidate.observations
        self.candidate_roots = np.ones(n,dtype=bool)
        self.logits = np.empty((n,180),np.float32)
        self.values = np.empty(n,np.float32)

    def evaluate(self):
        if self.opponent is None or self.candidate_roots.all():
            return self.candidate.evaluate()
        np.copyto(self.opponent.observations,self.observations)
        if not self.candidate_roots.any():
            return self.opponent.evaluate()
        cp,cv=self.candidate.evaluate();op,ov=self.opponent.evaluate()
        np.copyto(self.logits,op); np.copyto(self.values,ov)
        self.logits[self.candidate_roots]=cp[self.candidate_roots]
        self.values[self.candidate_roots]=cv[self.candidate_roots]
        return self.logits,self.values


def paired_interval(pair_scores, seed=1, resamples=10000):
    rng=np.random.default_rng(seed)
    means=[]
    for _ in range(0,resamples,100):
        means.extend(pair_scores[rng.integers(len(pair_scores),size=(min(100,resamples-len(means)),len(pair_scores)))].mean(1))
    return np.quantile(means,[0.025,0.975]).tolist()


def main(argv=None):
    p=argparse.ArgumentParser(description="Paired-seat Gumbel checkpoint tournament; no training updates")
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--opponent",default="random",help="checkpoint path or random")
    p.add_argument("--pairs",type=int,default=500)
    p.add_argument("--batch-pairs",type=int,default=64)
    p.add_argument("--simulations",type=int,default=512)
    p.add_argument("--candidates",type=int,default=32)
    p.add_argument("--threads",type=int,default=8)
    p.add_argument("--seed",type=int,default=908127)
    p.add_argument("--max-steps",type=int,default=4096)
    p.add_argument("--device",choices=["auto","cuda","cpu"],default="auto")
    p.add_argument("--output",type=Path,default=Path("runs/evaluation.json"))
    p.add_argument("--no-cuda-graph",action="store_true")
    args=p.parse_args(argv)
    if min(args.pairs,args.batch_pairs,args.max_steps,args.threads)<1:
        raise ValueError("Counts must be positive")
    device=device_setup(args.device,2)
    candidate=load_actor(args.candidate,device)
    opponent=None if args.opponent=="random" else load_actor(Path(args.opponent),device)
    cfg=SearchConfig(simulations=args.simulations,candidates=args.candidates,gumbel_scale=0,
                     max_depth=128,chance_initial=4,chance_cap=128)
    records=[]; start=time.perf_counter()
    for offset in range(0,args.pairs,args.batch_pairs):
        pairs=min(args.batch_pairs,args.pairs-offset); n=2*pairs
        with Batch(n,args.threads) as env,Search(env,cfg,args.threads) as search:
            for i in range(n):
                env.reset_at(i,split_seed(args.seed,offset+i//2),0)
            ev=MatchEvaluator(candidate,opponent,n,device,not args.no_cuda_graph)
            views=env.numpy_views();results=results_view(env)
            seat=np.arange(n)%2;active=np.ones(n,bool);scores=np.full(n,np.nan)
            policy_rng=np.random.default_rng(split_seed(args.seed^0xCAFE,offset))
            for tick in range(args.max_steps):
                env.observe()
                ev.candidate_roots=views["players"]==seat
                actions,_,_=search.run(ev,split_seed(args.seed^0xE7037ED1A0B428DB,offset*args.max_steps+tick))
                if opponent is None:
                    # Uniform among legal moves. Terminal rows keep 65535.
                    random_actions=np.where(views["masks"],policy_rng.random((n,180)),-1).argmax(1)
                    use_random=active & ~ev.candidate_roots
                    actions[use_random]=random_actions[use_random]
                np.copyto(views["actions"],actions);env.step()
                if results["invalid_action"][active].any():
                    raise RuntimeError("Illegal tournament action")
                done=active & results["terminated"].astype(bool)
                reward=results["reward"]
                absolute_winner=np.where(reward>0,results["actor"],1-results["actor"])
                scores[done]=np.where(reward[done]==0,0.5,(absolute_winner[done]==seat[done]).astype(float))
                active[done]=False
                if not active.any():break
            for i in range(n):
                records.append(dict(pair=offset+i//2,candidate_seat=int(seat[i]),
                                    score=None if active[i] else float(scores[i]),
                                    scores=results["scores"][i].astype(int).tolist()))
        print(json.dumps(dict(event="evaluation_progress",games=len(records),seconds=time.perf_counter()-start)),flush=True)
    pair_scores=[]
    for i in range(0,len(records),2):
        if records[i]["score"] is not None and records[i+1]["score"] is not None:
            pair_scores.append((records[i]["score"]+records[i+1]["score"])/2)
    completed=[r["score"] for r in records if r["score"] is not None]
    interval=paired_interval(np.array(pair_scores)) if pair_scores else None
    # All pairs must finish; an incomplete pair never turns into a half-point draw.
    mean=float(np.mean(pair_scores)) if pair_scores else None
    report=dict(candidate=str(args.candidate),opponent=args.opponent,search=asdict(cfg),seed=args.seed,
                games=len(records),wins=completed.count(1.0),draws=completed.count(0.5),losses=completed.count(0.0),
                truncated=len(records)-len(completed),complete_pairs=len(pair_scores),paired_score_rate=mean,
                paired_bootstrap_95=interval,seconds=time.perf_counter()-start,
                promotion_eligible=bool(len(pair_scores)>=500 and len(completed)==len(records) and mean>=0.55 and interval[0]>0.5),
                records=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="records"}),flush=True)


if __name__=="__main__":main()
