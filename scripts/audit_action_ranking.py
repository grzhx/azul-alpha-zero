"""Bounded continuation pipeline audit; never trains or changes checkpoints."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from azul import Batch
from azul_ai.evaluate import load_actor
from azul_ai.action_ranking import RankingConfig,ContinuationRunner
from azul_ai.optimized_search import FrozenActor,PipelineSearch,Tuning
from azul_ai.optimized_selfplay import collect_cohort
from azul_ai.search import SearchConfig
from azul_ai.train import device_setup,emit
from azul_ai.checkpoint_storage import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",type=Path,default=ROOT/"runs/value_calibrated_from_1900/actor-002200.pt")
    p.add_argument("--library",type=Path,default=None)
    p.add_argument("--output",type=Path,default=ROOT/"reports/value_ranking_smoke_2200.json")
    p.add_argument("--device",default="cuda")
    p.add_argument("--games",type=int,default=8)
    p.add_argument("--roots",type=int,default=4)
    p.add_argument("--envs",type=int,default=128)
    p.add_argument("--collection-simulations",type=int,default=128)
    args=p.parse_args()
    if min(args.games,args.roots,args.envs)<1:raise ValueError("Counts must be positive")
    start=time.perf_counter();device=device_setup(args.device,2)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    model=load_actor(args.checkpoint,device);actor=FrozenActor(model,device)
    actor.publish(model,int(checkpoint["iteration"]))
    search_config=SearchConfig(**checkpoint["search"])
    tuning=Tuning(**checkpoint["tuning"])
    config=RankingConfig(positions=args.roots,envs=args.envs)
    collector=SearchConfig(**{**asdict(search_config),"simulations":args.collection_simulations,
                             "candidates":min(search_config.candidates,args.collection_simulations)})
    with Batch(args.games,8,library=args.library) as env,PipelineSearch(
        env,actor,collector,tuning,8,1) as search:
        _,metrics=collect_cohort(env,search,771403,args.games,ranking_config=config,ranking_seed=773130,
                                progress=emit)
        anchors=metrics.pop("ranking_anchors")
    emit(dict(event="anchors",rounds=[a["round"] for a in anchors],count=len(anchors)))
    with ContinuationRunner(actor,search_config,tuning,config,8,2,True,args.library) as runner:
        data,ranking=runner.evaluate(anchors,443312,emit)
    records=[]
    if data is not None:
        for i in range(len(data["root"])):
            candidates=[]
            for j in range(config.topk):
                if data["attempted"][i,j]==0:continue
                counts=data["counts"][i,j].astype(int).tolist()
                candidates.append(dict(action=int(data["actions"][i,j]),wdl=counts,
                    value_wdl=data["value_counts"][i,j].astype(int).tolist(),
                    attempted=int(data["attempted"][i,j]),valid=bool(data["valid"][i,j]),
                    exact=bool(data["exact"][i,j]),root_sign=float(data["sign"][i,j]),
                    mean_score_margin=float(data["score_sum"][i,j]/max(1,sum(counts)))))
            records.append(dict(round=int(data["round"][i]),candidates=candidates,
                                confident_pairs=data["pairs"][i].astype(int).tolist()))
    report=dict(checkpoint=str(args.checkpoint),generation=int(checkpoint["iteration"]),
        library=str(args.library),device=str(device),collection=metrics,ranking=ranking,
        seconds=time.perf_counter()-start,records=records,training_updates=0,
        limitation="Pipeline validation only; small sample is not a strength measurement.")
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,report)
    emit({k:v for k,v in report.items() if k not in ("records","collection")})


if __name__=="__main__":main()
