import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
from pathlib import Path
import sys
import time
import math
from datetime import datetime, timezone
import numpy as np
import torch
from azul import Batch
from . import OPTIMIZED_VERSION
from .model import PolicyValue
from .search import SearchConfig
from .optimized_search import Tuning,FrozenActor,PipelineSearch
from .optimized_learning import DeviceReplay,GraphLearner,CheckpointJournal
from .optimized_selfplay import collect_cohort,collect_historical_cohort
from .selfplay import split_seed
from .train import device_setup,emit,save_checkpoint


def parser():
    p=argparse.ArgumentParser(description="Optimized Full Gumbel: equivariant network, pipelining, device replay")
    p.add_argument("--output",type=Path,default=Path("runs/optimized"))
    p.add_argument("--resume",type=Path)
    p.add_argument("--retune",action="store_true",help="Explicitly change LR/replay/sampling/search budget on resume; recorded as a new training phase")
    p.add_argument("--iterations",type=int,default=100)
    p.add_argument("--envs",type=int,default=2048,help="Maximum concurrent roots; cohort size adapts to positions-per-update")
    p.add_argument("--positions-per-update",type=int,default=131072)
    p.add_argument("--threads",type=int,default=16)
    p.add_argument("--queues",type=int,default=2)
    p.add_argument("--simulations",type=int,default=128)
    p.add_argument("--candidates",type=int,default=16)
    p.add_argument("--max-depth",type=int,default=96)
    p.add_argument("--max-steps",type=int,default=4096)
    p.add_argument("--chance-initial",type=int,default=4)
    p.add_argument("--chance-cap",type=int,default=64)
    p.add_argument("--chance-coefficient",type=float,default=2)
    p.add_argument("--chance-exponent",type=float,default=0.5)
    p.add_argument("--chance-sensitivity",type=float,default=1)
    p.add_argument("--q-mode",choices=["original","floor","fixed"],default="floor")
    p.add_argument("--q-floor",type=float,default=0.25)
    p.add_argument("--afterstate-prior",type=float,default=0.25)
    p.add_argument("--afterstate-loss",type=float,default=0.25)
    p.add_argument("--width",type=int,default=256)
    p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--batch-size",type=int,default=1024)
    p.add_argument("--replay-capacity",type=int,default=500000)
    p.add_argument("--recent-fraction",type=float,default=0.25)
    p.add_argument("--warmup",type=int,default=10000)
    p.add_argument("--sample-reuse",type=float,default=4)
    p.add_argument("--updates-per-iteration",type=int,default=0)
    p.add_argument("--learning-rate",type=float,default=2e-4)
    p.add_argument("--full-replay-every",type=int,default=10)
    p.add_argument("--replay-compression",choices=["gzip","none"],default="gzip")
    p.add_argument("--keep-recent",type=int,default=2)
    p.add_argument("--keep-anchors",type=int,default=4)
    p.add_argument("--cache-capacity",type=int,default=32768)
    p.add_argument("--no-subtree-reuse",action="store_true")
    p.add_argument("--no-graphs",action="store_true")
    p.add_argument("--fp32",action="store_true")
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--history-fraction",type=float,default=0.0,
                   help="Fraction of learner policy positions generated against frozen historical actors")
    p.add_argument("--history-iterations",type=int,nargs="*",default=[])
    p.add_argument("--history-weights",type=float,nargs="*",default=[])
    return p


def merge_data(parts):
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))


def allocate(total,weights):
    raw=np.asarray(weights,dtype=np.float64);raw=raw/raw.sum()*total
    result=np.floor(raw).astype(int)
    for index in np.argsort(-(raw-result))[:total-int(result.sum())]:result[index]+=1
    return result.tolist()


def smooth_weighted_index(step,weights):
    """Deterministic smooth weighted round-robin; exact over integer-weight cycles."""
    current=np.zeros(len(weights),dtype=np.float64);total=float(sum(weights));choice=0
    for _ in range(step+1):
        current+=weights;choice=int(np.argmax(current));current[choice]-=total
    return choice


def load_history_pool(root,iterations,weights,architecture):
    if not iterations:return [],[]
    if len(set(iterations))!=len(iterations):raise ValueError("Historical actor iterations must be unique")
    weights=[1.0]*len(iterations) if not weights else weights
    if len(weights)!=len(iterations) or any(not math.isfinite(x) or x<=0 for x in weights):
        raise ValueError("History weights must be positive and match history iterations")
    pool=[]
    for iteration in iterations:
        path=(root/f"actor-{iteration:06d}.pt").resolve()
        if not path.is_file() or path.parent!=root.resolve():raise FileNotFoundError(path)
        item=torch.load(path,map_location="cpu",weights_only=False)
        if item.get("engine_version")!=OPTIMIZED_VERSION or item.get("iteration")!=iteration:
            raise ValueError(f"Historical actor metadata mismatch: {path}")
        if item.get("architecture")!=architecture:raise ValueError(f"Historical actor architecture mismatch: {path}")
        with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
        pool.append(dict(iteration=iteration,path=path,payload=item,sha256=digest))
    return pool,weights


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    args=parser().parse_args(argv);payload=None;changes={};phase_history=[]
    if args.retune and not args.resume:raise ValueError("--retune requires --resume")
    if args.resume:
        payload=torch.load(args.resume,map_location="cpu",weights_only=False)
        if payload["engine_version"]!=OPTIMIZED_VERSION:raise ValueError("Use legacy train.py to resume v1 MLP runs; v2 is a new architecture")
        explicit={v.split("=")[0][2:].replace("-","_") for v in argv if v.startswith("--")}
        for k,v in payload["args"].items():
            if k in ("output","resume","iterations","device","retune"):continue
            if k in ("full_replay_every","replay_compression","keep_recent","keep_anchors") and k in explicit:continue
            if args.retune and k in explicit and k in ("learning_rate","replay_capacity","recent_fraction","sample_reuse","simulations",
                                                        "history_fraction","history_iterations","history_weights"):
                if getattr(args,k)!=v:changes[k]={"old":v,"new":getattr(args,k)}
                continue
            if k in explicit and getattr(args,k)!=v:raise ValueError(f"Cannot change protocol field on exact resume: {k}")
            setattr(args,k,v)
        for k,old in (("history_fraction",0.0),("history_iterations",[]),("history_weights",[])):
            if k not in payload["args"] and k in explicit:
                if not args.retune:raise ValueError(f"Enabling --{k.replace('_','-')} requires --retune")
                if getattr(args,k)!=old:changes[k]={"old":old,"new":getattr(args,k)}
        if "output" not in explicit:args.output=args.resume.parent
    elif (args.output/"latest.pt").exists() or (args.output/"initial.pt").exists():
        raise FileExistsError("Run exists; use --resume or a new output directory")
    if min(args.envs,args.queues,args.threads,args.iterations,args.positions_per_update,args.max_steps,args.batch_size,
           args.replay_capacity,args.width,args.blocks,args.full_replay_every,args.keep_recent,args.keep_anchors)<1:raise ValueError("Counts must be positive")
    if not 0<=args.recent_fraction<=1 or args.sample_reuse<=0 or args.afterstate_loss<0 or args.warmup<0:
        raise ValueError("Invalid learner parameters")
    if not 0<=args.history_fraction<1:raise ValueError("History fraction must be in [0, 1)")
    if bool(args.history_fraction)!=bool(args.history_iterations):
        raise ValueError("Positive history fraction requires historical actors, and vice versa")
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.sample_reuse):
        raise ValueError("Learning rate and reuse must be finite and positive")
    if payload and args.replay_capacity<payload["args"]["replay_capacity"]:
        raise ValueError("Retune cannot discard replay through capacity reduction")
    device=device_setup(args.device,2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=PolicyValue(args.width,args.blocks,kind="equivariant").to(device)
    pool,history_weights=load_history_pool(args.output.resolve(),args.history_iterations,args.history_weights,model.architecture)
    pool_manifest=[dict(iteration=x["iteration"],sha256=x["sha256"],weight=w)
                   for x,w in zip(pool,history_weights)]
    history_changed=any(k in changes for k in ("history_fraction","history_iterations","history_weights"))
    if payload and not history_changed and payload.get("history_pool") is not None and payload["history_pool"]!=pool_manifest:
        raise ValueError("Historical actor pool changed since this checkpoint was written")
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4,
        fused=device.type=="cuda",capturable=device.type=="cuda")
    replay=None
    search_config=SearchConfig(simulations=args.simulations,candidates=args.candidates,max_depth=args.max_depth,
        chance_initial=args.chance_initial,chance_cap=args.chance_cap)
    tuning=Tuning(q_mode={"original":0,"floor":1,"fixed":2}[args.q_mode],q_floor=args.q_floor,
        chance_coefficient=args.chance_coefficient,chance_exponent=args.chance_exponent,
        chance_sensitivity=args.chance_sensitivity,afterstate_prior=args.afterstate_prior,
        subtree_reuse=int(not args.no_subtree_reuse),cache_capacity=args.cache_capacity)
    start_iteration=updates=total_games=0;credit=0.0;mean_length=80.0
    if payload:
        model.load_state_dict(payload["model"]);optimizer.load_state_dict(payload["optimizer"])
        # load_state_dict restores the old LR too; an explicit retune must replace it.
        for group in optimizer.param_groups:
            group["lr"]=args.learning_rate
            group["capturable"]=device.type=="cuda"
            group["fused"]=device.type=="cuda"
        old_capacity=payload["args"]["replay_capacity"]
        if args.replay_capacity!=old_capacity:
            original=DeviceReplay(old_capacity,"cpu",payload["args"]["recent_fraction"])
            CheckpointJournal.restore(args.resume,original,payload)
            replay=original.expanded(args.replay_capacity,device,args.recent_fraction)
            del original
        else:
            replay=DeviceReplay(args.replay_capacity,device,args.recent_fraction)
            CheckpointJournal.restore(args.resume,replay,payload)
        start_iteration,updates,total_games,credit,mean_length=(payload[k] for k in ("iteration","updates","games","credit","mean_length"))
        if any(x["iteration"]>start_iteration for x in pool):
            raise ValueError("Historical opponents cannot be newer than the resumed learner checkpoint")
        phase_history=list(payload.get("phase_history",[]))
        if changes:
            phase_history.append(dict(after_iteration=start_iteration,changes=changes,
                source_checkpoint=str(args.resume.resolve()),utc=datetime.now(timezone.utc).isoformat(),
                note="User-authorized continuation with retuned protocol; all weights, optimizer moments and replay retained",
                history_pool=pool_manifest))
        if pool and payload.get("history_schedule") is None:
            phase_history.append(dict(after_iteration=start_iteration,
                changes={"history_schedule":{"old":"all_pool_each_iteration","new":"smooth_weighted_single"}},
                source_checkpoint=str(args.resume.resolve()),utc=datetime.now(timezone.utc).isoformat(),
                note="Equivalent long-run history weights with one batched opponent per publication",
                history_pool=pool_manifest))
    else:replay=DeviceReplay(args.replay_capacity,device,args.recent_fraction)
    actor=FrozenActor(model,device,not args.fp32)
    args.output.mkdir(parents=True,exist_ok=True);log=args.output/"metrics.jsonl"
    journal=CheckpointJournal(args.output,args.full_replay_every,args.replay_compression,args.keep_recent,args.keep_anchors)
    if payload and args.output.resolve()==args.resume.parent.resolve():
        journal.base=payload["replay_manifest"]["base"];journal.deltas=list(payload["replay_manifest"]["deltas"])
    if "replay_capacity" in changes:journal.force_full=True
    if changes:emit(dict(event="retune",**phase_history[-1],remaining_sample_credit=credit),log)
    emit(dict(event="start",engine=OPTIMIZED_VERSION,device=str(device),parameters=sum(p.numel() for p in model.parameters()),
        architecture=model.architecture,search=asdict(search_config),tuning=asdict(tuning),resumed_iteration=start_iteration,
        replay_device=str(replay.device),graphs=not args.no_graphs,queues=args.queues,
        learning_rate=optimizer.param_groups[0]["lr"],replay_capacity=replay.capacity,
        recent_fraction=replay.recent_fraction,sample_reuse=args.sample_reuse,
        history_fraction=args.history_fraction,
        history_pool=pool_manifest),log)
    learner=None
    with ExitStack() as stack:
        env=stack.enter_context(Batch(args.envs,args.threads,args.seed))
        search=stack.enter_context(PipelineSearch(env,actor,search_config,tuning,args.threads,args.queues,not args.no_graphs))
        history_model=history_actor=history_search=None
        if pool:
            history_model=PolicyValue(**model.architecture)
            history_model.load_state_dict(pool[0]["payload"]["model"])
            history_actor=FrozenActor(history_model,device,not args.fp32)
            history_search=stack.enter_context(PipelineSearch(env,history_actor,search_config,tuning,
                                                               args.threads,args.queues,not args.no_graphs))
        if payload:
            torch.set_rng_state(payload["torch_rng"])
            if device.type=="cuda" and payload["cuda_rng"] is not None:torch.cuda.set_rng_state_all(payload["cuda_rng"])
        else:
            save_checkpoint(args.output/"initial.pt",dict(engine_version=OPTIMIZED_VERSION,architecture=model.architecture,
                model=model.state_dict(),iteration=0,search=asdict(search_config),tuning=asdict(tuning)))
        for iteration in range(start_iteration,args.iterations):
            actor.publish(model,iteration+1)
            # The afterstate head is first taught by completed self-play before it contributes a prior.
            search.set_afterstate_prior(args.afterstate_prior if updates>0 else 0.0)
            target=args.positions_per_update
            self_games=min(args.envs,max(1,round(target*(1-args.history_fraction)/mean_length)))
            history_games=0 if not pool else max(1,round(2*target*args.history_fraction/mean_length))
            if history_games>args.envs:raise ValueError("Historical cohort exceeds --envs")
            parts=[];components=[]
            data,component=collect_cohort(env,search,split_seed(args.seed,iteration),self_games,args.max_steps,
                                          lambda row:emit(row,log))
            component.update(component="current_selfplay",iteration=iteration+1)
            parts.append(data);components.append(component);emit(component,log)
            if pool:
                pool_index=smooth_weighted_index(iteration,history_weights)
                entry=pool[pool_index];count=history_games
                history_model.load_state_dict(entry["payload"]["model"])
                history_actor.publish(history_model,(iteration+1)*len(pool)+pool_index+1)
                data,component=collect_historical_cohort(env,search,history_search,
                    split_seed(args.seed^0xD1B54A32D192ED03,(iteration+1)*len(pool)+pool_index),count,
                    entry["iteration"],args.max_steps,lambda row:emit(row,log))
                component["iteration"]=iteration+1
                parts.append(data);components.append(component);emit(component,log)
            data=merge_data(parts)
            metrics=dict(event="selfplay_mixed",iteration=iteration+1,
                games=sum(x["games"] for x in components),truncated=sum(x["truncated"] for x in components),
                decisions=sum(x["decisions"] for x in components),positions=len(data[0]),
                decision_positions=sum(x["decision_positions"] for x in components),
                afterstate_positions=sum(x["afterstate_positions"] for x in components),
                selfplay_games=components[0]["games"],historical_games=sum(x["games"] for x in components[1:]),
                requested_history_fraction=args.history_fraction,
                realized_history_policy_fraction=sum(x["decision_positions"] for x in components[1:])/
                    max(1,sum(x["decision_positions"] for x in components)),
                component_seconds=sum(x["seconds"] for x in components),
                historical_results=[dict(opponent_iteration=x["opponent_iteration"],games=x["games"],
                                         learner_score_rate=x["learner_score_rate"]) for x in components[1:]])
            emit(metrics,log)
            if not metrics["games"]:raise RuntimeError("No completed games; truncations are not draw targets")
            total_games+=metrics["games"]
            mean_length=0.7*mean_length+0.3*(metrics["decisions"]/metrics["games"])
            t=time.perf_counter();replay.add(data,iteration+1);ingest=time.perf_counter()-t
            credit+=metrics["decision_positions"]*args.sample_reuse
            n_updates=int(credit//args.batch_size) if replay.size>=args.warmup else 0
            if args.updates_per_iteration:n_updates=min(n_updates,args.updates_per_iteration)
            if n_updates and learner is None:
                learner=GraphLearner(model,optimizer,replay,args.batch_size,args.afterstate_loss,not args.no_graphs,not args.fp32)
            t=time.perf_counter();losses=learner.run(n_updates) if n_updates else {}
            updates+=n_updates;credit-=n_updates*args.batch_size
            emit(dict(event="train",iteration=iteration+1,updates=n_updates,total_updates=updates,
                seconds=time.perf_counter()-t,ingest_seconds=ingest,
                gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1048576 if device.type=="cuda" else 0,
                **losses,**replay.statistics()),log)
            state=dict(engine_version=OPTIMIZED_VERSION,architecture=model.architecture,search=asdict(search_config),tuning=asdict(tuning),
                args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},model=model.state_dict(),optimizer=optimizer.state_dict(),
                iteration=iteration+1,updates=updates,games=total_games,credit=credit,mean_length=mean_length,
                phase_history=phase_history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None)
            state["history_pool"]=pool_manifest
            state["history_schedule"]="smooth_weighted_single" if pool else None
            t=time.perf_counter();journal.save(state,replay,data,iteration+1)
            emit(dict(event="checkpoint",iteration=iteration+1,path=str(args.output/"latest.pt"),
                      seconds=time.perf_counter()-t,replay_base=journal.base,replay_deltas=len(journal.deltas)),log)


if __name__=="__main__":main()
