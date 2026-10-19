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
from azul import Batch, _load
from . import OPTIMIZED_VERSION
from .model import PolicyValue, upgraded_architecture, load_compatible_optimizer
from .search import SearchConfig
from .search_defaults import CHANCE_CAP, require_current_search
from .policy_diagnostics import policy_metrics
from .optimized_search import Tuning,FrozenActor,PipelineSearch
from .optimized_learning import DeviceReplay,GraphLearner,CheckpointJournal
from .optimized_selfplay import collect_cohort,collect_historical_cohort
from .action_ranking import RankingConfig,RankingReplay,ContinuationRunner
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
    p.add_argument("--high-simulations",type=int,default=0,
                   help="Optional high-budget self-play simulations")
    p.add_argument("--high-budget-period",type=int,default=4,
                   help="Use high budget every N publications; 0 disables mixing")
    p.add_argument("--candidates",type=int,default=16)
    p.add_argument("--max-depth",type=int,default=96)
    p.add_argument("--max-steps",type=int,default=4096)
    p.add_argument("--chance-initial",type=int,default=4)
    p.add_argument("--chance-cap",type=int,default=CHANCE_CAP)
    p.add_argument("--chance-coefficient",type=float,default=2)
    p.add_argument("--chance-exponent",type=float,default=0.5)
    p.add_argument("--chance-sensitivity",type=float,default=1)
    p.add_argument("--q-mode",choices=["original","floor","fixed"],default="floor")
    p.add_argument("--q-floor",type=float,default=0.25)
    p.add_argument("--afterstate-prior",type=float,default=0.25)
    p.add_argument("--afterstate-loss",type=float,default=0.25)
    p.add_argument("--width",type=int,default=256)
    p.add_argument("--blocks",type=int,default=4)
    p.add_argument("--end-features",action="store_true",help="Add 45 rule-derived ending features")
    p.add_argument("--value-tower",action="store_true",help="Use a separate residual value tower")
    p.add_argument("--value-tower-blocks",type=int,default=2)
    p.add_argument("--value-rank-weight",type=float,default=0.0,
                   help="Experimental terminal-utility MSE weight; does not supervise action rankings")
    p.add_argument("--value-label-smoothing",type=float,default=0.0)
    p.add_argument("--policy-loss-weight",type=float,default=1.0)
    p.add_argument("--value-loss-weight",type=float,default=1.0)
    p.add_argument("--init-from",type=Path,help="Initialize a new run from an optimized actor with a fresh optimizer")
    p.add_argument("--validation-games",type=int,default=0,help="Complete first-cohort games reserved outside replay")
    p.add_argument("--validation-every",type=int,default=10)
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
    p.add_argument("--native-library",type=Path,help="Optional isolated training DLL")
    p.add_argument("--release-cohort-arenas",action="store_true",
                   help="Reclaim obsolete CPU search arenas between complete cohorts; inference graphs remain resident")
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--history-fraction",type=float,default=0.0,
                   help="Fraction of learner policy positions generated against frozen historical actors")
    p.add_argument("--history-iterations",type=int,nargs="*",default=[])
    p.add_argument("--history-weights",type=float,nargs="*",default=[])
    p.add_argument("--action-ranking-positions",type=int,default=0,
                   help="Ephemeral hard-action ranking roots per self-play cohort; zero disables")
    p.add_argument("--action-ranking-topk",type=int,default=4)
    p.add_argument("--action-ranking-rollouts",type=int,default=4)
    p.add_argument("--action-ranking-round3-rollouts",type=int,default=8)
    p.add_argument("--action-ranking-round45-rollouts",type=int,default=16,
                   help="Higher continuation count for rounds 4 and 5")
    p.add_argument("--action-ranking-max-rollouts",type=int,default=32)
    p.add_argument("--action-ranking-simulations",type=int,default=512)
    p.add_argument("--action-ranking-round3-simulations",type=int,default=1024)
    p.add_argument("--action-ranking-round45-simulations",type=int,default=4096)
    p.add_argument("--action-ranking-policy-weight",type=float,default=0.05)
    p.add_argument("--action-ranking-value-weight",type=float,default=0.10)
    p.add_argument("--action-ranking-value-order-weight",type=float,default=0.05)
    p.add_argument("--action-ranking-confidence",type=float,default=0.95)
    p.add_argument("--action-ranking-envs",type=int,default=512)
    p.add_argument("--action-ranking-capacity",type=int,default=512)
    p.add_argument("--action-ranking-batch-size",type=int,default=16)
    p.add_argument("--action-ranking-library",type=Path,
                   help="Optional dedicated continuation DLL, leaves frontend DLLs untouched")
    p.add_argument("--action-ranking-min-gap",type=float,default=0.10)
    p.add_argument("--action-ranking-max-steps",type=int,default=4096)
    return p


def merge_data(parts):
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))


@torch.inference_mode()
def validate_value(model,data,device,batch_size):
    model.eval();obs,targets,masks,outcomes=data;totals={};policy_totals={};calibration={}
    for start in range(0,len(obs),batch_size):
        x=torch.as_tensor(obs[start:start+batch_size],device=device).float()
        y=torch.as_tensor(outcomes[start:start+batch_size],device=device).long()
        policy_logits,logits=model(x);p=logits.float().softmax(-1)
        diagnostic=policy_metrics(policy_logits,torch.as_tensor(targets[start:start+batch_size],device=device),
                                  torch.as_tensor(masks[start:start+batch_size],device=device))
        utility=p[:,0]-p[:,2];target=torch.where(y==0,1.,torch.where(y==2,-1.,0.))
        brier=(p-torch.nn.functional.one_hot(y,3)).square().sum(-1)
        ce=torch.nn.functional.cross_entropy(logits.float(),y,reduction="none")
        confidence,prediction=p.max(-1);correct=(prediction==y).float()
        bins=(confidence*10).long().clamp_max(9)
        ending=model.end_features(x)[:,-1]>0.5 if model.end_features is not None else torch.zeros_like(y,dtype=torch.bool)
        decision=x[:,168]==0
        rounds=(x[:,170]*10).round()
        for name,mask in (("decision",decision),("ending",decision&ending),("afterstate",~decision),
                          ("round4",decision&(rounds==4)),("round5",decision&(rounds==5)),
                          ("early",decision&(rounds<=2)),("round3",decision&(rounds==3))):
            values=torch.stack((mask.sum(),(brier*mask).sum(),(ce*mask).sum(),((utility-target)*mask).sum()))
            totals[name]=totals.get(name,torch.zeros(4,device=device))+values
            reliability=torch.stack([torch.bincount(bins,weights=w*mask,minlength=10)
                for w in (torch.ones_like(confidence),confidence,correct)])
            calibration[name]=calibration.get(name,torch.zeros_like(reliability))+reliability
            if name!="afterstate":
                metrics=torch.stack([mask.sum(),*((v*mask).sum() for v in diagnostic.values())])
                policy_totals[name]=policy_totals.get(name,torch.zeros_like(metrics))+metrics
    result={}
    for name,values in totals.items():
        n,b,c,bias=values.cpu().tolist()
        if n:
            result[name]={"positions":int(n),"brier":b/n,"cross_entropy":c/n,"value_bias":bias/n}
            bins_n,bins_conf,bins_correct=calibration[name].cpu().numpy()
            result[name]["wdl_ece"]=float(np.abs(bins_conf-bins_correct).sum()/n)
            result[name]["calibration_bins"]=[dict(count=int(m),confidence=float(pc/m),accuracy=float(ac/m))
                for m,pc,ac in zip(bins_n,bins_conf,bins_correct) if m>0]
    for name,values in policy_totals.items():
        n,*metrics=values.cpu().tolist()
        if n:result[name].update({k:v/n for k,v in zip(diagnostic,metrics)})
    return result


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
        historical_architecture=upgraded_architecture(item.get("architecture",{}),architecture)
        if historical_architecture!=architecture:raise ValueError(f"Historical actor architecture mismatch: {path}")
        with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
        pool.append(dict(iteration=iteration,path=path,payload=item,sha256=digest))
    return pool,weights


def main(argv=None):
    argv=sys.argv[1:] if argv is None else argv
    args=parser().parse_args(argv);payload=None;changes={};phase_history=[]
    ranking_keys=[k for k in vars(args) if k.startswith("action_ranking_")]
    if args.init_from and args.resume:raise ValueError("Use either --init-from or --resume")
    if args.retune and not args.resume:raise ValueError("--retune requires --resume")
    if args.resume:
        payload=torch.load(args.resume,map_location="cpu",weights_only=False)
        if payload["engine_version"]!=OPTIMIZED_VERSION:raise ValueError("Use legacy train.py to resume v1 MLP runs; v2 is a new architecture")
        explicit={v.split("=")[0][2:].replace("-","_") for v in argv if v.startswith("--")}
        for key in ("init_from","value_tower","value_tower_blocks","value_rank_weight","value_label_smoothing"):
            if key in explicit and key not in payload["args"]:
                raise ValueError(f"New value architecture/objective requires a new run initialized with --init-from: {key}")
        for k,v in payload["args"].items():
            if k in ("output","resume","iterations","device","retune"):continue
            if k in ("full_replay_every","replay_compression","keep_recent","keep_anchors") and k in explicit:continue
            if args.retune and k in explicit and k in ("learning_rate","replay_capacity","recent_fraction","sample_reuse","simulations",
                                                        "high_simulations","high_budget_period",
                                                        "chance_cap",
                                                        "policy_loss_weight","value_loss_weight",
                                                        "history_fraction","history_iterations","history_weights","end_features",
                                                        "envs","positions_per_update","native_library","release_cohort_arenas",*ranking_keys):
                if getattr(args,k)!=v:
                    new=getattr(args,k)
                    changes[k]={"old":v,"new":str(new) if isinstance(new,Path) else new}
                continue
            if k in explicit and getattr(args,k)!=v:raise ValueError(f"Cannot change protocol field on exact resume: {k}")
            setattr(args,k,v)
        for k,old in (("history_fraction",0.0),("history_iterations",[]),("history_weights",[]),("end_features",False),
                      ("policy_loss_weight",1.0),("value_loss_weight",1.0)):
            if k not in payload["args"] and k in explicit:
                if not args.retune:raise ValueError(f"Enabling --{k.replace('_','-')} requires --retune")
                if getattr(args,k)!=old:changes[k]={"old":old,"new":getattr(args,k)}
        if "action_ranking_positions" not in payload["args"] and args.action_ranking_positions:
            if not args.retune:raise ValueError("Enabling action ranking on old checkpoints requires --retune")
            changes["action_ranking_positions"]={"old":0,"new":args.action_ranking_positions}
        if "release_cohort_arenas" not in payload["args"] and args.release_cohort_arenas:
            if not args.retune:raise ValueError("Enabling arena reclamation requires --retune")
            changes["release_cohort_arenas"]={"old":False,"new":True}
        if "output" not in explicit:args.output=args.resume.parent
        if payload["architecture"].get("end_features",False) and not args.end_features:
            raise ValueError("Cannot remove trained ending features on resume")
    elif (args.output/"latest.pt").exists() or (args.output/"initial.pt").exists():
        raise FileExistsError("Run exists; use --resume or a new output directory")
    if min(args.envs,args.queues,args.threads,args.iterations,args.positions_per_update,args.max_steps,args.batch_size,
           args.replay_capacity,args.width,args.blocks,args.full_replay_every,args.keep_recent,args.keep_anchors)<1:raise ValueError("Counts must be positive")
    if not 0<=args.recent_fraction<=1 or args.sample_reuse<=0 or args.afterstate_loss<0 or args.warmup<0:
        raise ValueError("Invalid learner parameters")
    if args.value_tower_blocks<0 or args.value_rank_weight<0 or not 0<=args.value_label_smoothing<1:
        raise ValueError("Invalid value calibration parameters")
    if not all(math.isfinite(w) and w>0 for w in (args.policy_loss_weight,args.value_loss_weight)):
        raise ValueError("Policy/value loss weights must be positive")
    if args.validation_games<0 or args.validation_every<1:raise ValueError("Invalid validation counts")
    if args.high_simulations<0 or args.high_budget_period<0:
        raise ValueError("Invalid mixed-budget parameters")
    if args.high_budget_period and args.high_simulations and args.high_simulations<args.simulations:
        raise ValueError("high-simulations must be >= simulations")
    ranking_config=RankingConfig(positions=args.action_ranking_positions,topk=args.action_ranking_topk,
        rollouts=args.action_ranking_rollouts,round3_rollouts=args.action_ranking_round3_rollouts,
        round45_rollouts=args.action_ranking_round45_rollouts,max_rollouts=args.action_ranking_max_rollouts,
        simulations=args.action_ranking_simulations,round3_simulations=args.action_ranking_round3_simulations,
        round45_simulations=args.action_ranking_round45_simulations,envs=args.action_ranking_envs,
        confidence=args.action_ranking_confidence,min_utility_gap=args.action_ranking_min_gap,
        max_steps=args.action_ranking_max_steps)
    ranking_config.validate()
    if min(args.action_ranking_capacity,args.action_ranking_batch_size)<1 or not all(
        math.isfinite(w) and w>=0 for w in (args.action_ranking_policy_weight,
            args.action_ranking_value_weight,args.action_ranking_value_order_weight)):
        raise ValueError("Invalid ranking learner parameters")
    if not 0<=args.history_fraction<1:raise ValueError("History fraction must be in [0, 1)")
    if bool(args.history_fraction)!=bool(args.history_iterations):
        raise ValueError("Positive history fraction requires historical actors, and vice versa")
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0 or not math.isfinite(args.sample_reuse):
        raise ValueError("Learning rate and reuse must be finite and positive")
    if payload and args.replay_capacity<payload["args"]["replay_capacity"]:
        raise ValueError("Retune cannot discard replay through capacity reduction")
    native_revision=require_current_search(_load(args.native_library))
    device=device_setup(args.device,2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    model=PolicyValue(args.width,args.blocks,kind="equivariant",end_features=args.end_features,
                      value_tower=args.value_tower,value_tower_blocks=args.value_tower_blocks).to(device)
    pool,history_weights=load_history_pool(args.output.resolve(),args.history_iterations,args.history_weights,model.architecture)
    pool_manifest=[dict(iteration=x["iteration"],sha256=x["sha256"],weight=w)
                   for x,w in zip(pool,history_weights)]
    history_changed=any(k in changes for k in ("history_fraction","history_iterations","history_weights"))
    if payload and not history_changed and payload.get("history_pool") is not None and payload["history_pool"]!=pool_manifest:
        raise ValueError("Historical actor pool changed since this checkpoint was written")
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4,
        fused=device.type=="cuda",capturable=device.type=="cuda")
    replay=None
    search_config=SearchConfig(simulations=args.simulations,candidates=min(args.candidates,args.simulations),max_depth=args.max_depth,
        chance_initial=args.chance_initial,chance_cap=args.chance_cap)
    tuning=Tuning(q_mode={"original":0,"floor":1,"fixed":2}[args.q_mode],q_floor=args.q_floor,
        chance_coefficient=args.chance_coefficient,chance_exponent=args.chance_exponent,
        chance_sensitivity=args.chance_sensitivity,afterstate_prior=args.afterstate_prior,
        subtree_reuse=int(not args.no_subtree_reuse),cache_capacity=args.cache_capacity)
    # Equal budgets use the same search arenas; keeping a second set doubles
    # retained tree memory without adding any search diversity.
    high_search_config=None if not (args.high_simulations>args.simulations and args.high_budget_period) else SearchConfig(
        simulations=args.high_simulations,candidates=min(args.candidates,args.high_simulations),max_depth=args.max_depth,
        chance_initial=args.chance_initial,chance_cap=args.chance_cap)
    start_iteration=updates=total_games=0;credit=0.0;mean_length=80.0
    if payload:
        model.load_compatible_state_dict(payload["model"])
        load_compatible_optimizer(optimizer,model,payload)
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
    else:
        if args.init_from:
            source=torch.load(args.init_from,map_location="cpu",weights_only=False)
            if source.get("engine_version")!=OPTIMIZED_VERSION:
                raise ValueError("--init-from requires an optimized v2 actor")
            arch=upgraded_architecture(source["architecture"],model.architecture)
            if arch!=model.architecture:raise ValueError("--init-from architecture mismatch")
            model.load_compatible_state_dict(source["model"])
            start_iteration=int(source["iteration"])
            if args.iterations<=start_iteration:raise ValueError("iterations must exceed source actor generation")
            phase_history.append(dict(after_iteration=start_iteration,source_checkpoint=str(args.init_from.resolve()),
                note="Actor-weight restart; fresh optimizer, replay and RNG; no later-generation training state restored",
                utc=datetime.now(timezone.utc).isoformat(),architecture=model.architecture))
        replay=DeviceReplay(args.replay_capacity,device,args.recent_fraction)
    actor=FrozenActor(model,device,not args.fp32)
    args.output.mkdir(parents=True,exist_ok=True);log=args.output/"metrics.jsonl"
    journal=CheckpointJournal(args.output,args.full_replay_every,args.replay_compression,args.keep_recent,args.keep_anchors)
    if payload and args.output.resolve()==args.resume.parent.resolve():
        journal.base=payload["replay_manifest"]["base"];journal.deltas=list(payload["replay_manifest"]["deltas"])
    if "replay_capacity" in changes:journal.force_full=True
    if changes:emit(dict(event="retune",**phase_history[-1],remaining_sample_credit=credit),log)
    emit(dict(event="start",engine=OPTIMIZED_VERSION,search_revision=native_revision,
        factory_canonical_chance=True,device=str(device),parameters=sum(p.numel() for p in model.parameters()),
        architecture=model.architecture,search=asdict(search_config),tuning=asdict(tuning),resumed_iteration=start_iteration,
        replay_device=str(replay.device),graphs=not args.no_graphs,queues=args.queues,
        learning_rate=optimizer.param_groups[0]["lr"],replay_capacity=replay.capacity,
        policy_loss_weight=args.policy_loss_weight,value_loss_weight=args.value_loss_weight,
        recent_fraction=replay.recent_fraction,sample_reuse=args.sample_reuse,
        history_fraction=args.history_fraction,
        history_pool=pool_manifest,action_ranking=asdict(ranking_config),
        ranking_loss_weights=dict(policy=args.action_ranking_policy_weight,
            soft_value=args.action_ranking_value_weight,value_order=args.action_ranking_value_order_weight)),log)
    learner=None;ranking_replay=None
    if ranking_config.positions:
        ranking_replay=RankingReplay(args.action_ranking_capacity,args.action_ranking_topk,device)
        if payload and "ranking_replay" in payload:
            ranking_replay.restore(payload["ranking_replay"])
    validation_path=args.output/"value-validation.npz";validation=None
    if validation_path.exists():
        with np.load(validation_path,allow_pickle=False) as stored:validation=tuple(stored[f"field{i}"] for i in range(4))
    if payload and args.validation_games and validation is None:raise FileNotFoundError(validation_path)
    with ExitStack() as stack:
        env=stack.enter_context(Batch(args.envs,args.threads,args.seed,library=args.native_library))
        search=stack.enter_context(PipelineSearch(env,actor,search_config,tuning,args.threads,args.queues,not args.no_graphs))
        ranking_runner=None
        if ranking_config.positions:
            ranking_runner=stack.enter_context(ContinuationRunner(actor,search_config,tuning,ranking_config,
                args.threads,args.queues,not args.no_graphs,args.action_ranking_library))
        high_search=None
        if high_search_config is not None:
            high_search=stack.enter_context(PipelineSearch(env,actor,high_search_config,tuning,args.threads,args.queues,not args.no_graphs))
        history_model=history_actor=history_search=history_high_search=None
        if pool:
            history_architecture=dict(model.architecture)
            history_model=PolicyValue(**history_architecture)
            history_model.load_compatible_state_dict(pool[0]["payload"]["model"])
            history_actor=FrozenActor(history_model,device,not args.fp32)
            history_search=stack.enter_context(PipelineSearch(env,history_actor,search_config,tuning,
                                                               args.threads,args.queues,not args.no_graphs))
            if high_search_config is not None:
                history_high_search=stack.enter_context(PipelineSearch(env,history_actor,high_search_config,tuning,
                                                                        args.threads,args.queues,not args.no_graphs))
        if payload:
            torch.set_rng_state(payload["torch_rng"])
            if device.type=="cuda" and payload["cuda_rng"] is not None:torch.cuda.set_rng_state_all(payload["cuda_rng"])
        else:
            save_checkpoint(args.output/"initial.pt",dict(engine_version=OPTIMIZED_VERSION,architecture=model.architecture,
                model=model.state_dict(),iteration=start_iteration,search=asdict(search_config),tuning=asdict(tuning)))
        for iteration in range(start_iteration,args.iterations):
            actor.publish(model,iteration+1)
            use_high = high_search_config is not None and ((iteration+1) % args.high_budget_period == 0)
            active_search = high_search if use_high else search
            # The afterstate head is first taught by completed self-play before it contributes a prior.
            active_search.set_afterstate_prior(args.afterstate_prior if updates>0 else 0.0)
            target=args.positions_per_update
            self_games=min(args.envs,max(1,round(target*(1-args.history_fraction)/mean_length)))
            history_games=0 if not pool else max(1,round(2*target*args.history_fraction/mean_length))
            if history_games>args.envs:raise ValueError("Historical cohort exceeds --envs")
            parts=[];components=[]
            holdout=args.validation_games if validation is None else 0
            data,component=collect_cohort(env,active_search,split_seed(args.seed,iteration),self_games,args.max_steps,
                                          lambda row:emit(row,log),holdout_games=holdout,
                                          ranking_config=ranking_config,
                                          ranking_seed=split_seed(args.seed^0x51ED2705,iteration))
            anchors=component.pop("ranking_anchors",[])
            if args.release_cohort_arenas or ranking_runner:
                active_search.release_arenas()
            if ranking_runner:
                try:
                    ranking_data,ranking_metrics=ranking_runner.evaluate(anchors,
                        split_seed(args.seed^0xBBE0563303A4615F,iteration),lambda row:emit(row,log))
                except Exception:
                    if anchors:
                        np.savez_compressed(args.output/"failed-ranking-anchors.npz",
                            snapshots=np.stack([np.frombuffer(a["snapshot"],np.uint8) for a in anchors]),
                            **{name:np.asarray([a[name] for a in anchors]) for name in
                               ("obs","mask","policy","selected","root_player","game","category","round")},
                            seed=np.asarray(split_seed(args.seed^0xBBE0563303A4615F,iteration),np.uint64),
                            teacher_version=np.asarray(actor.version,np.uint64))
                    raise
                ranking_replay.add(ranking_data)
                ranking_runner.release_arenas()
                emit(dict(ranking_metrics,iteration=iteration+1),log)
            if holdout:
                validation=component.pop("holdout_data")
                np.savez_compressed(validation_path,**{f"field{i}":v for i,v in enumerate(validation)})
                emit(dict(event="value_validation",iteration=iteration+1,stage="before_learning",
                          metrics=validate_value(model,validation,device,args.batch_size)),log)
            component.update(component="current_selfplay",iteration=iteration+1)
            parts.append(data);components.append(component);emit(component,log)
            if pool:
                pool_index=smooth_weighted_index(iteration,history_weights)
                entry=pool[pool_index];count=history_games
                history_model.load_compatible_state_dict(entry["payload"]["model"])
                history_actor.publish(history_model,(iteration+1)*len(pool)+pool_index+1)
                active_history_search = history_high_search if use_high else history_search
                data,component=collect_historical_cohort(env,active_search,active_history_search,
                    split_seed(args.seed^0xD1B54A32D192ED03,(iteration+1)*len(pool)+pool_index),count,
                    entry["iteration"],args.max_steps,lambda row:emit(row,log))
                component["iteration"]=iteration+1
                if args.release_cohort_arenas or ranking_runner:
                    active_search.release_arenas();active_history_search.release_arenas()
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
                selfplay_simulations=(args.high_simulations if use_high else args.simulations),
                high_budget=bool(use_high),
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
                learner=GraphLearner(model,optimizer,replay,args.batch_size,args.afterstate_loss,
                                     not args.no_graphs,not args.fp32,
                                     value_rank_weight=args.value_rank_weight,
                                     value_label_smoothing=args.value_label_smoothing,
                                     policy_loss_weight=args.policy_loss_weight,
                                     value_loss_weight=args.value_loss_weight,
                                     ranking_replay=ranking_replay,ranking_batch_size=args.action_ranking_batch_size,
                                     ranking_policy_weight=args.action_ranking_policy_weight,
                                     ranking_value_weight=args.action_ranking_value_weight,
                                     ranking_value_order_weight=args.action_ranking_value_order_weight)
            t=time.perf_counter();losses=learner.run(n_updates) if n_updates else {}
            updates+=n_updates;credit-=n_updates*args.batch_size
            emit(dict(event="train",iteration=iteration+1,updates=n_updates,total_updates=updates,
                seconds=time.perf_counter()-t,ingest_seconds=ingest,
                gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1048576 if device.type=="cuda" else 0,
                **losses,**replay.statistics()),log)
            if validation is not None and ((iteration+1)%args.validation_every==0 or iteration+1==args.iterations):
                emit(dict(event="value_validation",iteration=iteration+1,stage="after_learning",
                          metrics=validate_value(model,validation,device,args.batch_size)),log)
            state=dict(engine_version=OPTIMIZED_VERSION,architecture=model.architecture,search=asdict(search_config),tuning=asdict(tuning),
                args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},model=model.state_dict(),optimizer=optimizer.state_dict(),
                iteration=iteration+1,updates=updates,games=total_games,credit=credit,mean_length=mean_length,
                phase_history=phase_history,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else None)
            state["history_pool"]=pool_manifest
            if ranking_replay is not None:state["ranking_replay"]=ranking_replay.snapshot()
            state["history_schedule"]="smooth_weighted_single" if pool else None
            t=time.perf_counter();journal.save(state,replay,data,iteration+1)
            emit(dict(event="checkpoint",iteration=iteration+1,path=str(args.output/"latest.pt"),
                      seconds=time.perf_counter()-t,replay_base=journal.base,replay_deltas=len(journal.deltas)),log)


if __name__=="__main__":main()
