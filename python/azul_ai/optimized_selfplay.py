import ctypes as C
import time
import numpy as np
from .search import pointer
from .selfplay import results_view,split_seed


def collect_cohort(batch,search,seed,games,max_steps=4096,progress=None):
    batch.reset(seed);n=batch.n;views=batch.numpy_views();result=results_view(batch)
    batch._lib.azul_batch_step_afterstates.argtypes=[C.c_void_p,C.POINTER(C.c_uint16),C.c_void_p,
        C.POINTER(C.c_float),C.POINTER(C.c_uint8),C.POINTER(C.c_uint8)]
    batch._lib.azul_batch_step_afterstates.restype=C.c_int
    active=np.zeros(n,bool);active[:games]=True
    winners=np.full(n,-2,np.int8)
    after_obs=np.zeros((n,172),np.float32);flags=np.zeros(n,np.uint8);perspectives=np.zeros(n,np.uint8)
    observations=[];policies=[];masks=[];players=[];lanes=[]
    metrics={};decisions=after_count=0;start=last=time.perf_counter();packing_time=0;tail_start=None
    for tick in range(max_steps):
        batch.observe();ids=np.flatnonzero(active)
        observations.append(views["observations"][ids].astype(np.float16))
        masks.append(views["masks"][ids].copy());players.append(views["players"][ids].copy());lanes.append(ids)
        actions,pi,_=search.run(split_seed(seed^0xA0761D6478BD642F,tick),active)
        for k,v in search.metrics.items():metrics[k]=metrics.get(k,0)+v
        t=time.perf_counter();policies.append(pi[ids].astype(np.float16));views["actions"][:]=actions
        code=batch._lib.azul_batch_step_afterstates(batch._handle,pointer(views["actions"],C.c_uint16),
            batch.results,pointer(after_obs),pointer(flags,C.c_uint8),pointer(perspectives,C.c_uint8))
        if code or result["invalid_action"][active].any():raise RuntimeError("Invalid optimized environment step")
        after_ids=np.flatnonzero(active & flags.astype(bool))
        if len(after_ids):
            observations.append(after_obs[after_ids].astype(np.float16));policies.append(np.zeros((len(after_ids),180),np.float16))
            masks.append(np.zeros((len(after_ids),180),np.uint8));players.append(perspectives[after_ids].copy());lanes.append(after_ids)
            after_count+=len(after_ids)
        done=active & result["terminated"].astype(bool)
        r=result["reward"];winner=np.where(r>0,result["actor"],1-result["actor"]).astype(np.int8)
        winners[done]=np.where(r[done]==0,-1,winner[done])
        decisions+=len(ids);active[done]=False;packing_time+=time.perf_counter()-t
        if tail_start is None and active.sum()<=games/10:tail_start=time.perf_counter()
        if progress and time.perf_counter()-last>=30:
            progress(dict(event="selfplay_progress",tick=tick+1,completed=int((winners!=-2).sum()),games=games,
                          decisions=decisions,decisions_per_s=decisions/(time.perf_counter()-start)))
            last=time.perf_counter()
        if not active.any():break
    pack_start=time.perf_counter()
    obs=np.concatenate(observations);pi=np.concatenate(policies);mask=np.concatenate(masks)
    actor=np.concatenate(players);lane=np.concatenate(lanes);valid=winners[lane]!=-2
    outcome=np.where(winners[lane]==-1,1,np.where(winners[lane]==actor,0,2)).astype(np.uint8)
    data=(obs[valid],pi[valid],mask[valid],outcome[valid])
    seconds=time.perf_counter()-start
    metrics.update(event="selfplay",games=int((winners!=-2).sum()),truncated=int(active.sum()),decisions=decisions,
        positions=int(valid.sum()),decision_positions=int((valid & (obs[:,168]==0)).sum()),afterstate_positions=after_count,
        seconds=seconds,decisions_per_s=decisions/seconds,valid_leaf_fraction=metrics.get("valid_leaves",0)/max(1,metrics.get("valid_leaves",0)+metrics.get("padded_leaves",0)),
        tail_seconds=0 if tail_start is None else time.perf_counter()-tail_start,
        sample_pack_seconds=packing_time+time.perf_counter()-pack_start,
        wins0=int((winners==0).sum()),wins1=int((winners==1).sum()),draws=int((winners==-1).sum()))
    return data,metrics


def collect_historical_cohort(batch, learner_search, opponent_search, seed, games,
                              opponent_iteration, max_steps=4096, progress=None):
    """Learner versus one frozen historical actor.

    Only learner decision policies are retained. Chance afterstates from either
    side remain valid W/D/L supervision because observations carry perspective.
    Learner seat and starting player span all four combinations evenly.
    """
    n=batch.n
    if games<1 or games>n:raise ValueError("historical cohort must fit the native batch")
    batch.reset(seed);views=batch.numpy_views();result=results_view(batch)
    batch._lib.azul_batch_step_afterstates.argtypes=[C.c_void_p,C.POINTER(C.c_uint16),C.c_void_p,
        C.POINTER(C.c_float),C.POINTER(C.c_uint8),C.POINTER(C.c_uint8)]
    batch._lib.azul_batch_step_afterstates.restype=C.c_int
    active=np.zeros(n,bool);ids=np.arange(games);active[ids]=True
    learner_seat=np.zeros(n,np.uint8)
    for game,lane in enumerate(ids):
        learner_seat[lane]=game&1
        batch.reset_at(int(lane),split_seed(seed,game),int((game//2)&1))
    winners=np.full(n,-2,np.int8)
    after_obs=np.zeros((n,172),np.float32);flags=np.zeros(n,np.uint8);perspectives=np.zeros(n,np.uint8)
    observations=[];policies=[];masks=[];players=[];lanes=[]
    totals={};decisions=learner_decisions=opponent_decisions=after_count=0
    start=last=time.perf_counter();packing_time=0;tail_start=None
    for tick in range(max_steps):
        batch.observe();current=views["players"].copy()
        learner_turn=active & (current==learner_seat)
        opponent_turn=active & ~learner_turn
        actions=np.full(n,65535,np.uint16)
        if learner_turn.any():
            selected,pi,_=learner_search.run(split_seed(seed^0xA0761D6478BD642F,tick),learner_turn)
            actions[learner_turn]=selected[learner_turn]
            keep=np.flatnonzero(learner_turn)
            observations.append(views["observations"][keep].astype(np.float16))
            policies.append(pi[keep].astype(np.float16));masks.append(views["masks"][keep].copy())
            players.append(current[keep]);lanes.append(keep);learner_decisions+=len(keep)
            for k,v in learner_search.metrics.items():totals[f"learner_{k}"]=totals.get(f"learner_{k}",0)+v
        if opponent_turn.any():
            selected,_,_=opponent_search.run(split_seed(seed^0xE7037ED1A0B428DB,tick),opponent_turn)
            actions[opponent_turn]=selected[opponent_turn];opponent_decisions+=int(opponent_turn.sum())
            for k,v in opponent_search.metrics.items():totals[f"opponent_{k}"]=totals.get(f"opponent_{k}",0)+v
        t=time.perf_counter();views["actions"][:]=actions
        code=batch._lib.azul_batch_step_afterstates(batch._handle,pointer(views["actions"],C.c_uint16),
            batch.results,pointer(after_obs),pointer(flags,C.c_uint8),pointer(perspectives,C.c_uint8))
        if code or result["invalid_action"][active].any():raise RuntimeError("Invalid historical environment step")
        after_ids=np.flatnonzero(active & flags.astype(bool))
        if len(after_ids):
            observations.append(after_obs[after_ids].astype(np.float16));policies.append(np.zeros((len(after_ids),180),np.float16))
            masks.append(np.zeros((len(after_ids),180),np.uint8));players.append(perspectives[after_ids].copy());lanes.append(after_ids)
            after_count+=len(after_ids)
        done=active & result["terminated"].astype(bool)
        reward=result["reward"];absolute=np.where(reward>0,result["actor"],1-result["actor"]).astype(np.int8)
        winners[done]=np.where(reward[done]==0,-1,absolute[done])
        decisions+=int(active.sum());active[done]=False;packing_time+=time.perf_counter()-t
        if tail_start is None and active.sum()<=games/10:tail_start=time.perf_counter()
        if progress and time.perf_counter()-last>=30:
            progress(dict(event="history_progress",opponent_iteration=opponent_iteration,tick=tick+1,
                completed=int((winners!=-2).sum()),games=games,decisions=decisions,
                decisions_per_s=decisions/(time.perf_counter()-start)))
            last=time.perf_counter()
        if not active.any():break
    pack_start=time.perf_counter()
    obs=np.concatenate(observations);pi=np.concatenate(policies);mask=np.concatenate(masks)
    actor=np.concatenate(players);lane=np.concatenate(lanes);valid=winners[lane]!=-2
    outcome=np.where(winners[lane]==-1,1,np.where(winners[lane]==actor,0,2)).astype(np.uint8)
    learner_result=np.array([0.5 if winners[i]==-1 else float(winners[i]==learner_seat[i])
                             for i in ids if winners[i]!=-2])
    seconds=time.perf_counter()-start
    totals.update(event="historical_selfplay",opponent_iteration=int(opponent_iteration),
        games=int((winners[ids]!=-2).sum()),truncated=int((winners[ids]==-2).sum()),decisions=decisions,
        learner_decisions=learner_decisions,opponent_decisions=opponent_decisions,
        positions=int(valid.sum()),decision_positions=learner_decisions,afterstate_positions=after_count,
        learner_score_rate=float(learner_result.mean()) if len(learner_result) else None,
        seconds=seconds,decisions_per_s=decisions/seconds,
        tail_seconds=0 if tail_start is None else time.perf_counter()-tail_start,
        sample_pack_seconds=packing_time+time.perf_counter()-pack_start)
    return (obs[valid],pi[valid],mask[valid],outcome[valid]),totals
