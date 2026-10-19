"""Frozen self-play continuation labels. Inference remains the same Gumbel model."""
from contextlib import ExitStack
import ctypes as C
from dataclasses import dataclass, asdict, replace
import heapq
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from azul import Batch, StepResult
from .optimized_search import PipelineSearch
from .search import pointer
from .selfplay import results_view, split_seed


@dataclass(frozen=True)
class RankingConfig:
    positions: int = 0
    topk: int = 4
    rollouts: int = 4
    round3_rollouts: int = 8
    round45_rollouts: int = 16
    max_rollouts: int = 32
    simulations: int = 512
    round3_simulations: int = 1024
    round45_simulations: int = 4096
    min_utility_gap: float = 0.10  # W + .5D, in [0,1]
    confidence: float = 0.95
    max_steps: int = 4096
    envs: int = 512

    def validate(self):
        counts=(self.rollouts,self.round3_rollouts,self.round45_rollouts,self.max_rollouts,
                self.simulations,self.round3_simulations,self.round45_simulations,self.max_steps,self.envs)
        if self.positions<0 or not 2<=self.topk<=16 or min(counts)<1:
            raise ValueError("Invalid continuation counts")
        if min(self.simulations,self.round3_simulations,self.round45_simulations)<2:
            raise ValueError("Continuation search needs at least two simulations")
        if self.max_rollouts<max(self.rollouts,self.round3_rollouts,self.round45_rollouts):
            raise ValueError("max_rollouts must cover all initial rollout counts")
        if self.max_rollouts>65535:
            raise ValueError("Continuation counts exceed uint16 storage")
        if not 0<=self.min_utility_gap<1 or not 0<self.confidence<1:
            raise ValueError("Invalid ranking confidence/gap")
        if self.round45_simulations<max(self.simulations,self.round3_simulations):
            raise ValueError("Rounds 4/5 must use the highest search budget")
        if self.round45_rollouts<max(self.rollouts,self.round3_rollouts):
            raise ValueError("Rounds 4/5 must use the most initial continuations")

    def repeats(self, round_number):
        return self.round45_rollouts if round_number>=4 else (
            self.round3_rollouts if round_number==3 else self.rollouts)

    def budget(self, round_number):
        return self.round45_simulations if round_number>=4 else (
            self.round3_simulations if round_number==3 else self.simulations)


def bind_branch_api(lib):
    fn=getattr(lib,"azul_batch_branch",None)
    if fn is None:
        raise RuntimeError("Value supervision needs a rebuilt DLL with azul_batch_branch")
    fn.argtypes=[C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,
                 C.POINTER(C.c_uint64),C.POINTER(C.c_uint16),C.POINTER(C.c_float),
                 C.POINTER(C.c_uint8),C.POINTER(StepResult)]
    fn.restype=C.c_int
    return fn


def state_category(obs):
    """No fixed-round ending assumption. Completed patterns project onto the wall."""
    players=np.asarray(obs[30:154],np.float32).reshape(2,62)
    wall=players[:,:25].reshape(2,5,5)>0.5
    pattern=players[:,25:50].reshape(2,5,5)>0.5
    full=players[:,50:55]>=0.999
    projected=wall.copy()
    for player in range(2):
        for row in range(5):
            if full[player,row] and pattern[player,row].any():
                projected[player,row,(row+int(pattern[player,row].argmax()))%5]=True
    ending=bool(projected.all(-1).any())
    missing=(~wall).sum(-1)
    risk=bool((missing==1).any())
    round_number=int(np.rint(float(obs[170])*10))
    if round_number>=4:
        return "late"
    if ending or risk:
        return "ending"
    return "middle" if round_number==3 else "early"


class AnchorSampler:
    """Bounded per-stratum heap; snapshots only for roots entering the heap."""
    def __init__(self, config, seed):
        self.config=config;self.rng=np.random.default_rng(seed)
        self.heaps={k:[] for k in ("late","ending","middle","early")}
        self.counter=0

    def consider(self,batch,ids,obs,masks,policies,selected,exclude_before=0,values=None):
        if not self.config.positions:
            return
        for local,lane in enumerate(ids):
            if lane<exclude_before or int(masks[local].sum())<2:
                continue
            p=policies[lane].astype(np.float64)*masks[local]
            p/=max(float(p.sum()),1e-12)
            top=np.sort(p[masks[local].astype(bool)])[-2:]
            gap=float(top[-1]-top[-2])
            entropy=float(-(p*np.log(np.maximum(p,1e-12))).sum())
            # Small random component breaks ties and reserves broad coverage.
            uncertainty=1-abs(float(values[lane])) if values is not None else 1.
            priority=((1-gap)+entropy/max(1,math.log(int(masks[local].sum()))))*(.2+.8*max(0,uncertainty))
            priority+=.2*self.rng.random()
            category=state_category(obs[local]);heap=self.heaps[category]
            self.counter+=1
            if len(heap)>=self.config.positions and priority<=heap[0][0]:
                continue
            record=dict(snapshot=batch.snapshot(int(lane)),obs=obs[local].copy(),
                        mask=masks[local].copy(),policy=p.astype(np.float32),
                        selected=int(selected[lane]),root_player=int(batch.players[lane]),
                        game=int(lane),category=category,
                        round=int(np.rint(float(obs[local,170])*10)))
            item=(priority,self.counter,record)
            if len(heap)<self.config.positions:heapq.heappush(heap,item)
            else:heapq.heapreplace(heap,item)

    def finish(self,completed):
        # Late half, middle quarter, ending/early remainder. Unfilled quotas
        # are reassigned. All anchors originate in completed non-held-out games.
        available={k:sorted([x for x in heap if completed[x[2]["game"]]],reverse=True)
                   for k,heap in self.heaps.items()}
        n=self.config.positions
        quotas=dict(late=(n+1)//2,middle=n//4,ending=n//8,early=n//8)
        selected=[];unused=[]
        for k,items in available.items():
            selected.extend(items[:quotas[k]]);unused.extend(items[quotas[k]:])
        selected.extend(sorted(unused,reverse=True)[:max(0,n-len(selected))])
        # Avoid one game's many nearby states dominating an augmentation batch.
        limited=[];extra=[];seen={}
        for item in sorted(selected,reverse=True):
            game=item[2]["game"];seen[game]=seen.get(game,0)+1
            (limited if seen[game]<=2 else extra).append(item)
        limited.extend(extra[:max(0,n-len(limited))])
        return [x[2] for x in limited[:n]]


def choose_candidates(policy,mask,topk,selected=None,obs=None,required=()):
    ids=np.flatnonzero(np.asarray(mask,dtype=bool))
    ordered=ids[np.argsort(-np.asarray(policy)[ids],kind="stable")].tolist()
    for must in reversed(tuple(required)+(() if selected is None else (selected,))):
        if must in ordered:
            ordered.remove(must);ordered.insert(0,int(must))
    chosen=[];keys=set()
    sources=None if obs is None else np.rint(np.asarray(obs[:30])*20).astype(np.int16).reshape(6,5)
    for action in ordered:
        source=action//30
        # Identical factory contents give equivalent moves for the same color/dst.
        key=(tuple(sources[source]),action%30) if sources is not None and source<5 else ("center",int(action))
        if key in keys:continue
        keys.add(key);chosen.append(int(action))
        if len(chosen)==topk:break
    return np.asarray(chosen,dtype=np.uint16)


def ranking_evidence(counts,exact=None,confidence=.95,looks=1,min_gap=.1):
    """Distribution-free simultaneous bounds, valid with adaptive stopping.

    Outcomes W=1, D=.5, L=0 are bounded. Hoeffding + Bonferroni over
    candidates and predetermined looks avoids zero empirical variance claims.
    Only exact terminal outcomes receive zero-width bounds.
    """
    counts=np.asarray(counts,dtype=np.float64);n=counts.sum(-1)
    mean=(counts[:,0]+.5*counts[:,1])/np.maximum(n,1)
    delta=(1-confidence)/max(1,len(n)*looks)
    radius=np.sqrt(math.log(2/delta)/(2*np.maximum(n,1)))
    exact=np.zeros(len(n),bool) if exact is None else np.asarray(exact,bool)
    radius=np.where(exact & (n>0),0,radius)
    lo=np.maximum(0,mean-radius);hi=np.minimum(1,mean+radius)
    pairs=np.zeros((len(n),len(n)),np.float32)
    for i in range(len(n)):
        for j in range(len(n)):
            if i!=j and n[i]>0 and n[j]>0 and lo[i]>hi[j]+min_gap:
                pairs[i,j]=1
    return mean,lo,hi,pairs


class ContinuationRunner:
    """Persistent arenas/bucketed GPU inference; independent replicates in lanes."""
    def __init__(self,actor,base_config,tuning,config,threads,queues,graphs,library=None):
        config.validate();self.config=config;self.actor=actor;self.stack=ExitStack()
        try:
            self.env=self.stack.enter_context(Batch(config.envs,threads,library=library))
            self.branch=bind_branch_api(self.env._lib)
            self.views=self.env.numpy_views();self.results=results_view(self.env)
            self.successors=np.empty((config.envs,172),np.float32)
            self.perspectives=np.empty(config.envs,np.uint8)
            self.searches={}
            for budget in sorted({config.simulations,config.round3_simulations,config.round45_simulations}):
                cfg=replace(base_config,simulations=budget,candidates=min(base_config.candidates,budget))
                self.searches[budget]=self.stack.enter_context(PipelineSearch(
                    self.env,actor,cfg,tuning,threads,queues,graphs))
            self.base_config=base_config;self.tuning=tuning
        except BaseException:
            self.stack.close();raise

    def close(self):self.stack.close()
    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def release_arenas(self):
        for search in self.searches.values():search.release_arenas()

    def _search(self,seed,active,version,progress):
        # Current continuation round decides budget, not just the initial root.
        self.env.observe();rounds=np.rint(self.views["observations"][:,170]*10).astype(int)
        actions=np.full(self.config.envs,65535,np.uint16)
        for budget,search in self.searches.items():
            wanted=np.asarray([self.config.budget(r) for r in rounds])==budget
            lanes=active & wanted
            if lanes.any():
                selected,_,_=search.run(seed,lanes,progress=progress,tree_version=version)
                actions[lanes]=selected[lanes]
        return actions

    def _wave(self,tasks,specs,seed,wave,progress):
        self.release_arenas()
        n=len(tasks)
        snapshots=np.stack([np.frombuffer(specs[r]["snapshot"],np.uint8) for r,a,rep in tasks])
        # Distinct streams for every candidate/replicate; no inherited real RNG.
        seeds=np.asarray([split_seed(seed^0xDB4F0B9175AE2165,(r*16+a)*self.config.max_rollouts+rep)
                          for r,a,rep in tasks],np.uint64)
        actions=np.asarray([specs[r]["actions"][a] for r,a,rep in tasks],np.uint16)
        status=self.branch(self.env._handle,n,snapshots.ctypes.data,snapshots.shape[1],
            pointer(seeds,C.c_uint64),pointer(actions,C.c_uint16),
            pointer(self.successors),pointer(self.perspectives,C.c_uint8),self.env.results)
        if status:raise RuntimeError("Invalid continuation branch")
        children=self.successors[:n].copy();perspectives=self.perspectives[:n].copy()
        done=self.results["terminated"][:n].astype(bool).copy()
        exact=done.copy();outcomes=np.full(n,-1,np.int8);margins=np.full(n,np.nan,np.float32)
        def finish(lanes):
            for lane in np.flatnonzero(lanes):
                r,a,rep=tasks[lane];root=specs[r]["root_player"]
                reward=float(self.results["reward"][lane]);actor=int(self.results["actor"][lane])
                outcomes[lane]=1 if reward==0 else (0 if (actor if reward>0 else 1-actor)==root else 2)
                s=self.results["scores"][lane];margins[lane]=int(s[root])-int(s[1-root])
        finish(done)
        active=np.zeros(self.config.envs,bool);active[:n]=~done
        # Every wave invalidates inherited evidence; same-wave trajectories
        # can still re-root their own independent search trees.
        version=split_seed(seed^0xA0761D6478BD642F,wave)
        last=time.perf_counter()
        for tick in range(self.config.max_steps):
            if not active.any():break
            actions=self._search(split_seed(seed^0xE7037ED1A0B428DB,wave*self.config.max_steps+tick),
                                 active,version,None)
            self.views["actions"][:]=actions;self.env.step()
            if self.results["invalid_action"][active].any():raise RuntimeError("Illegal continuation action")
            ended=active[:n] & self.results["terminated"][:n].astype(bool)
            finish(ended);active[:n]&=~ended
            if progress and time.perf_counter()-last>=30:
                progress(dict(event="continuation_progress",wave=wave,tick=tick+1,active=int(active.sum())))
                last=time.perf_counter()
        return children,perspectives,exact,outcomes,margins

    def evaluate(self,anchors,seed,progress=None):
        start=time.perf_counter()
        metrics=dict(event="action_ranking",roots=len(anchors),continuations=0,truncated=0,confident_pairs=0)
        if not anchors:
            return None,dict(metrics,seconds=0)
        k=self.config.topk;specs=[]
        for anchor in anchors:
            spec=dict(anchor);spec["actions"]=choose_candidates(
                spec["policy"],spec["mask"],k,spec["selected"],spec["obs"])
            if len(spec["actions"])>=2:specs.append(spec)
        if not specs:return None,dict(metrics,seconds=time.perf_counter()-start)
        # Candidate generation itself uses each round's configured search budget.
        # Include the actual training move even if high-budget policy ranks it low.
        for offset in range(0,len(specs),self.config.envs):
            group=specs[offset:offset+self.config.envs];active=np.zeros(self.config.envs,bool)
            for i,spec in enumerate(group):self.env.restore(i,spec["snapshot"]);active[i]=True
            self.env.observe()
            for budget,search in self.searches.items():
                lanes=active.copy()
                lanes[:len(group)]=[self.config.budget(s["round"])==budget for s in group]
                if not lanes.any():continue
                moves,pi,_=search.run(split_seed(seed,offset+budget),lanes,
                    tree_version=split_seed(seed,offset+budget+9999))
                for i in np.flatnonzero(lanes):
                    spec=group[i];candidates=choose_candidates(pi[i],spec["mask"],k,spec["selected"],
                        spec["obs"],required=(int(moves[i]),))
                    spec["actions"]=candidates
        count=len(specs)
        counts=np.zeros((count,k,3),np.uint16);children=np.zeros((count,k,172),np.float16)
        value_counts=np.zeros_like(counts)
        perspectives=np.zeros((count,k),np.uint8);exact=np.zeros((count,k),bool)
        attempted=np.zeros((count,k),np.uint16);score_sum=np.zeros((count,k),np.float32)
        targets=[self.config.repeats(s["round"]) for s in specs]
        next_rep=np.zeros((count,k),np.uint16);wave=0
        looks=1+math.ceil(math.log2(max(1,self.config.max_rollouts/min(targets))))
        while True:
            tasks=[]
            for r,spec in enumerate(specs):
                for a in range(len(spec["actions"])):
                    if exact[r,a]:continue
                    tasks.extend((r,a,rep) for rep in range(int(next_rep[r,a]),targets[r]))
                    next_rep[r,a]=targets[r]
            if not tasks:break
            for offset in range(0,len(tasks),self.config.envs):
                chunk=tasks[offset:offset+self.config.envs]
                successor,view,finished,outcomes,margins=self._wave(chunk,specs,seed,wave,progress);wave+=1
                for i,(r,a,rep) in enumerate(chunk):
                    children[r,a]=successor[i];perspectives[r,a]=view[i];exact[r,a]|=finished[i]
                    attempted[r,a]+=1;metrics["continuations"]+=1
                    if outcomes[i]>=0:
                        counts[r,a,outcomes[i]]+=1;score_sum[r,a]+=margins[i]
                        # A fixed initial block avoids optional-stopping bias
                        # in soft value targets; ranking uses all samples.
                        if rep<self.config.repeats(specs[r]["round"]):
                            value_counts[r,a,outcomes[i]]+=1
                    else:metrics["truncated"]+=1
            more=False
            for r,spec in enumerate(specs):
                mean,lo,hi,pairs=ranking_evidence(counts[r,:len(spec["actions"])],exact[r,:len(spec["actions"])],
                    self.config.confidence,looks,self.config.min_utility_gap)
                best=int(mean.argmax())
                # Stop only after the top candidate separates from every rival.
                resolved=all(j==best or pairs[best,j]>0 for j in range(len(mean)))
                observed=counts[r,:len(mean)]
                # Stop augmentation on uniformly saturated observed outcomes.
                # This is a compute decision, never a claim of equivalence.
                same_saturation=bool((observed[:,0]==observed.sum(-1)).all() or
                                     (observed[:,2]==observed.sum(-1)).all())
                if not resolved and not same_saturation and targets[r]<self.config.max_rollouts and not exact[r,:len(mean)].all():
                    targets[r]=min(self.config.max_rollouts,targets[r]*2);more=True
            if not more:break
        pairs=np.zeros((count,k,k),np.float32)
        valid=np.zeros((count,k),np.uint8);sign=np.zeros((count,k),np.float32)
        probabilities=np.zeros((count,k,3),np.float32);actions=np.zeros((count,k),np.int64)
        for r,spec in enumerate(specs):
            n=len(spec["actions"]);actions[r,:n]=spec["actions"]
            # Censored trajectories do not become draws or biased soft labels.
            ok=(counts[r,:n].sum(-1)==attempted[r,:n]) & (attempted[r,:n]>0)
            valid[r,:n]=ok
            _,_,_,evidence=ranking_evidence(counts[r,:n],exact[r,:n],self.config.confidence,looks,
                                           self.config.min_utility_gap)
            evidence*=ok[:,None]&ok[None,:];pairs[r,:n,:n]=evidence
            sign[r,:n]=np.where(perspectives[r,:n]==spec["root_player"],1,-1)
            for a in range(n):
                if not ok[a]:continue
                p=value_counts[r,a].astype(np.float32)/value_counts[r,a].sum()
                probabilities[r,a]=p if sign[r,a]>0 else p[::-1]
        metrics.update(roots=count,confident_pairs=int(pairs.sum()),value_candidates=int(valid.sum()),
                       round_counts={str(r):sum(s["round"]==r for s in specs) for r in sorted({s["round"] for s in specs})},
                       budgets=asdict(self.config),teacher_version=int(self.actor.version),
                       base_search=asdict(self.base_config),tuning=asdict(self.tuning),
                       seed=int(seed),teacher="frozen_current_selfplay",seconds=time.perf_counter()-start)
        data=dict(root=np.stack([s["obs"] for s in specs]).astype(np.float16),
                  children=children,actions=actions,target=probabilities,valid=valid,sign=sign,pairs=pairs,
                  counts=counts,value_counts=value_counts,exact=exact.astype(np.uint8),attempted=attempted,
                  score_sum=score_sum,birth=np.full(count,self.actor.version,np.int64),
                  round=np.asarray([s["round"] for s in specs],np.uint8))
        return data,metrics


class RankingReplay:
    """Small device-resident ring; one record per root regardless of repeats."""
    def __init__(self,capacity,topk,device):
        self.capacity=capacity;self.topk=topk;self.device=torch.device(device);self.size=self.cursor=0
        self.size_gpu=torch.zeros((),dtype=torch.int64,device=self.device)
        shapes=dict(root=((172,),torch.float16),children=((topk,172),torch.float16),
            actions=((topk,),torch.int64),target=((topk,3),torch.float32),valid=((topk,),torch.uint8),
            sign=((topk,),torch.float32),pairs=((topk,topk),torch.float32),counts=((topk,3),torch.int32),
            value_counts=((topk,3),torch.int32),
            exact=((topk,),torch.uint8),attempted=((topk,),torch.int32),score_sum=((topk,),torch.float32),
            birth=((),torch.int64),round=((),torch.uint8))
        self.fields=tuple(shapes)
        for name,(shape,dtype) in shapes.items():
            setattr(self,name,torch.zeros((capacity,*shape),dtype=dtype,device=self.device))

    def add(self,data):
        if data is None:return
        n=min(len(data["root"]),self.capacity);indices=(torch.arange(n,device=self.device)+self.cursor)%self.capacity
        for name in self.fields:getattr(self,name)[indices]=torch.as_tensor(
            data[name][-n:].copy(),dtype=getattr(self,name).dtype,device=self.device)
        self.size=min(self.capacity,self.size+n);self.cursor=(self.cursor+n)%self.capacity;self.size_gpu.fill_(self.size)

    def sample(self,n,deterministic=False):
        size=self.size_gpu.clamp_min(1)
        ids=(torch.arange(n,device=self.device)*size//n)%size if deterministic else (torch.rand(n,device=self.device)*size).long()
        names=("root","children","actions","target","valid","sign","pairs")
        rows={name:getattr(self,name)[ids] for name in names}
        rows["valid"]=rows["valid"]*(self.size_gpu>0)
        rows["pairs"]=rows["pairs"]*(self.size_gpu>0)
        return rows

    def snapshot(self):
        return dict(capacity=self.capacity,topk=self.topk,size=self.size,cursor=self.cursor,
                    data={name:getattr(self,name)[:self.size].cpu() for name in self.fields})

    def restore(self,state):
        if (state["capacity"],state["topk"])!=(self.capacity,self.topk):raise ValueError("Ranking replay shape changed")
        self.size,self.cursor=state["size"],state["cursor"]
        for name in self.fields:getattr(self,name)[:self.size].copy_(state["data"][name])
        self.size_gpu.fill_(self.size)


def ranking_losses(model,rows,device,bf16=False):
    """Policy pairwise ranking, soft successor WDL, and successor value ordering."""
    b,k=rows["actions"].shape
    with torch.autocast(torch.device(device).type,dtype=torch.bfloat16,enabled=bf16,cache_enabled=False):
        logits,_=model(rows["root"].float())
        _,wdl=model(rows["children"].float().reshape(b*k,172))
    selected=logits.float().gather(1,rows["actions"]);pairs=rows["pairs"].float()
    valid=rows["valid"].float();wdl=wdl.float().reshape(b,k,3)
    predicted=wdl.softmax(-1)
    # Value probabilities use each successor's perspective; compare in root view.
    utilities=(predicted[:,:,0]-predicted[:,:,2])*rows["sign"]
    ce=-(rows["target"]*wdl.log_softmax(-1)).sum(-1)
    def per_root_mean(values,weights):
        denominator=weights.flatten(1).sum(-1)
        per_root=(values*weights).flatten(1).sum(-1)/denominator.clamp_min(1)
        exists=(denominator>0).float()
        return (per_root*exists).sum()/exists.sum().clamp_min(1)
    pl=per_root_mean(F.softplus(-(selected[:,:,None]-selected[:,None,:])),pairs)
    vl=per_root_mean(ce,valid)
    # Fixed utility temperature; no learned calibration head.
    vr=per_root_mean(F.softplus(-(utilities[:,:,None]-utilities[:,None,:])/.25),pairs)
    accuracy=per_root_mean((selected[:,:,None]>selected[:,None,:]).float(),pairs)
    value_accuracy=per_root_mean((utilities[:,:,None]>utilities[:,None,:]).float(),pairs)
    return pl,vl,vr,accuracy,value_accuracy
