"""Device-resident replay, captured optimizer steps, incremental checkpoints."""
import copy
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .model import permute_factories
from .policy_diagnostics import policy_metrics
from .train import save_checkpoint
from .checkpoint_storage import save_replay,load_replay,retention,checked_path
from .action_ranking import ranking_losses


class DeviceReplay:
    fields=("obs","policy","mask","outcome","birth")
    def __init__(self,capacity,device="cuda",recent_fraction=0.25):
        self.capacity,self.device=capacity,torch.device(device)
        self.size=self.cursor=0;self.recent_fraction=recent_fraction
        self.obs=torch.empty((capacity,172),device=self.device,dtype=torch.float16)
        self.policy=torch.empty((capacity,180),device=self.device,dtype=torch.float16)
        self.mask=torch.empty((capacity,180),device=self.device,dtype=torch.uint8)
        self.outcome=torch.empty(capacity,device=self.device,dtype=torch.uint8)
        self.birth=torch.empty(capacity,device=self.device,dtype=torch.int64)
        self.seen=torch.zeros(capacity,device=self.device,dtype=torch.int32)
        self.size_gpu=torch.zeros((),device=self.device,dtype=torch.int64)
        self.cursor_gpu=torch.zeros_like(self.size_gpu)
        self.version_gpu=torch.zeros_like(self.size_gpu)

    def add(self,data,version):
        if len(data[0])>self.capacity:data=tuple(x[-self.capacity:] for x in data)
        n=len(data[0]);indices=(torch.arange(n,device=self.device)+self.cursor)%self.capacity
        for name,x in zip(self.fields[:4],data):
            getattr(self,name)[indices]=torch.as_tensor(x,device=self.device,dtype=getattr(self,name).dtype)
        self.birth[indices]=version;self.seen[indices]=0
        self.size=min(self.capacity,self.size+n);self.cursor=(self.cursor+n)%self.capacity
        self.size_gpu.fill_(self.size);self.cursor_gpu.fill_(self.cursor);self.version_gpu.fill_(version)

    def sample(self,batch_size):
        # Dynamic size/cursor remain tensors, so graph replay sees newly inserted data.
        u=torch.rand(batch_size,device=self.device)
        uniform=(u*self.size_gpu).long()
        recent_count=(self.size_gpu//5).clamp_min(1)
        recent=(self.cursor_gpu-1-(u*recent_count).long())%self.size_gpu
        selector=torch.arange(batch_size,device=self.device)<int(batch_size*self.recent_fraction)
        indices=torch.where(selector,recent,uniform)
        self.seen.scatter_add_(0,indices,torch.ones_like(indices,dtype=torch.int32))
        return tuple(getattr(self,name)[indices] for name in self.fields[:4]),self.version_gpu-self.birth[indices]

    def snapshot(self):
        return dict(capacity=self.capacity,size=self.size,cursor=self.cursor,recent_fraction=self.recent_fraction,
                    version=int(self.version_gpu.item()),
                    **{name:getattr(self,name)[:self.size].cpu() for name in self.fields},seen=self.seen[:self.size].cpu())

    def restore(self,state):
        if state["capacity"]!=self.capacity:raise ValueError("Replay capacity mismatch")
        self.size,self.cursor=state["size"],state["cursor"]
        for name in (*self.fields,"seen"):getattr(self,name)[:self.size].copy_(state[name])
        self.size_gpu.fill_(self.size);self.cursor_gpu.fill_(self.cursor);self.version_gpu.fill_(state["version"])

    def expanded(self,capacity,device,recent_fraction):
        """Preserve chronology, labels, ages and usage when enlarging a restored ring."""
        if capacity<self.capacity:raise ValueError("Replay migration only supports expansion")
        result=DeviceReplay(capacity,device,recent_fraction)
        if self.size==self.capacity:
            order=(torch.arange(self.size,device=self.device)+self.cursor)%self.capacity
        else:order=torch.arange(self.size,device=self.device)
        for name in (*self.fields,"seen"):
            getattr(result,name)[:self.size].copy_(getattr(self,name)[order])
        result.size=self.size;result.cursor=self.size%capacity
        result.size_gpu.fill_(result.size);result.cursor_gpu.fill_(result.cursor)
        result.version_gpu.copy_(self.version_gpu)
        return result

    def metadata(self):
        return dict(size=self.size,cursor=self.cursor,version=int(self.version_gpu.item()),seen=self.seen[:self.size].cpu())

    def statistics(self):
        age=(self.version_gpu-self.birth[:self.size]).float();seen=self.seen[:self.size].float()
        rounds=(self.obs[:self.size,170].float()*10).round()
        return dict(replay_size=self.size,replay_mean_age=float(age.mean().item()),
                    replay_max_age=float(age.max().item()),replay_mean_sample_count=float(seen.mean().item()),
                    replay_unseen_fraction=float((seen==0).float().mean().item()),
                    replay_afterstate_fraction=float(self.obs[:self.size,168].float().mean().item()),
                    replay_early_round_fraction=float((rounds<=3).float().mean().item()),
                    replay_middle_round_fraction=float(((rounds>3)&(rounds<=6)).float().mean().item()),
                    replay_late_round_fraction=float((rounds>6).float().mean().item()))


class GraphLearner:
    def __init__(self,model,optimizer,replay,batch_size=1024,after_weight=0.25,
                 graphs=True,bf16=True,*,value_rank_weight=0.0,value_label_smoothing=0.0,
                 policy_loss_weight=1.0,value_loss_weight=1.0,diagnostics=True,
                 ranking_replay=None,ranking_batch_size=16,ranking_policy_weight=.05,
                 ranking_value_weight=.1,ranking_value_order_weight=.05):
        self.model,self.optimizer,self.replay=model,optimizer,replay
        self.batch_size,self.after_weight=batch_size,after_weight
        self.value_rank_weight=float(value_rank_weight)
        self.value_label_smoothing=float(value_label_smoothing)
        self.policy_loss_weight=float(policy_loss_weight)
        self.value_loss_weight=float(value_loss_weight)
        self.diagnostics_enabled=bool(diagnostics)
        self.ranking_replay=ranking_replay;self.ranking_batch_size=ranking_batch_size
        self.ranking_weights=(ranking_policy_weight,ranking_value_weight,ranking_value_order_weight)
        self.device=replay.device;self.cuda=self.device.type=="cuda";self.bf16=bf16 and self.cuda
        self.graph=None
        # loss, policy, decision-value, after-value, Brier, age, gradnorm,
        # decision utility MSE, afterstate utility MSE, invalid, policy KL,
        # policy top1 agreement, policy target entropy, policy signal norm,
        # value signal norm
        self.totals=torch.zeros(17,device=self.device)
        self.ranking_totals=torch.zeros(5,device=self.device)
        if self.cuda and graphs:
            params={k:v.detach().clone() for k,v in model.state_dict().items()}
            old_opt={p:{k:v.clone() if torch.is_tensor(v) else copy.deepcopy(v) for k,v in state.items()}
                     for p,state in optimizer.state.items()}
            seen=replay.seen.clone();rng=torch.cuda.get_rng_state(self.device)
            stream=torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):self._step()
            stream.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):self._step()
            torch.cuda.synchronize(self.device)
            model.load_state_dict(params)
            for p,state in optimizer.state.items():
                for k,v in state.items():
                    if torch.is_tensor(v):
                        if p in old_opt and k in old_opt[p]:v.copy_(old_opt[p][k])
                        else:v.zero_()
            replay.seen.copy_(seen);self.totals.zero_();self.ranking_totals.zero_();torch.cuda.set_rng_state(rng,self.device)

    def _step(self):
        (obs,target,mask,outcome),ages=self.replay.sample(self.batch_size)
        obs=obs.float();target=target.float();target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
        permutations=torch.rand((self.batch_size,5),device=self.device).argsort(-1)
        obs,target,mask=permute_factories(obs,target,mask,permutations)
        after=obs[:,168]>0;decision=~after
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.bf16,cache_enabled=False):
            logits,wdl=self.model(obs)
        logits=logits.float().masked_fill(~mask.bool(),-1e9)
        pl=-(target*F.log_softmax(logits,-1)).sum(-1)
        hard=F.one_hot(outcome.long(),3).float()
        smooth=self.value_label_smoothing
        targets_value=hard*(1.0-smooth)+smooth/3.0
        vl=-(targets_value*F.log_softmax(wdl.float(),-1)).sum(-1)
        probs=wdl.float().softmax(-1)
        utility_target=torch.where(outcome==0,1.0,torch.where(outcome==2,-1.0,0.0)).float()
        utility_pred=probs[:,0]-probs[:,2]
        rank_loss=(utility_pred-utility_target).square()
        diagnostics=policy_metrics(logits,target,mask)
        def decision_mean(values):
            return (values*decision).sum()/decision.sum().clamp_min(1)
        policy_signal=decision_mean((logits.softmax(-1)-target).square().sum(-1).sqrt())
        policy_loss=(pl*decision).sum()/decision.sum().clamp_min(1)
        decision_loss=(vl*decision).sum()/decision.sum().clamp_min(1)
        after_loss=(vl*after).sum()/after.sum().clamp_min(1)
        rank_decision=(rank_loss*decision).sum()/decision.sum().clamp_min(1)
        rank_after=(rank_loss*after).sum()/after.sum().clamp_min(1)
        value_signal=((wdl.float().softmax(-1)-targets_value).square().sum(-1).sqrt()).mean()
        loss=self.policy_loss_weight*policy_loss+self.value_loss_weight*(decision_loss+self.after_weight*after_loss)+self.value_rank_weight*(rank_decision+self.after_weight*rank_after)
        if self.ranking_replay is not None:
            ranking=ranking_losses(self.model,self.ranking_replay.sample(self.ranking_batch_size),self.device,self.bf16)
            loss=loss+sum(w*l for w,l in zip(self.ranking_weights,ranking[:3]))
            self.ranking_totals.add_(torch.stack([x.detach() for x in ranking]))
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0,foreach=True)
        self.optimizer.step()
        probs=wdl.float().softmax(-1)
        brier=((probs-F.one_hot(outcome.long(),3))**2).sum(-1).mean()
        invalid=(~torch.isfinite(loss) | ~torch.isfinite(norm)).float()
        self.totals.add_(torch.stack((loss.detach(),policy_loss.detach(),decision_loss.detach(),after_loss.detach(),
                                     brier.detach(),ages.float().mean(),norm.detach(),
                                     rank_decision.detach(),rank_after.detach(),invalid,
                                     decision_mean(diagnostics["policy_target_kl"]),
                                     decision_mean(diagnostics["policy_top1_agreement"]),
                                     decision_mean(diagnostics["policy_target_entropy"]),
                                     policy_signal.detach(),value_signal.detach(),
                                     decision_mean(diagnostics["policy_top3_agreement"]),
                                     decision_mean(diagnostics["policy_top3_target_mass"]))))

    def _shared_parameters(self):
        return [p for name,p in self.model.named_parameters()
                if p.requires_grad and name.startswith(("factory_encoder.","input.","blocks.","norm."))]

    def diagnose_gradients(self):
        """Measure raw and configured shared-trunk gradient balance on one batch."""
        if not self.diagnostics_enabled or self.replay.size < 1:
            return {}
        # Stratified deterministic reads span the ring without touching its seen
        # counters or the CPU/CUDA RNG used for sampling and augmentation.
        indices=((torch.arange(self.batch_size,device=self.device)*self.replay.size_gpu)//self.batch_size
                 +self.replay.cursor_gpu)%self.replay.size_gpu
        obs,target,mask,outcome=(getattr(self.replay,name)[indices] for name in self.replay.fields[:4])
        obs=obs.float();target=target.float();target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
        decision=obs[:,168]==0
        after=~decision
        with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.bf16,cache_enabled=False):
            logits,wdl=self.model(obs)
        logits=logits.float().masked_fill(~mask.bool(),-1e9)
        policy=( -(target*F.log_softmax(logits,-1)).sum(-1)*decision).sum()/decision.sum().clamp_min(1)
        hard=F.one_hot(outcome.long(),3).float()
        smooth=self.value_label_smoothing
        value_target=hard*(1.0-smooth)+smooth/3.0
        value_each=-(value_target*F.log_softmax(wdl.float(),-1)).sum(-1)
        value=(value_each*decision).sum()/decision.sum().clamp_min(1)
        value=value+self.after_weight*(value_each*after).sum()/after.sum().clamp_min(1)
        utility=wdl.float().softmax(-1)
        terminal=torch.where(outcome==0,1.,torch.where(outcome==2,-1.,0.))
        utility_loss=(utility[:,0]-utility[:,2]-terminal).square()
        auxiliary=self.value_rank_weight*((utility_loss*decision).sum()/decision.sum().clamp_min(1)
                         +self.after_weight*(utility_loss*after).sum()/after.sum().clamp_min(1))
        configured=self.value_loss_weight*value+auxiliary
        if self.ranking_replay is not None:
            rp,rv,ro,_,_=ranking_losses(self.model,self.ranking_replay.sample(
                self.ranking_batch_size,deterministic=True),self.device,self.bf16)
            policy=policy+self.ranking_weights[0]/self.policy_loss_weight*rp
            extra=self.ranking_weights[1]*rv+self.ranking_weights[2]*ro
            configured=configured+extra
            auxiliary=auxiliary+extra
        shared=self._shared_parameters()
        pg=torch.autograd.grad(policy,shared,retain_graph=True,allow_unused=True)
        vg=torch.autograd.grad(value+auxiliary,shared,retain_graph=True,allow_unused=True)
        configured_vg=torch.autograd.grad(configured,shared,allow_unused=True)
        pn=torch.sqrt(sum((g.float().square().sum() for g in pg if g is not None),torch.zeros((),device=self.device)))
        vn=torch.sqrt(sum((g.float().square().sum() for g in vg if g is not None),torch.zeros((),device=self.device)))
        configured_norm=torch.sqrt(sum((g.float().square().sum() for g in configured_vg if g is not None),torch.zeros((),device=self.device)))
        dot=sum(((p.float()*v.float()).sum() for p,v in zip(pg,configured_vg) if p is not None and v is not None),
                torch.zeros((),device=self.device))
        cosine=dot/(pn*configured_norm).clamp_min(1e-12)
        vector=torch.stack((pn,vn,configured_norm,vn/pn.clamp_min(1e-12),
                            configured_norm/(self.policy_loss_weight*pn).clamp_min(1e-12),
                            cosine,(cosine<0).float())).detach().cpu().tolist()
        return dict(zip(("policy_shared_grad_norm","value_shared_grad_norm","configured_value_shared_grad_norm",
                         "raw_value_policy_grad_ratio","weighted_value_policy_grad_ratio",
                         "policy_value_grad_cosine","policy_value_grad_conflict"),vector))

    def run(self,steps):
        self.model.train();self.totals.zero_();self.ranking_totals.zero_()
        for _ in range(steps):
            if self.graph is None:self._step()
            else:self.graph.replay()
        # One required diagnostic transfer at the publication boundary, none per update.
        values=self.totals.detach().cpu().numpy()/max(1,steps)
        if not np.isfinite(values).all() or values[9]!=0:raise FloatingPointError("Nonfinite learner; checkpoint not written")
        result=dict(zip(("loss","policy_loss","value_loss","afterstate_loss","training_batch_brier", "sample_mean_age","gradient_norm",
                         "value_utility_mse","afterstate_utility_mse","invalid_updates","policy_target_kl",
                         "policy_top1_agreement","policy_target_entropy","policy_signal_norm","value_signal_norm",
                         "policy_top3_agreement","policy_top3_target_mass"),map(float,values)))
        result.update(self.diagnose_gradients())
        if self.ranking_replay is not None:
            result.update(dict(zip(("ranking_policy_loss","continuation_value_ce","ranking_value_loss",
                                    "ranking_policy_accuracy","ranking_value_accuracy"),
                                   map(float,self.ranking_totals.detach().cpu().numpy()/max(1,steps)))))
        return result


class CheckpointJournal:
    def __init__(self,output,full_interval=10,compression="gzip",keep_recent=2,keep_anchors=4):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.interval=full_interval;self.base=None;self.deltas=[]
        self.suffix=".pt.gz" if compression=="gzip" else ".pt"
        self.keep_recent,self.keep_anchors=keep_recent,keep_anchors
        self.force_full=False

    def save(self,payload,replay,data,iteration):
        obsolete=[]
        if self.force_full or self.base is None or iteration%self.interval==0:
            if self.base is not None:obsolete=[self.base,*self.deltas]
            self.base=f"replay-{iteration:06d}{self.suffix}";self.deltas=[]
            save_replay(self.output/self.base,replay.snapshot())
        else:
            name=f"replay-delta-{iteration:06d}{self.suffix}"
            save_replay(self.output/name,dict(data=data,version=iteration))
            self.deltas.append(name)
        payload=dict(payload,replay_manifest=dict(base=self.base,deltas=list(self.deltas)),replay_metadata=replay.metadata())
        save_checkpoint(self.output/"latest.pt",payload)
        self.force_full=False
        actor={k:payload[k] for k in ("engine_version","architecture","model","iteration","search","tuning")}
        save_checkpoint(self.output/f"actor-{iteration:06d}.pt",actor)
        # Reclaim only superseded journal files AFTER the new manifest is committed.
        # Actor snapshots are retained. No recursive deletion or globbing.
        for name in obsolete:
            path=self.output/name
            if path.parent.resolve()!=self.output.resolve() or path.name!=name:
                raise ValueError("Unexpected replay journal path")
            path.unlink(missing_ok=True)
        retention(self.output,self.keep_recent,self.keep_anchors)

    @staticmethod
    def restore(path,replay,payload):
        root=Path(path).parent;manifest=payload["replay_manifest"]
        replay.restore(load_replay(checked_path(root,manifest["base"])))
        for name in manifest["deltas"]:
            delta=load_replay(checked_path(root,name))
            replay.add(delta["data"],delta["version"])
        meta=payload["replay_metadata"]
        if (replay.size,replay.cursor)!=(meta["size"],meta["cursor"]):raise ValueError("Replay journal inconsistent")
        replay.seen[:replay.size].copy_(meta["seen"]);replay.version_gpu.fill_(meta["version"])
