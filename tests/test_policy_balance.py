"""Analytic policy metrics and state-preserving gradient diagnostics."""
import argparse
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from azul_ai.model import PolicyValue,permute_factories
from azul_ai.optimized_learning import DeviceReplay,GraphLearner
from azul_ai.policy_diagnostics import policy_metrics

DEVICE="cpu"


def make_learner(policy=1.2,value=.8,auxiliary=.1):
    torch.manual_seed(421)
    model=PolicyValue(32,1,kind="equivariant",end_features=True,value_tower=True).to(DEVICE)
    replay=DeviceReplay(128,DEVICE,0.)
    rng=np.random.default_rng(11)
    obs=rng.random((64,172),dtype=np.float32);obs[:,168]=0;obs[::4,168]=1
    mask=np.ones((64,180),np.uint8);mask[::4]=0
    target=rng.random((64,180),dtype=np.float32);target[::4]=0
    replay.add((obs,target,mask,rng.integers(0,3,64,dtype=np.uint8)),2200)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-5,fused=DEVICE=="cuda",capturable=DEVICE=="cuda")
    learner=GraphLearner(model,optimizer,replay,32,graphs=False,bf16=False,value_rank_weight=auxiliary,
                         value_label_smoothing=.01,policy_loss_weight=policy,value_loss_weight=value)
    return learner


class BalanceTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_graph_balanced_updates(self):
        if DEVICE!="cuda":self.skipTest("Run --device cuda for graph check")
        original=make_learner()
        learner=GraphLearner(original.model,original.optimizer,original.replay,32,graphs=True,bf16=True,
                             value_rank_weight=.1,value_label_smoothing=.01,
                             policy_loss_weight=1.2,value_loss_weight=.8)
        metrics=learner.run(2)
        self.assertEqual(metrics["invalid_updates"],0.)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))
        self.assertTrue(0<=metrics["policy_top1_agreement"]<=1)
        self.assertTrue(0<=metrics["policy_top3_target_mass"]<=1.0001)

    def test_analytic_metrics(self):
        probabilities=torch.tensor([[.6,.3,.1,.01]],device=DEVICE)
        target=torch.tensor([[.5,.5,0.,0.]],device=DEVICE)
        mask=torch.tensor([[1,1,1,0]],device=DEVICE)
        metrics=policy_metrics(probabilities.log(),target,mask)
        entropy=float(metrics["policy_target_entropy"])
        kl=float(metrics["policy_target_kl"])
        self.assertAlmostEqual(entropy,np.log(2),places=6)
        self.assertAlmostEqual(kl,-.5*np.log(.6)-.5*np.log(.3)-np.log(2),places=6)
        self.assertEqual(float(metrics["policy_top1_agreement"]),1.)
        self.assertEqual(float(metrics["policy_top3_agreement"]),1.)
        self.assertEqual(float(metrics["policy_top3_target_mass"]),1.)

    def test_diagnostic_is_read_only_and_weights_scale_ratio(self):
        learner=make_learner(auxiliary=0.)
        learner.run(1)
        weights={k:v.clone() for k,v in learner.model.state_dict().items()}
        grads=[p.grad.clone() if p.grad is not None else None for p in learner.model.parameters()]
        seen=learner.replay.seen.clone()
        cpu_rng=torch.get_rng_state().clone()
        cuda_rng=torch.cuda.get_rng_state().clone() if DEVICE=="cuda" else None
        result=learner.diagnose_gradients()
        self.assertEqual(set(learner._shared_parameters()),set(
            p for n,p in learner.model.named_parameters() if n.startswith(("factory_encoder.","input.","blocks.","norm."))))
        for key,tensor in weights.items():self.assertTrue(torch.equal(tensor,learner.model.state_dict()[key]),key)
        for before,param in zip(grads,learner.model.parameters()):
            self.assertTrue(param.grad is None if before is None else torch.equal(before,param.grad))
        self.assertTrue(torch.equal(seen,learner.replay.seen))
        self.assertTrue(torch.equal(cpu_rng,torch.get_rng_state()))
        if cuda_rng is not None:self.assertTrue(torch.equal(cuda_rng,torch.cuda.get_rng_state()))
        self.assertAlmostEqual(result["weighted_value_policy_grad_ratio"],
                               result["raw_value_policy_grad_ratio"]*.8/1.2,places=5)
        self.assertTrue(-1.00001 <= result["policy_value_grad_cosine"] <= 1.00001)

    def test_weighted_update_uses_full_value_objective(self):
        learner=make_learner()
        reference=copy.deepcopy(learner.model)
        cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state() if DEVICE=="cuda" else None
        (obs,target,mask,outcome),_=learner.replay.sample(32)
        obs=obs.float();target=target.float();target=target/target.sum(-1,keepdim=True).clamp_min(1e-12)
        permutations=torch.rand((32,5),device=DEVICE).argsort(-1)
        obs,target,mask=permute_factories(obs,target,mask,permutations)
        logits,wdl=reference(obs);decision=obs[:,168]==0;after=~decision
        policy=-(target*logits.float().masked_fill(~mask.bool(),-1e9).log_softmax(-1)).sum(-1)
        target_value=F.one_hot(outcome.long(),3).float()*.99+.01/3
        value=-(target_value*wdl.float().log_softmax(-1)).sum(-1)
        probs=wdl.float().softmax(-1)
        y=torch.where(outcome==0,1.,torch.where(outcome==2,-1.,0.))
        utility=(probs[:,0]-probs[:,2]-y).square()
        def mean(x,m):return (x*m).sum()/m.sum().clamp_min(1)
        expected=1.2*mean(policy,decision)+.8*(mean(value,decision)+.25*mean(value,after))
        expected+=.1*(mean(utility,decision)+.25*mean(utility,after))
        expected.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,foreach=True)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
        with patch.object(learner.optimizer,"step",return_value=None):learner._step()
        for actual,wanted in zip(learner.model.parameters(),reference.parameters()):
            if wanted.grad is None:self.assertIsNone(actual.grad)
            else:self.assertTrue(torch.allclose(actual.grad,wanted.grad,atol=1e-6,rtol=1e-5))
        self.assertAlmostEqual(float(learner.totals[0]),float(expected.detach()),places=5)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args,remaining=parser.parse_known_args();DEVICE=args.device
    torch.set_num_threads(2)
    unittest.main(argv=[sys.argv[0],*remaining])
