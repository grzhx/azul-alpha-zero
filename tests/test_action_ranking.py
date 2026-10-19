"""Continuation labels: real chance RNG, exact endings, confidence, CUDA and resume."""
import argparse
import ctypes as C
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from azul import Batch
from azul_ai.action_ranking import (RankingConfig,AnchorSampler,ContinuationRunner,
    RankingReplay,ranking_losses,ranking_evidence,choose_candidates,bind_branch_api)
from azul_ai.model import PolicyValue
from azul_ai.optimized_search import FrozenActor,Tuning,PipelineSearch
from azul_ai.optimized_learning import DeviceReplay,GraphLearner
from azul_ai.optimized_selfplay import collect_cohort
from azul_ai.search import SearchConfig,pointer

from azul import _load
LIBRARY=Path(_load(None)._name)
DEVICE="cpu"


class Player(C.Structure):
    _fields_=[("wall",C.c_uint32),("score",C.c_uint16),("pc",C.c_uint8*5),
              ("pn",C.c_uint8*5),("ft",C.c_uint8*5),("fn",C.c_uint8)]


class State(C.Structure):
    _fields_=[("rng",C.c_uint64),("players",Player*2),("sources",(C.c_uint8*5)*6),
        ("bag",C.c_uint8*5),("discard",C.c_uint8*5),("round",C.c_uint32),("ply",C.c_uint32),
        ("bag_size",C.c_uint8),("remaining",C.c_uint8),("current",C.c_uint8),
        ("next",C.c_uint8),("token",C.c_uint8),("phase",C.c_uint8)]


def fixture(ending=False,round_number=5):
    s=State();s.rng=123;s.round=round_number;s.current=0;s.phase=0
    s.sources[5][0]=1;s.remaining=1;s.bag[:]=[19,20,20,20,20]
    for p in s.players:p.pc[:]=[5]*5
    if ending:
        s.players[0].wall=15;s.players[0].pc[0]=4;s.players[0].pn[0]=1
        s.bag[:]=[18,19,19,19,19]
    s.bag_size=sum(s.bag)
    return bytes(s)


def anchor(env,snapshot):
    env.restore(0,snapshot);env.observe();v=env.numpy_views()
    mask=v["masks"][0].copy();ids=np.flatnonzero(mask)
    return dict(snapshot=snapshot,obs=v["observations"][0].copy(),mask=mask,
                policy=mask.astype(np.float32)/mask.sum(),selected=int(ids[0]),
                root_player=int(v["players"][0]),game=0,category="late",round=5)


class RankingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_confidence_and_budget(self):
        cfg=RankingConfig();cfg.validate()
        self.assertEqual(cfg.budget(4),4096);self.assertEqual(cfg.budget(5),4096)
        self.assertEqual(cfg.budget(3),1024);self.assertEqual(cfg.budget(1),512)
        _,lo,hi,pairs=ranking_evidence([[4,0,0],[0,0,4]],confidence=.95,looks=3)
        self.assertFalse(pairs.any())  # Four lucky outcomes do not prove superiority.
        _,lo,hi,pairs=ranking_evidence([[1,0,0],[0,0,1]],exact=[True,True])
        self.assertEqual(pairs[0,1],1);self.assertEqual(lo[0],hi[0])
        mask=np.zeros(180,np.uint8);mask[[0,30,150]]=1
        obs=np.zeros(172);obs[0]=obs[5]=.1
        self.assertEqual(len(choose_candidates(mask,mask,4,obs=obs)),2)

    def test_native_rng_predeal_and_atomicity(self):
        with Batch(16,2,library=LIBRARY) as env:
            fn=bind_branch_api(env._lib);self.assertEqual(C.sizeof(State),env._lib.azul_snapshot_size())
            snapshot=fixture(False);snapshots=np.stack([np.frombuffer(snapshot,np.uint8)]*16)
            seeds=np.arange(16,dtype=np.uint64)+100
            actions=np.full(16,155,np.uint16);obs=np.zeros((16,172),np.float32);view=np.zeros(16,np.uint8)
            def call():return fn(env._handle,16,snapshots.ctypes.data,snapshots.shape[1],pointer(seeds,C.c_uint64),
                pointer(actions,C.c_uint16),pointer(obs),pointer(view,C.c_uint8),env.results)
            self.assertEqual(call(),0)
            self.assertTrue((obs[:,168]==1).all())  # Afterstate BEFORE deal.
            env.observe();self.assertTrue((env.numpy_views()["observations"][:,170]==.6).all())
            post=[env.snapshot(i) for i in range(16)]
            self.assertGreater(len(set(x[56:86] for x in post)),1)  # Independent factory outcomes.
            self.assertEqual(call(),0);self.assertEqual(post,[env.snapshot(i) for i in range(16)])
            actions[-1]=65535;self.assertEqual(call(),-1)
            self.assertEqual(post,[env.snapshot(i) for i in range(16)])

    def test_equal_scores_use_completed_rows(self):
        with Batch(1,1,library=LIBRARY) as env:
            s=State.from_buffer_copy(fixture(True));s.players[1].score=7
            env.restore(0,bytes(s));env.step([151])
            self.assertEqual(list(env.results[0].scores),[7,7])
            self.assertEqual(env.results[0].reward,1)  # Row tiebreak is a win, not draw.

    def test_exact_terminal_and_root_perspective(self):
        model=PolicyValue(32,1,kind="equivariant",end_features=True)
        actor=FrozenActor(model,DEVICE);actor.publish(model,1)
        cfg=RankingConfig(positions=2,topk=4,rollouts=2,round3_rollouts=2,round45_rollouts=2,
            max_rollouts=4,simulations=4,round3_simulations=4,round45_simulations=8,envs=8)
        with Batch(1,1,library=LIBRARY) as env:
            a=anchor(env,fixture(True))
        with ContinuationRunner(actor,SearchConfig(simulations=4,candidates=4),Tuning(cache_capacity=64),
                                cfg,2,1,False,LIBRARY) as runner:
            data,metrics=runner.evaluate([a],421)
        self.assertTrue(data["exact"].any());self.assertTrue(data["valid"].any())
        valid=data["valid"].astype(bool)
        np.testing.assert_allclose(data["target"][valid].sum(-1),1)
        # Actor 0 wins; successor is from player 1 view, so its target is loss.
        self.assertTrue((data["sign"][valid]==-1).all())
        self.assertTrue((data["target"][valid][:,2]==1).all())
        self.assertTrue((data["children"][valid][:,169]==1).all())
        self.assertEqual(data["root"].shape[0],1)  # Repeats never inflate root count.

    def test_losses_learn_order_and_cuda_capture(self):
        device=torch.device(DEVICE);model=PolicyValue(32,1,kind="equivariant",end_features=True).to(device)
        rr=RankingReplay(8,2,device);rng=np.random.default_rng(1)
        data=dict(root=rng.random((2,172)).astype(np.float16),
            children=rng.random((2,2,172)).astype(np.float16),actions=np.array([[0,150],[1,151]]),
            target=np.array([[[1,0,0],[0,0,1]]]*2,np.float32),valid=np.ones((2,2),np.uint8),
            sign=np.ones((2,2),np.float32),pairs=np.array([[[0,1],[0,0]]]*2,np.float32),
            counts=np.array([[[10,0,0],[0,0,10]]]*2,np.int32),exact=np.zeros((2,2),np.uint8),
            value_counts=np.array([[[10,0,0],[0,0,10]]]*2,np.int32),
            attempted=np.full((2,2),10,np.int32),score_sum=np.zeros((2,2),np.float32),
            birth=np.ones(2,np.int64),round=np.array([4,5],np.uint8))
        data["root"][:,168:170]=0;data["children"][:,:,168:170]=0
        rr.add(data);snapshot=rr.snapshot();rr2=RankingReplay(8,2,device);rr2.restore(snapshot)
        for name in rr.fields:self.assertTrue(torch.equal(getattr(rr,name)[:2],getattr(rr2,name)[:2]))
        ordinary=DeviceReplay(64,device)
        mask=np.ones((64,180),np.uint8)
        obs=rng.random((64,172)).astype(np.float16);obs[:,168:170]=0
        ordinary.add((obs,mask.astype(np.float16)/180,mask,np.arange(64,dtype=np.uint8)%3),1)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,fused=DEVICE=="cuda",capturable=DEVICE=="cuda")
        learner=GraphLearner(model,opt,ordinary,16,graphs=DEVICE=="cuda",ranking_replay=rr,ranking_batch_size=2)
        metrics=learner.run(2)
        self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
        self.assertIn("continuation_value_ce",metrics)
        seen=ordinary.seen.clone();rng_before=torch.get_rng_state().clone()
        learner.diagnose_gradients()
        self.assertTrue(torch.equal(seen,ordinary.seen));self.assertTrue(torch.equal(rng_before,torch.get_rng_state()))

    def test_pairwise_gradient_and_successor_sign(self):
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__();self.policy=torch.nn.Parameter(torch.zeros(180,device=DEVICE))
                self.values=torch.nn.Parameter(torch.zeros((2,3),device=DEVICE))
            def forward(self,x):
                return self.policy.expand(len(x),-1),self.values[x[:,0].long()]
        model=Probe();children=torch.zeros((1,2,172),device=DEVICE);children[0,1,0]=1
        rows=dict(root=torch.zeros((1,172),device=DEVICE),children=children,
            actions=torch.tensor([[0,150]],device=DEVICE),target=torch.tensor([[[0.,0.,1.],[1.,0.,0.]]],device=DEVICE),
            valid=torch.ones((1,2),device=DEVICE),sign=-torch.ones((1,2),device=DEVICE),
            pairs=torch.tensor([[[0.,1.],[0.,0.]]],device=DEVICE))
        pl,vl,vr,_,_=ranking_losses(model,rows,DEVICE)
        pl.backward(retain_graph=True)
        self.assertLess(model.policy.grad[0].item(),0);self.assertGreater(model.policy.grad[150].item(),0)
        vr.backward()
        # Root prefers successor 0. Both successors are opponents to move:
        # raising successor 0's loss probability should improve root utility.
        self.assertLess(model.values.grad[0,2].item(),0)
        self.assertGreater(model.values.grad[0,0].item(),0)

    def test_holdout_exclusion(self):
        model=PolicyValue(32,1,kind="equivariant",end_features=True);actor=FrozenActor(model,"cpu")
        config=RankingConfig(positions=4)
        with Batch(8,2,library=LIBRARY) as env,PipelineSearch(env,actor,
            SearchConfig(simulations=4,candidates=4),Tuning(cache_capacity=64),2,1,False) as search:
            _,metrics=collect_cohort(env,search,53,8,holdout_games=2,ranking_config=config)
            anchors=metrics["ranking_anchors"]
            self.assertTrue(anchors);self.assertTrue(all(a["game"]>=2 for a in anchors))
            self.assertLessEqual(len(anchors),4)

    def test_wave_version_blocks_cross_replicate_reuse(self):
        model=PolicyValue(32,1,kind="equivariant");actor=FrozenActor(model,"cpu");actor.publish(model,1)
        with Batch(2,1,library=LIBRARY) as env,PipelineSearch(env,actor,
            SearchConfig(simulations=4,candidates=4),Tuning(cache_capacity=64),1,1,False) as search:
            actions,pi,values=search.run(0,np.zeros(2,bool),tree_version=99)
            self.assertTrue((actions==65535).all());self.assertFalse(pi.any());self.assertFalse(values.any())
            search.run(1,tree_version=100)
            search.run(2,tree_version=100)
            self.assertGreater(search.metrics["reused_nodes"],0)
            search.run(3,tree_version=101)
            self.assertEqual(search.metrics["reused_nodes"],0)
            self.assertEqual(actor.version,1)  # Reuse epoch does not publish new weights.
            before=env.snapshot(0)
            search.release_arenas()
            selected,_,_=search.run(3,tree_version=101)
            self.assertEqual(search.metrics["reused_nodes"],0)
            self.assertEqual(before,env.snapshot(0))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    args,left=p.parse_known_args();DEVICE=args.device
    unittest.main(argv=[sys.argv[0],*left])
