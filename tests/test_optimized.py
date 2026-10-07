import ctypes as C
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from azul import Batch
from azul_ai.model import PolicyValue,permute_factories
from azul_ai.search import SearchConfig,Search,pointer
from azul_ai.optimized_search import FrozenActor,PipelineSearch,Tuning,BucketEvaluator
from azul_ai.optimized_learning import DeviceReplay,GraphLearner,CheckpointJournal
from azul_ai.optimized_selfplay import collect_historical_cohort
from azul_ai.optimized_train import smooth_weighted_index


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_equivariance_all_permutations_and_after_head(self):
        torch.manual_seed(1);model=PolicyValue(64,2,kind="equivariant").eval()
        obs=torch.rand(1,172);obs[:,168]=0
        logits,value=model(obs)
        for p in itertools.permutations(range(5)):
            perm=torch.tensor([p]);o,expected,_=permute_factories(obs,logits,torch.ones_like(logits),perm)
            actual,v=model(o)
            self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
            self.assertTrue(torch.allclose(v,value,atol=1e-6))
        obs[:,168]=1;_,v=model(obs);v.sum().backward()
        self.assertGreater(model.afterstate.weight.grad.abs().sum().item(),0)
        self.assertEqual(model.value.weight.grad.abs().sum().item(),0)

    def test_smooth_history_schedule(self):
        sequence=[smooth_weighted_index(i,[1,2,3,4]) for i in range(100)]
        self.assertEqual([sequence.count(i) for i in range(4)],[10,20,30,40])
        self.assertTrue(all(len(set(sequence[i:i+10]))>=3 for i in range(0,100,10)))

    def test_cpu_pipeline_active_mask_cache_and_reuse(self):
        model=PolicyValue(32,1,kind="equivariant");actor=FrozenActor(model,"cpu")
        with Batch(16,2) as env,PipelineSearch(env,actor,SearchConfig(simulations=16,candidates=8),
                Tuning(afterstate_prior=0),threads=2,queues=2,graphs=False) as s:
            active=np.zeros(16,bool);active[[0,2,8]]=True
            a,p,_=s.run(55,active);self.assertTrue((a[~active]==65535).all());self.assertTrue(np.allclose(p[active].sum(1),1))
            self.assertLess(s.metrics["valid_leaves"],16*17)
            env.numpy_views()["actions"][:]=a;env.step()
            s.run(56,active);self.assertGreater(s.metrics["reused_nodes"],0)
            actor.publish(model,1);s.run(57,active);self.assertEqual(s.metrics["reused_nodes"],0)

    def test_historical_cohort_keeps_only_learner_policies(self):
        torch.manual_seed(17);learner_model=PolicyValue(32,1,kind="equivariant")
        opponent_model=PolicyValue(32,1,kind="equivariant")
        learner=FrozenActor(learner_model,"cpu");opponent=FrozenActor(opponent_model,"cpu")
        config=SearchConfig(simulations=8,candidates=4)
        tuning=Tuning(afterstate_prior=0,cache_capacity=128)
        with Batch(8,2) as env,PipelineSearch(env,learner,config,tuning,2,1,False) as a, \
                PipelineSearch(env,opponent,config,tuning,2,1,False) as b:
            data,metrics=collect_historical_cohort(env,a,b,1234,8,3)
        obs,policy,mask,outcome=data;after=obs[:,168]>0
        self.assertEqual(metrics["games"],8);self.assertEqual(metrics["truncated"],0)
        self.assertEqual(metrics["learner_decisions"],metrics["decision_positions"])
        self.assertEqual(metrics["learner_decisions"]+metrics["opponent_decisions"],metrics["decisions"])
        self.assertEqual(int((~after).sum()),metrics["learner_decisions"])
        self.assertTrue((policy[~after].sum(1)>0.99).all())
        self.assertTrue((mask[~after].sum(1)>0).all())
        self.assertTrue((policy[after]==0).all() and (mask[after]==0).all())
        self.assertTrue(((outcome>=0)&(outcome<=2)).all())
        self.assertEqual(metrics["games"],8)  # contiguous active lanes retain all games

    def test_journal_ring_metadata_and_missing_delta(self):
        with tempfile.TemporaryDirectory() as temp:
            r=DeviceReplay(17,"cpu");j=CheckpointJournal(temp,3)
            def data(n):return (np.random.rand(n,172).astype(np.float16),np.zeros((n,180),np.float16),
                               np.ones((n,180),np.uint8),np.arange(n,dtype=np.uint8)%3)
            for version in (1,2):
                d=data(12);r.add(d,version);r.sample(20)
                payload=dict(engine_version="test",architecture={},model={},iteration=version,search={},tuning={})
                j.save(payload,r,d,version)
            restored=DeviceReplay(17,"cpu");path=Path(temp)/"latest.pt"
            payload=torch.load(path,weights_only=False)
            CheckpointJournal.restore(path,restored,payload)
            for name in (*r.fields,"seen"):self.assertTrue(torch.equal(getattr(r,name),getattr(restored,name)))
            self.assertEqual((r.size,r.cursor),(restored.size,restored.cursor))
            (Path(temp)/payload["replay_manifest"]["deltas"][0]).unlink()
            with self.assertRaises(FileNotFoundError):CheckpointJournal.restore(path,restored,payload)
            d=data(4);r.add(d,3);j.save(dict(payload,iteration=3),r,d,3)
            current=torch.load(path,weights_only=False)
            CheckpointJournal.restore(path,restored,current)
            self.assertFalse((Path(temp)/"replay-000001.pt").exists())
            self.assertTrue(torch.equal(restored.obs,r.obs))

    def test_native_simd_response_validation(self):
        with Batch(1,1) as b,Search(b,SearchConfig(simulations=8,candidates=4),1) as s:
            lib=b._lib;obs=np.zeros((1,172),np.float32);ready=np.zeros(1,np.uint8)
            logits=np.zeros((1,180),np.float32);values=np.zeros(1,np.float32)
            self.assertEqual(lib.azul_search_begin(s.handle,b._handle,42),0)
            self.assertEqual(lib.azul_search_request(s.handle,pointer(obs),pointer(ready,C.c_uint8)),1)
            for pos,value in ((0,np.inf),(179,np.nan),(73,-np.inf)):
                logits.fill(0);logits[0,pos]=value
                self.assertEqual(lib.azul_search_submit(s.handle,pointer(logits),pointer(values)),-1)
            logits.fill(0)
            self.assertEqual(lib.azul_search_submit(s.handle,pointer(logits),pointer(values)),0)

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_frozen_actor_and_bucket_precision(self):
        model=PolicyValue(64,2,kind="equivariant").cuda();actor=FrozenActor(model,"cuda")
        ev=BucketEvaluator(actor,64);ev.observations[:]=np.random.default_rng(1).random((64,172));ev.observations[:,168]=0
        ready=np.zeros(64,np.uint8);ready[[1,5,63]]=1
        ev.launch(ready);ev.finish();before=ev.logits[[1,5,63]].copy()
        self.assertEqual(ev.metrics["valid_leaves"],3);self.assertEqual(ev.metrics["padded_leaves"],13)
        with torch.no_grad():model.policy.bias.add_(2)
        ev.launch(ready);ev.finish();self.assertTrue(np.array_equal(before,ev.logits[[1,5,63]]))
        actor.publish(model,1);ev.launch(ready);ev.finish();self.assertFalse(np.allclose(before,ev.logits[[1,5,63]]))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_historical_cohort(self):
        learner_model=PolicyValue(32,1,kind="equivariant").cuda()
        opponent_model=PolicyValue(32,1,kind="equivariant").cuda()
        learner=FrozenActor(learner_model,"cuda");opponent=FrozenActor(opponent_model,"cuda")
        config=SearchConfig(simulations=8,candidates=4);tuning=Tuning(afterstate_prior=0,cache_capacity=128)
        with Batch(8,2) as env,PipelineSearch(env,learner,config,tuning,2,1) as a, \
                PipelineSearch(env,opponent,config,tuning,2,1) as b:
            data,metrics=collect_historical_cohort(env,a,b,4321,8,2)
        self.assertEqual(metrics["games"],8);self.assertEqual(metrics["truncated"],0)
        self.assertEqual(int((data[0][:,168]==0).sum()),metrics["learner_decisions"])

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_captured_learning_does_not_update_during_capture(self):
        torch.manual_seed(2);model=PolicyValue(32,1,kind="equivariant").cuda()
        r=DeviceReplay(256,"cuda");obs=np.random.default_rng(42).random((128,172)).astype(np.float16)
        obs[:,168]=0;obs[::4,168]=1
        mask=np.ones((128,180),np.uint8);mask[::4]=0
        policy=mask.astype(np.float16)/180;r.add((obs,policy,mask,np.arange(128,dtype=np.uint8)%3),1)
        opt=torch.optim.AdamW(model.parameters(),lr=0.001,fused=True,capturable=True)
        before={k:v.clone() for k,v in model.state_dict().items()}
        rng=torch.cuda.get_rng_state();learner=GraphLearner(model,opt,r,32)
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(),rng))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertEqual(r.seen.sum().item(),0)
        metrics=learner.run(3)
        self.assertEqual(r.seen.sum().item(),96);self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        self.assertEqual(metrics["invalid_updates"],0)


if __name__=="__main__":unittest.main()
