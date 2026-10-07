import ctypes as C
import itertools
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from azul import Batch
from azul_ai.model import PolicyValue,permute_factories,loss_for_batch
from azul_ai.search import Search,SearchConfig,Evaluator,pointer
from azul_ai.selfplay import split_seed
from azul_ai.train import Replay


class Tests(unittest.TestCase):
    def test_all_120_factory_symmetries(self):
        torch.set_num_threads(2)
        with Batch(1,1,42) as b:
            b.observe();v=b.numpy_views()
            obs=torch.from_numpy(v["observations"].copy())
            mask=torch.from_numpy(v["masks"].copy())
            policy=mask.float()/mask.sum()
            for perm in itertools.permutations(range(5)):
                p=torch.tensor([perm]);o,pi,m=permute_factories(obs,policy,mask,p)
                inverse=p.argsort(1)
                ro,rp,rm=permute_factories(o,pi,m,inverse)
                self.assertTrue(torch.equal(ro,obs) and torch.equal(rp,policy) and torch.equal(rm,mask))
                self.assertTrue(torch.equal(o[:,25:],obs[:,25:]))
                self.assertTrue(torch.equal(pi[:,150:],policy[:,150:]))

    def test_equivalent_factory_moves_in_native_engine(self):
        # Clone snapshots, permute only source counts through verified ctypes layout.
        class Player(C.Structure):
            _fields_=[("wall",C.c_uint32),("score",C.c_uint16),("pc",C.c_uint8*5),
                      ("pn",C.c_uint8*5),("ft",C.c_uint8*5),("fn",C.c_uint8)]
        class State(C.Structure):
            _fields_=[("rng",C.c_uint64),("players",Player*2),("sources",(C.c_uint8*5)*6),
                      ("bag",C.c_uint8*5),("discard",C.c_uint8*5),("round",C.c_uint32),("ply",C.c_uint32),
                      ("bag_size",C.c_uint8),("remaining",C.c_uint8),("current",C.c_uint8),
                      ("next",C.c_uint8),("token",C.c_uint8),("phase",C.c_uint8)]
        with Batch(2,1,42) as b:
            self.assertEqual(C.sizeof(State),b._lib.azul_snapshot_size())
            raw=b.snapshot(0);s=State.from_buffer_copy(raw)
            perm=[4,2,0,1,3];inverse=np.argsort(perm)
            for new,old in enumerate(perm):s.sources[new][:]=State.from_buffer_copy(raw).sources[old][:]
            b.restore(1,bytes(s));b.observe();v=b.numpy_views()
            a=int(np.flatnonzero(v["masks"][0,:150])[0]);mapped=int(inverse[a//30])*30+a%30
            b.step([a,mapped]);b.observe()
            o=v["observations"]
            self.assertTrue(np.array_equal(o[1,:25].reshape(5,5),o[0,:25].reshape(5,5)[perm]))
            self.assertTrue(np.array_equal(o[1,25:],o[0,25:]))

    def test_parallel_search_isolation_and_mask(self):
        torch.manual_seed(5)
        model=PolicyValue(32,1)
        with Batch(9,1,42) as b,Search(b,SearchConfig(simulations=32,candidates=8),1) as serial,Search(b,SearchConfig(simulations=32,candidates=8),4) as parallel:
            ev=Evaluator(model,9,"cpu",False)
            before=[b.snapshot(i) for i in range(9)]
            a,p,v=serial.run(ev,55);a,p,v=a.copy(),p.copy(),v.copy()
            aa,pp,vv=parallel.run(ev,55)
            self.assertTrue(np.array_equal(a,aa) and np.array_equal(p,pp) and np.array_equal(v,vv))
            self.assertEqual(before,[b.snapshot(i) for i in range(9)])
            b.observe();mask=b.numpy_views()["masks"]
            self.assertTrue((p[mask==0]==0).all());self.assertTrue(np.allclose(p.sum(1),1))
            self.assertTrue(mask[np.arange(9),a].all())

    def test_masked_loss_and_replay_wrap(self):
        model=PolicyValue(32,1)
        o=torch.rand(8,172);m=torch.zeros(8,180,dtype=torch.uint8);m[:,[2,30]]=1
        p=m.float()/2;z=torch.arange(8)%3
        loss,_,_=loss_for_batch(model,o,p,m,z);loss.backward()
        self.assertTrue(loss.isfinite());self.assertTrue(all(x.grad.isfinite().all() for x in model.parameters()))
        r=Replay(5)
        data=lambda n:(np.ones((n,172),np.float16),np.ones((n,180),np.float16),np.ones((n,180),np.uint8),np.arange(n,dtype=np.uint8)%3)
        r.add(data(3));r.add(data(4));rr=Replay.restore(r.state())
        self.assertEqual((r.size,r.cursor),(5,2));self.assertTrue(np.array_equal(rr.outcomes,r.outcomes))

    @unittest.skipUnless(torch.cuda.is_available(),"CUDA unavailable")
    def test_cuda_graph_matches_eager_and_tracks_updated_weights(self):
        torch.manual_seed(42)
        model=PolicyValue(64,2).cuda()
        graph=Evaluator(model,16,"cuda",True);eager=Evaluator(model,16,"cuda",False)
        data=np.random.default_rng(1).random((16,172),dtype=np.float32)
        for k in range(2):
            graph.observations[:]=data;eager.observations[:]=data
            a,b=graph.evaluate();c,d=eager.evaluate()
            self.assertTrue(np.allclose(a,c,atol=0.02) and np.allclose(b,d,atol=0.01))
            if k==0:
                old=a.copy()
                with torch.no_grad():model.policy.bias.add_(0.2)
            else:self.assertFalse(np.allclose(old,a))


if __name__=="__main__":unittest.main()
