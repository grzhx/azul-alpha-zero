"""Protocol migration must preserve the old ring and permit subsequent exact resume."""
import subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from azul_ai.optimized_learning import DeviceReplay,CheckpointJournal

r=DeviceReplay(7,"cpu")
for i in (1,2):
    x=np.full((5,172),i,np.float16)
    r.add((x,np.ones((5,180),np.float16),np.ones((5,180),np.uint8),np.arange(5,dtype=np.uint8)%3),i)
r.sample(20);order=(torch.arange(7)+r.cursor)%7
grown=r.expanded(13,"cpu",0)
assert grown.size==7 and grown.cursor==7 and grown.recent_fraction==0
for k in (*r.fields,"seen"):assert torch.equal(getattr(grown,k)[:7],getattr(r,k)[order]),k

def run(args,success=True):
    c=subprocess.run([sys.executable,str(ROOT/"scripts/train_optimized.py"),*args,"--device","cpu"],cwd=ROOT,capture_output=True,text=True)
    if (c.returncode==0)!=success:raise AssertionError(c.stdout+c.stderr)
    return c
with tempfile.TemporaryDirectory(prefix="azul-retune-") as tmp:
    root=Path(tmp);output=root/"run"
    run(["--output",str(output),"--device","cpu","--envs","4","--threads","2","--queues","1",
        "--positions-per-update","320","--simulations","8","--candidates","4","--width","32","--blocks","1",
        "--replay-capacity","1000","--batch-size","32","--warmup","1","--updates-per-iteration","2","--iterations","1"])
    latest=output/"latest.pt";old=torch.load(latest,weights_only=False,map_location="cpu")
    run(["--resume",str(latest),"--iterations","2","--learning-rate","0.0001"],False)
    run(["--resume",str(latest),"--retune","--iterations","2","--learning-rate","0.0001",
         "--replay-capacity","2000","--recent-fraction","0","--sample-reuse","2","--simulations","16"])
    new=torch.load(latest,weights_only=False,map_location="cpu")
    assert len(new["phase_history"])==1 and len(new["phase_history"][0]["changes"])==5
    assert new["optimizer"]["param_groups"][0]["lr"]==0.0001
    assert new["updates"]==old["updates"]+2
    assert new["replay_manifest"]["deltas"]==[] and "000002" in new["replay_manifest"]["base"]
    replay=DeviceReplay(2000,"cpu",0);CheckpointJournal.restore(latest,replay,new)
    assert (replay.birth[:replay.size]==1).any() and (replay.birth[:replay.size]==2).any()
    run(["--resume",str(latest),"--iterations","3"])
    final=torch.load(latest,weights_only=False,map_location="cpu")
    assert len(final["phase_history"])==1 and final["args"]["simulations"]==16
    assert final["optimizer"]["param_groups"][0]["lr"]==0.0001
print("Retune passed: chronology, optimizer LR, full replay migration, phase history and subsequent resume")
