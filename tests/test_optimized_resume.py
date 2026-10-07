"""Exact resume check for incremental replay + GPU/CPU RNG + captured learner."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from azul_ai.optimized_learning import DeviceReplay,CheckpointJournal

p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");args=p.parse_args()
root=Path(__file__).resolve().parents[1]
def run(path,iterations,resume=None):
    cmd=[sys.executable,str(root/"scripts/train_optimized.py"),"--device",args.device,"--output",str(path),
         "--iterations",str(iterations)]
    if resume:cmd += ["--resume",str(resume)]
    else:cmd += ["--envs","4","--positions-per-update","320","--threads","2","--queues","2",
                "--simulations","8","--candidates","4","--width","32","--blocks","1","--batch-size","32",
                "--replay-capacity","1000","--warmup","1","--updates-per-iteration","2"]
    completed=subprocess.run(cmd,cwd=root,capture_output=True,text=True)
    if completed.returncode:raise RuntimeError(completed.stdout+completed.stderr)
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
with tempfile.TemporaryDirectory(prefix="azul-v2-resume-") as temp:
    a=Path(temp)/"direct";b=Path(temp)/"resume"
    run(a,2);run(b,1);run(b,2,b/"latest.pt")
    x=torch.load(a/"latest.pt",weights_only=False,map_location="cpu")
    y=torch.load(b/"latest.pt",weights_only=False,map_location="cpu")
    for key in ("model","optimizer","iteration","updates","games","credit","mean_length","torch_rng","cuda_rng","replay_metadata"):
        assert equal(x[key],y[key]),key
    rx=DeviceReplay(1000,"cpu");ry=DeviceReplay(1000,"cpu")
    CheckpointJournal.restore(a/"latest.pt",rx,x);CheckpointJournal.restore(b/"latest.pt",ry,y)
    for name in (*rx.fields,"seen"):
        assert torch.equal(getattr(rx,name)[:rx.size],getattr(ry,name)[:ry.size]),name
print(f"Optimized exact resume passed on {args.device}: model, optimizer, journal, replay metadata and RNG")
