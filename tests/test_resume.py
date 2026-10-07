"""End-to-end exact CPU resume, including RNG, optimizer and replay ring state."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def run(path,iterations,resume=None):
    cmd=[sys.executable,str(ROOT/"scripts/train.py"),"--device","cpu","--iterations",str(iterations),
         "--envs","4","--threads","2","--cpu-threads","1","--simulations","8","--candidates","4",
         "--width","32","--blocks","1","--batch-size","32","--warmup","1","--replay-capacity","1000",
         "--updates-per-iteration","2","--output",str(path)]
    if resume:cmd += ["--resume",str(resume)]
    subprocess.run(cmd,cwd=ROOT,check=True,capture_output=True,text=True)

def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

with tempfile.TemporaryDirectory(prefix="azul-resume-") as temp:
    direct=Path(temp)/"direct";resumed=Path(temp)/"resumed"
    run(direct,2);run(resumed,1);run(resumed,2,resumed/"latest.pt")
    a=torch.load(direct/"latest.pt",weights_only=False,map_location="cpu")
    b=torch.load(resumed/"latest.pt",weights_only=False,map_location="cpu")
    for key in ("model","optimizer","replay","torch_rng","numpy_rng","iteration","updates","games","credit"):
        assert equal(a[key],b[key]),key
print("Exact resume passed: model, optimizer, replay, RNGs, counters match uninterrupted training")
