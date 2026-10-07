"""Switch a normal run to history play, then resume without restating pool flags."""
import json,subprocess,sys,tempfile
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]

def run(args,success=True):
    cmd=[sys.executable,str(ROOT/"scripts/train_optimized.py"),*args,"--device","cpu"]
    completed=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    if (completed.returncode==0)!=success:raise AssertionError(completed.stdout+completed.stderr)

with tempfile.TemporaryDirectory(prefix="azul-history-") as temp:
    output=Path(temp)/"run"
    base=["--output",str(output),"--envs","8","--threads","2","--queues","1",
          "--positions-per-update","480","--simulations","8","--candidates","4",
          "--width","32","--blocks","1","--batch-size","32","--replay-capacity","3000",
          "--warmup","1","--updates-per-iteration","2"]
    run([*base,"--iterations","1"])
    latest=output/"latest.pt"
    run(["--resume",str(latest),"--retune","--iterations","2","--history-fraction","0.4",
         "--history-iterations","1","--history-weights","1"])
    state=torch.load(latest,map_location="cpu",weights_only=False)
    assert state["iteration"]==2 and state["args"]["history_fraction"]==0.4
    assert state["history_pool"][0]["iteration"]==1 and len(state["history_pool"][0]["sha256"])==64
    assert state["history_schedule"]=="smooth_weighted_single"
    assert any("history_fraction" in x["changes"] for x in state["phase_history"])
    rows=[json.loads(x) for x in (output/"metrics.jsonl").read_text().splitlines()]
    mixed=[x for x in rows if x.get("event")=="selfplay_mixed"][-1]
    history=[x for x in rows if x.get("event")=="historical_selfplay"]
    assert mixed["historical_games"]>0 and 0.25<mixed["realized_history_policy_fraction"]<0.55
    assert history and history[-1]["opponent_iteration"]==1
    run(["--resume",str(latest),"--iterations","3"])
    resumed=torch.load(latest,map_location="cpu",weights_only=False)
    assert resumed["iteration"]==3 and resumed["history_pool"]==state["history_pool"]
    actor=output/"actor-000001.pt";payload=torch.load(actor,map_location="cpu",weights_only=False)
    payload["iteration"]=999;torch.save(payload,actor)
    run(["--resume",str(latest),"--iterations","4"],False)
print("History pool passed: learner-only policy data, recorded mix, exact resume, hash/metadata rejection")
