"""End-to-end value restart: frozen old opponents, mixed budgets, holdout and resume."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from azul_ai.model import PolicyValue
from azul_ai import OPTIMIZED_VERSION
from azul_ai.optimized_learning import DeviceReplay,CheckpointJournal


with tempfile.TemporaryDirectory(prefix="azul-value-training-") as tmp:
    root=Path(tmp);out=root/"new";out.mkdir()
    model=PolicyValue(32,1,kind="equivariant",end_features=True)
    source=root/"source.pt"
    ck=dict(engine_version=OPTIMIZED_VERSION,architecture=model.architecture,model=model.state_dict(),iteration=8)
    torch.save(ck,source);torch.save(ck,out/"actor-000008.pt")
    (out/"pinned_checkpoints.json").write_text(json.dumps({"iterations":[8]}))
    common=["--device","cpu","--output",str(out),"--iterations","10"]
    args=["--init-from",str(source),"--end-features","--value-tower","--value-tower-blocks","1",
          "--value-rank-weight",".1","--value-label-smoothing",".01","--width","32","--blocks","1",
          "--envs","8","--threads","2","--queues","1","--simulations","4","--high-simulations","8",
          "--high-budget-period","2","--positions-per-update","200","--batch-size","32",
          "--replay-capacity","2000","--warmup","1","--updates-per-iteration","2",
          "--history-fraction",".4","--history-iterations","8","--validation-games","1","--validation-every","1"]
    def run(options):
        r=subprocess.run([sys.executable,str(ROOT/"scripts/train_optimized.py"),*options],cwd=ROOT,capture_output=True,text=True)
        assert r.returncode==0,r.stdout+r.stderr
    run(common+args)
    rows=[json.loads(x) for x in (out/"metrics.jsonl").read_text().splitlines()]
    mixed=[r for r in rows if r["event"]=="selfplay_mixed"]
    assert [r["selfplay_simulations"] for r in mixed]==[4,8]
    assert [r["high_budget"] for r in mixed]==[False,True]
    assert all(r["truncated"]==0 for r in mixed)
    validation=[r for r in rows if r["event"]=="value_validation"]
    assert len(validation)==3
    checkpoint=torch.load(out/"latest.pt",weights_only=False)
    assert checkpoint["iteration"]==10 and checkpoint["updates"]==4
    assert checkpoint["phase_history"][0]["after_iteration"]==8
    assert checkpoint["architecture"]["value_tower"]
    assert checkpoint["args"]["high_simulations"]==8
    initial=torch.load(out/"initial.pt",weights_only=False)
    for key,value in model.state_dict().items():assert torch.equal(value,initial["model"][key]),key
    replay=DeviceReplay(2000,"cpu");CheckpointJournal.restore(out/"latest.pt",replay,checkpoint)
    with __import__('numpy').load(out/"value-validation.npz") as data:
        held=data["field0"]
    assert not any((replay.obs[:replay.size].numpy()==x).all(1).any() for x in held)
    validation_bytes=(out/"value-validation.npz").read_bytes()
    run(["--resume",str(out/"latest.pt"),"--iterations","11","--device","cpu"])
    final=torch.load(out/"latest.pt",weights_only=False)
    assert final["iteration"]==11 and final["updates"]==6
    assert final["args"]["value_tower"] and final["args"]["high_simulations"]==8
    assert validation_bytes==(out/"value-validation.npz").read_bytes()
    # Raising the base to the saved high budget must reuse the base search set.
    run(["--resume",str(out/"latest.pt"),"--retune","--iterations","12",
         "--device","cpu","--simulations","8"])
    rows=[json.loads(x) for x in (out/"metrics.jsonl").read_text().splitlines()]
    equal_budget=[r for r in rows if r["event"]=="selfplay_mixed" and r["iteration"]==12][0]
    assert equal_budget["selfplay_simulations"]==8 and not equal_budget["high_budget"]
    # A frozen current-line actor can join the old history pool on retune.
    (out/"pinned_checkpoints.json").write_text(json.dumps({"iterations":[8,11]}))
    run(["--resume",str(out/"latest.pt"),"--retune","--iterations","13",
         "--device","cpu","--high-simulations","0","--high-budget-period","0",
         "--chance-cap","256",
         "--policy-loss-weight","1.2","--value-loss-weight","0.8",
         "--history-iterations","8","11","--history-weights","1","2"])
    final=torch.load(out/"latest.pt",weights_only=False)
    assert final["iteration"]==13 and final["updates"]==10
    assert final["args"]["high_simulations"]==0 and final["args"]["high_budget_period"]==0
    assert final["search"]["chance_cap"]==256
    assert final["args"]["policy_loss_weight"]==1.2 and final["args"]["value_loss_weight"]==.8
    assert [p["iteration"] for p in final["history_pool"]]==[8,11]
    assert (out/"actor-000011.pt").exists()
    assert validation_bytes==(out/"value-validation.npz").read_bytes()
    train_rows=[r for r in [json.loads(x) for x in (out/"metrics.jsonl").read_text().splitlines()]
                if r["event"]=="train"]
    assert "policy_target_kl" in train_rows[-1] and "policy_value_grad_cosine" in train_rows[-1]
    run(["--resume",str(out/"latest.pt"),"--iterations","14","--device","cpu"])
    final=torch.load(out/"latest.pt",weights_only=False)
    assert final["args"]["policy_loss_weight"]==1.2 and final["args"]["value_loss_weight"]==.8
print("Value restart passed: init chronology, old history, mixed budgets, held-out games and exact protocol resume")
