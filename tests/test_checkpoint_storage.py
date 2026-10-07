import json,tempfile,sys
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from azul_ai.checkpoint_storage import save_replay,load_replay,retention
with tempfile.TemporaryDirectory() as temp:
    root=Path(temp); payload={"x":torch.arange(1000),"a":np.arange(300,dtype=np.float16)}
    save_replay(root/"replay.pt.gz",payload); restored=load_replay(root/"replay.pt.gz")
    assert torch.equal(payload["x"],restored["x"]) and np.array_equal(payload["a"],restored["a"])
    for i in range(1,20):torch.save({"iteration":i},root/f"actor-{i:06d}.pt")
    (root/"pinned_checkpoints.json").write_text(json.dumps({"iterations":[3]}))
    inv=retention(root,keep_recent=2,keep_anchors=3)
    assert (root/"actor-000003.pt").exists() and (root/"actor-000019.pt").exists()
    assert (root/"actor-000018.pt").exists()
    assert not (root/"actor-000017.pt").exists()
print("Checkpoint storage tests passed: lossless gzip, bounded retention, pins")
