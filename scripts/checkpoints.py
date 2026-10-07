"""Manage v2 checkpoint storage without training or changing network weights."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from azul_ai.checkpoint_storage import retention,compact_existing,atomic_json,checked_path
p=argparse.ArgumentParser()
p.add_argument("directory",type=Path)
p.add_argument("--compact",action="store_true",help="Losslessly compress live replay files, update latest, reclaim old raw files")
p.add_argument("--prune",action="store_true",help="Apply bounded actor retention")
p.add_argument("--keep-recent",type=int,default=2);p.add_argument("--keep-anchors",type=int,default=4)
p.add_argument("--pin",type=int,nargs="*",default=[]);p.add_argument("--unpin",type=int,nargs="*",default=[])
args=p.parse_args();root=args.directory.resolve()
if not root.is_dir():raise FileNotFoundError(root)
if args.pin or args.unpin:
    path=root/"pinned_checkpoints.json"
    pins=set(json.loads(path.read_text())["iterations"]) if path.exists() else set()
    for i in args.pin:
        if i<1 or not checked_path(root,f"actor-{i:06d}.pt").exists():raise ValueError(f"Cannot pin missing actor {i}")
        pins.add(i)
    pins.difference_update(args.unpin);atomic_json(path,{"iterations":sorted(pins)})
report=compact_existing(root,args.keep_recent,args.keep_anchors) if args.compact else retention(root,args.keep_recent,args.keep_anchors,args.prune)
print(json.dumps(report,ensure_ascii=False,indent=2))
