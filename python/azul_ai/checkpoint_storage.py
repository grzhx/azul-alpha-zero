"""Bounded actor retention and lossless replay compression.

Only known generated files in a run directory are reclaimed, after committing
the new manifest. Initial weights, latest resume state and pinned actors survive.
"""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import torch
import numpy as np
from .train import save_checkpoint


def checked_path(root,name):
    root=Path(root).resolve();p=root/name
    if Path(name).name!=name or p.resolve().parent!=root:
        raise ValueError("Checkpoint path must be a plain filename inside the run")
    return p


def save_replay(path,payload):
    path=Path(path)
    if path.suffix!=".gz":return save_checkpoint(path,payload)
    tmp=path.with_name(path.name+".tmp")
    try:
        with open(tmp,"wb") as f:
            with gzip.GzipFile(fileobj=f,mode="wb",compresslevel=1,mtime=0) as stream:
                if isinstance(payload,dict) and "data" in payload and isinstance(payload["data"],tuple):
                    # Serialize one array at a time so a 100k-position delta
                    # never requires a second full tuple-sized Python pickle.
                    stream.write(b"AZUL_DELTA_NPY_V1\n")
                    header={"version":payload["version"],"fields":len(payload["data"]),
                            "shapes":[tuple(x.shape) for x in payload["data"]],
                            "dtypes":[str(x.dtype) for x in payload["data"]]}
                    stream.write((json.dumps(header,separators=(",",":"))+"\n").encode("utf-8"))
                    for field in payload["data"]:np.save(stream,np.asarray(field),allow_pickle=False)
                else:torch.save(payload,stream)
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()


def load_replay(path):
    path=Path(path)
    if path.suffix==".gz":
        # Decompress once: repeated random seeks into a gzip stream are expensive.
        with gzip.open(path,"rb") as f:
            if f.read(len(b"AZUL_DELTA_NPY_V1\n"))==b"AZUL_DELTA_NPY_V1\n":
                header=json.loads(f.readline().decode("utf-8"))
                data=tuple(np.load(f,allow_pickle=False) for _ in range(header["fields"]))
                return {"version":header["version"],"data":data}
            f.seek(0);payload=io.BytesIO(f.read())
        return torch.load(payload,map_location="cpu",weights_only=False)
    return torch.load(path,map_location="cpu",weights_only=False)


def atomic_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(tmp,path)


def retention(root,keep_recent=2,keep_anchors=4,prune=True):
    root=Path(root).resolve()
    if keep_recent<1 or keep_anchors<1:raise ValueError("Keep at least one recent actor and one historical anchor")
    pins_path=root/"pinned_checkpoints.json"
    pins=set(json.loads(pins_path.read_text(encoding="utf-8"))["iterations"]) if pins_path.exists() else set()
    if any(not isinstance(i,int) or i<1 for i in pins):raise ValueError("Invalid pinned checkpoint iteration")
    actors={}
    for p in root.glob("actor-*.pt"):
        match=re.fullmatch(r"actor-(\d+)\.pt",p.name)
        if match and p.is_file():actors[int(match[1])]=checked_path(root,p.name)
    iterations=sorted(actors)
    recent=set(iterations[-keep_recent:])
    # Online-stable anchors: powers of two. Arbitrary 'closest to N/2' snapshots
    # may already have been discarded, whereas powers of two can be retained online.
    powers=[i for i in iterations if i>0 and i&(i-1)==0]
    anchors=set(powers[-keep_anchors:])
    keep=recent|anchors|pins
    removed=[]
    for i,p in actors.items():
        if i not in keep and prune:
            size=p.stat().st_size;p.unlink();removed.append(dict(iteration=i,bytes=size))
    kept=[]
    for i in iterations:
        if i in keep or not prune:
            kept.append(dict(iteration=i,file=actors[i].name,bytes=actors[i].stat().st_size,
                roles=[r for r,ids in (("recent",recent),("anchor",anchors),("pinned",pins)) if i in ids]))
    inventory=dict(keep_recent=keep_recent,keep_anchors=keep_anchors,pins=sorted(pins),actors=kept,
        missing_pinned=[i for i in pins if i not in actors],initial_retained=(root/"initial.pt").exists(),
        resume_checkpoint="latest.pt",removed=removed,
        directory_bytes=sum(p.stat().st_size for p in root.iterdir() if p.is_file() and not p.is_symlink()))
    if prune:atomic_json(root/"checkpoint_inventory.json",inventory)
    return inventory


def compact_existing(root,keep_recent=2,keep_anchors=4):
    root=Path(root).resolve();latest=root/"latest.pt"
    state=torch.load(latest,map_location="cpu",weights_only=False)
    if "replay_manifest" not in state:raise ValueError("This command manages v2 journal runs, not legacy v1 checkpoints")
    manifest=state["replay_manifest"];old=[manifest["base"],*manifest["deltas"]];mapping={}
    before=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    for name in old:
        source=checked_path(root,name)
        if name.endswith(".gz"):
            if not source.exists():raise FileNotFoundError(source)
            mapping[name]=name;continue
        target=checked_path(root,name+".gz");tmp=target.with_name(target.name+".tmp")
        try:
            with source.open("rb") as src, tmp.open("wb") as dst:
                with gzip.GzipFile(fileobj=dst,mode="wb",compresslevel=1,mtime=0) as compressed:
                    shutil.copyfileobj(src,compressed,length=1024*1024)
            with source.open("rb") as f:original_hash=hashlib.file_digest(f,"sha256").digest()
            with gzip.open(tmp,"rb") as f:restored_hash=hashlib.file_digest(f,"sha256").digest()
            if original_hash!=restored_hash:raise IOError("Lossless compression verification failed")
            os.replace(tmp,target)
        finally:
            if tmp.exists():tmp.unlink()
        mapping[name]=target.name
    state["replay_manifest"]={"base":mapping[manifest["base"]],"deltas":[mapping[x] for x in manifest["deltas"]]}
    state["args"]["replay_compression"]="gzip"
    state["args"]["keep_recent"]=keep_recent;state["args"]["keep_anchors"]=keep_anchors
    save_checkpoint(latest,state)
    # New latest now references only verified files; old raw replays can be reclaimed.
    for name in old:
        if mapping[name]!=name:checked_path(root,name).unlink()
    report=retention(root,keep_recent,keep_anchors)
    report["before_bytes"]=before
    report["after_bytes"]=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
    return report
