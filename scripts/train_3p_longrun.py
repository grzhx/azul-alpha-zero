from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd, log):
    with log.open("a", encoding="utf-8") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)


def checkpoint_iteration(path):
    import torch
    return int(torch.load(path, map_location="cpu", weights_only=False).get("iteration", 0))


def evaluate(a, candidate, opponent, end, log):
    report = a.output / f"eval-stable-{end:06d}.json"
    cmd = [sys.executable, "scripts/evaluate_3p.py", "--candidate", str(candidate), "--opponent", str(opponent), "--pairs", str(a.eval_pairs), "--batch-pairs", str(a.eval_batch), "--simulations", str(a.eval_simulations), "--candidates", str(a.eval_candidates), "--threads", str(a.threads), "--max-depth", str(a.max_depth), "--max-steps", str(a.eval_max_steps), "--output", str(report), "--device", a.device, "--no-graphs"]
    r = run(cmd, log)
    if r.returncode:
        raise SystemExit(r.returncode)
    data = json.loads(report.read_text(encoding="utf-8"))
    print(json.dumps({"event": "stable_eval", "iteration": end, "mean_score": data.get("mean_score"), "ci": data.get("bootstrap_95"), "promotion_eligible": data.get("promotion_eligible")}), flush=True)
    return data


def main(a):
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    log = out / "stable-longrun.log"
    latest = out / "latest.pt"
    if not latest.exists():
        raise SystemExit(f"missing checkpoint: {latest}")
    start = checkpoint_iteration(latest)
    if start < a.start_generation:
        raise SystemExit(f"checkpoint generation {start} is below requested start {a.start_generation}")
    stable_anchor = out / "stable-anchor-0400.pt"
    if not stable_anchor.exists():
        shutil.copy2(latest, stable_anchor)
    # Freeze a small pool. Existing 400 is the start anchor; later block actors are retained.
    history = [stable_anchor]
    for p in sorted(out.glob("actor-3p-*.pt")):
        try:
            it = int(p.stem.split("-")[-1])
        except ValueError:
            continue
        if it >= a.start_generation and it % a.history_stride == 0:
            history.append(p)
    history = sorted({p.resolve(): p for p in history}.values(), key=lambda p: checkpoint_iteration(p))[-a.history_size:]
    while start < a.max_iterations:
        end = min(start + a.block, a.max_iterations)
        # The block-start model is immutable and is the evaluation anchor for this block.
        block_anchor = out / f"stable-anchor-{start:06d}.pt"
        shutil.copy2(latest, block_anchor)
        if block_anchor not in history:
            history.append(block_anchor)
        history = sorted({p.resolve(): p for p in history}.values(), key=lambda p: checkpoint_iteration(p))[-a.history_size:]
        cmd = [sys.executable, "scripts/train_3p.py", "--iterations", str(end), "--resume", str(latest), "--output", str(out), "--replay-dir", str(out / "replay3p"), "--history"] + [str(p) for p in history] + ["--envs", str(a.envs), "--threads", str(a.threads), "--simulations", str(a.simulations), "--candidates", str(a.candidates), "--max-depth", str(a.max_depth), "--max-steps", str(a.max_steps), "--updates-per-iteration", str(a.updates), "--replay-chunks", str(a.replay_chunks), "--checkpoint-every", str(a.block), "--lr", str(a.lr), "--device", a.device]
        r = run(cmd, log)
        if r.returncode:
            raise SystemExit(r.returncode)
        candidate = latest
        data = evaluate(a, candidate, block_anchor, end, log)
        # Keep only latest, current block actor, block anchors, and selected history actors.
        for p in out.glob("actor-3p-*.pt"):
            try:
                it = int(p.stem.split("-")[-1])
            except ValueError:
                continue
            if it != end and it % a.history_stride != 0:
                p.unlink(missing_ok=True)
        if not data.get("promotion_eligible", False):
            (out / "STABLE_STOPPED_NO_RELIABLE_GAIN").write_text(json.dumps(data, indent=2), encoding="utf-8")
            break
        start = end
        history = sorted({p.resolve(): p for p in history if p.exists()}.values(), key=lambda p: checkpoint_iteration(p))[-a.history_size:]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("runs/three_player_maxn_longrun"))
    p.add_argument("--start-generation", type=int, default=400)
    p.add_argument("--max-iterations", type=int, default=10000)
    p.add_argument("--block", type=int, default=100)
    p.add_argument("--history-size", type=int, default=4)
    p.add_argument("--history-stride", type=int, default=100)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--simulations", type=int, default=64)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--max-depth", type=int, default=96)
    p.add_argument("--max-steps", type=int, default=4096)
    p.add_argument("--updates", type=int, default=16)
    p.add_argument("--replay-chunks", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--eval-pairs", type=int, default=300)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--eval-simulations", type=int, default=128)
    p.add_argument("--eval-candidates", type=int, default=16)
    p.add_argument("--eval-max-steps", type=int, default=4096)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    main(p.parse_args())
