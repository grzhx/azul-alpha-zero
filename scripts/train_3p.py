from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from azul3 import ACTION_COUNT, OBSERVATION_SIZE, Batch3
from azul_ai.model3 import PolicyValue3, utility_targets
from azul_ai.search3 import Evaluator3, Search3, SearchConfig3


def split_seed(base, i):
    x = (int(base) + 0x9E3779B97F4A7C15 * (int(i) + 1)) & ((1 << 64) - 1)
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return (x ^ (x >> 31)) & ((1 << 64) - 1)


def device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name if name != "auto" else "cpu")


def load_model(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PolicyValue3(width=ck["architecture"]["width"], blocks=ck["architecture"]["blocks"])
    m.load_state_dict(ck["model"])
    return m.to(dev).eval()


def load_or_init(args, dev):
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        m = PolicyValue3(**{k: ck["architecture"][k] for k in ("width", "blocks")})
        m.load_state_dict(ck["model"])
        return m.to(dev), int(ck.get("iteration", 0)), ck
    if args.base:
        return PolicyValue3.from_1800(args.base, dev), 0, None
    return PolicyValue3(args.width, args.blocks).to(dev), 0, None


class ReplayStore:
    """Persistent, bounded replay chunks. Samples are stored as fp16 to control disk use."""

    def __init__(self, root, capacity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = max(1, int(capacity))
        self.chunks = []
        self.refresh()

    def refresh(self):
        all_chunks = sorted(self.root.glob("chunk-*.npz"), key=lambda p: int(p.stem.split("-")[-1]))
        self.chunks = all_chunks[-self.capacity:]
        for old in all_chunks[:-self.capacity]:
            old.unlink(missing_ok=True)

    @staticmethod
    def _save(path, data):
        o, p, m, u = data
        np.savez_compressed(path, observations=o.astype(np.float16), policy=p.astype(np.float16), masks=m.astype(np.uint8), utilities=u.astype(np.float16))

    @staticmethod
    def _load(path):
        z = np.load(path)
        return (z["observations"].astype(np.float32), z["policy"].astype(np.float32), z["masks"].astype(bool), z["utilities"].astype(np.float32))

    def add(self, iteration, data):
        path = self.root / f"chunk-{int(iteration):06d}.npz"
        tmp = path.with_suffix(".tmp.npz")
        self._save(tmp, data)
        tmp.replace(path)
        self.refresh()
        self.refresh()

    def load_all(self):
        self.refresh()
        if not self.chunks:
            return None
        rows = [self._load(p) for p in self.chunks]
        return tuple(np.concatenate([x[i] for x in rows]) for i in range(4))

    def manifest(self):
        self.refresh()
        return {"capacity": self.capacity, "chunks": [p.name for p in self.chunks], "positions": int(sum(np.load(p)["observations"].shape[0] for p in self.chunks))}


def collect(env, search, actor, seed, games, max_steps, dev, no_graphs):
    ev = Evaluator3(actor, env.n, dev, not no_graphs)
    env.reset(seed)
    views = env.numpy_views()
    active = np.ones(env.n, bool)
    active[games:] = False
    obs, pol, mask, actor_ids, lanes = [], [], [], [], []
    scores = np.zeros((env.n, 3), np.uint16)
    steps = 0
    start = time.perf_counter()
    for tick in range(max_steps):
        env.observe()
        ids = np.flatnonzero(active)
        obs.append(views["observations"][ids].copy())
        mask.append(views["masks"][ids].copy())
        actor_ids.append(views["players"][ids].copy())
        lanes.append(ids)
        actions, policy, _ = search.run(ev, split_seed(seed, tick))
        pol.append(policy[ids].copy())
        views["actions"][:] = actions
        env.step()
        bad = np.fromiter((r.invalid_action for r in env.results), dtype=np.uint8, count=env.n)
        if bad[active].any():
            raise RuntimeError("3P invalid action")
        for i in ids:
            scores[i] = env.results[i].scores
        term = np.array([bool(r.terminated) for r in env.results])
        active[term] = False
        steps += len(ids)
        if not active.any():
            break
    if active.any():
        return None, {"games": int((~active).sum()), "truncated": int(active.sum())}
    o = np.concatenate(obs)
    p = np.concatenate(pol)
    m = np.concatenate(mask)
    a = np.concatenate(actor_ids)
    lane = np.concatenate(lanes)
    targets = utility_targets(torch.from_numpy(scores[lane].astype(np.float32))).numpy()
    rotated = np.stack([np.roll(targets[i], -int(a[i])) for i in range(len(a))]).astype(np.float32)
    return (o.astype(np.float32), p.astype(np.float32), m, rotated), {"games": games, "truncated": 0, "positions": len(o), "steps": steps, "seconds": time.perf_counter() - start}


def validation_loss(model, data, dev, batch_size):
    if data is None:
        return None, None
    o, p, m, u = data
    losses = []
    vlosses = []
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(o), batch_size):
            sl = slice(begin, min(begin + batch_size, len(o)))
            logits, val = model(torch.from_numpy(o[sl]).to(dev))
            logits = logits.float().masked_fill(~torch.from_numpy(m[sl]).to(dev).bool(), -1e9)
            losses.append(float((-(torch.from_numpy(p[sl]).to(dev) * torch.log_softmax(logits, -1)).sum(-1)).mean()))
            vlosses.append(float(torch.nn.functional.mse_loss(val.float(), torch.from_numpy(u[sl]).to(dev).float())))
    return float(np.mean(losses)), float(np.mean(vlosses))


def train(args):
    dev = device(args.device)
    model, start, old_ck = load_or_init(args, dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.resume:
        opt.load_state_dict(old_ck["optimizer"])
        for group in opt.param_groups:
            group["lr"] = args.lr
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    replay = ReplayStore(args.replay_dir or out / "replay3p", args.replay_chunks)
    log = out / "metrics-stable.jsonl"
    rng = np.random.default_rng(args.seed + start)
    env = Batch3(args.envs, args.threads, args.seed)
    search = Search3(env, SearchConfig3(simulations=args.simulations, candidates=min(args.candidates, args.simulations), max_depth=args.max_depth), args.threads)
    history_paths = [Path(x) for x in args.history if Path(x).exists()]
    history_models = [load_model(p, dev) for p in history_paths]
    validation_path = out / "validation3p.npz"
    validation = None
    if validation_path.exists():
        validation = ReplayStore._load(validation_path)
    total_games = int(old_ck.get("games", 0)) if old_ck else 0
    all_replay = replay.load_all()
    if all_replay is None and old_ck and old_ck.get("replay_manifest"):
        print(json.dumps({"event": "replay_restore", "status": "manifest_missing_chunks", "manifest": old_ck["replay_manifest"]}), flush=True)
    pool_weights = [float(args.current_weight)] + [float(args.history_weight)] * len(history_models)
    pool_weights = np.asarray(pool_weights, np.float64)
    pool_weights /= pool_weights.sum() if pool_weights.sum() else 1.0
    for it in range(start, args.iterations):
        choice = int(rng.choice(len(history_models) + 1, p=pool_weights))
        actor = model if choice == 0 else history_models[choice - 1]
        actor.eval()
        data, met = collect(env, search, actor, split_seed(args.seed, it), args.envs, args.max_steps, dev, args.no_graphs)
        if data is None:
            raise RuntimeError(f"truncated 3P cohort at iteration {it + 1}: {met}")
        replay.add(it + 1, data)
        if validation is None:
            nval = min(args.validation_size, len(data[0]))
            validation = tuple(x[:nval].copy() for x in data)
            ReplayStore._save(validation_path, validation)
        all_replay = replay.load_all()
        cat = all_replay
        n = len(cat[0])
        batch = min(args.batch_size, n)
        losses = []
        model.train()
        for _ in range(max(1, args.updates_per_iteration)):
            idx = rng.integers(n, size=batch)
            x = torch.from_numpy(cat[0][idx]).to(dev)
            t = torch.from_numpy(cat[1][idx]).to(dev)
            ms = torch.from_numpy(cat[2][idx]).to(dev)
            u = torch.from_numpy(cat[3][idx]).to(dev)
            logits, val = model(x)
            logits = logits.float().masked_fill(~ms.bool(), -1e9)
            pl = -(t.float() * torch.log_softmax(logits, -1)).sum(-1).mean()
            vl = torch.nn.functional.mse_loss(val.float(), u.float())
            loss = pl + vl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append((float(loss.detach()), float(pl.detach()), float(vl.detach())))
        val_pl, val_vl = validation_loss(model, validation, dev, batch)
        row = {"event": "train_3p_stable", "iteration": it + 1, "games": total_games + met["games"], "positions": n, "loss": float(np.mean([x[0] for x in losses])), "policy_loss": float(np.mean([x[1] for x in losses])), "value_loss": float(np.mean([x[2] for x in losses])), "validation_policy_loss": val_pl, "validation_value_loss": val_vl, "actor": "current" if choice == 0 else str(history_paths[choice - 1]), "replay": replay.manifest(), "history_pool": [str(x) for x in history_paths], "seconds": met["seconds"], "device": str(dev)}
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        total_games += met["games"]
        if (it + 1) % args.checkpoint_every == 0 or it + 1 == args.iterations:
            ck = {"engine_version": "azul3-stable-v2", "architecture": model.architecture, "model": model.state_dict(), "optimizer": opt.state_dict(), "iteration": it + 1, "games": total_games, "args": vars(args), "replay_manifest": replay.manifest(), "validation": validation_path.name, "history_pool": [str(x) for x in history_paths]}
            torch.save(ck, out / "latest.pt")
            torch.save(ck, out / f"actor-3p-{it + 1:06d}.pt")
    search.close()
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("runs/optimized/actor-001800.pt"))
    p.add_argument("--resume", type=Path)
    p.add_argument("--output", type=Path, default=Path("runs/three_player"))
    p.add_argument("--replay-dir", type=Path)
    p.add_argument("--history", type=Path, nargs="*", default=[])
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--simulations", type=int, default=128)
    p.add_argument("--candidates", type=int, default=32)
    p.add_argument("--max-depth", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--updates-per-iteration", type=int, default=16)
    p.add_argument("--replay-chunks", type=int, default=4)
    p.add_argument("--checkpoint-every", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--current-weight", type=float, default=0.6)
    p.add_argument("--history-weight", type=float, default=0.2)
    p.add_argument("--validation-size", type=int, default=8192)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--seed", type=int, default=18003)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--no-graphs", action="store_true")
    train(p.parse_args())
