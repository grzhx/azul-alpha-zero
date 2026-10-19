"""All decision states from full 4096 self-play games, compared at three budgets."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
import numpy as np
import torch
from azul import Batch
from azul_ai.evaluate import load_actor
from azul_ai.model import EndFeatures
from azul_ai.optimized_search import FrozenActor, PipelineSearch, Tuning
from azul_ai.search import SearchConfig
from azul_ai.search_defaults import require_current_search
from azul_ai.selfplay import results_view, split_seed
from azul_ai.train import device_setup
from evaluate_tree_counts import save_json, emit


def config(payload, budget):
    cfg = SearchConfig(**payload["search"])
    cfg.simulations = budget
    cfg.gumbel_scale = 0
    tuning = Tuning(**payload["tuning"])
    tuning.subtree_reuse = 0
    return cfg, tuning


def semantic_actions(obs, actions):
    """Factories with identical contents are equivalent choices; center is fixed."""
    result = []
    for x, action in zip(obs, actions):
        source, color, row = int(action) // 30, (int(action) // 6) % 5, int(action) % 6
        source_key = ("center",) if source == 5 else tuple(np.rint(x[source*5:source*5+5] * 20).astype(int))
        result.append((source_key, color, row))
    return result


def rates(obs, actions, groups):
    result = {}
    semantic = {b: semantic_actions(obs, a) for b, a in actions.items()}
    for a, b in itertools.combinations(sorted(actions), 2):
        raw = actions[a] != actions[b]
        meaningful = np.asarray([x != y for x, y in zip(semantic[a], semantic[b])])
        detail = {}
        for name, mask in groups.items():
            n = int(mask.sum())
            if n:
                detail[name] = dict(positions=n, differing_actions=int(raw[mask].sum()),
                    action_difference_rate=float(raw[mask].mean()),
                    semantically_different=int(meaningful[mask].sum()),
                    semantic_difference_rate=float(meaningful[mask].mean()))
        result[f"{a}_vs_{b}"] = detail
    return result


def generate(args, actor, payload, device):
    cfg, tuning = config(payload, 4096)
    data_path = args.output / "positions.npz"
    if data_path.exists():
        saved = json.loads((args.output / "dataset.json").read_text())
        if saved["contract"] != args.contract:
            raise ValueError("Dataset contract mismatch")
        return saved
    rows = {key: [] for key in ("snapshot", "observation", "tick", "lane", "round", "legal_count", "ending", "action4096")}
    start = last = time.perf_counter()
    finals = {}
    with Batch(args.games, threads=args.threads, library=args.library) as env, PipelineSearch(
            env, actor, cfg, tuning, threads=args.threads, queues=1, graphs=device.type == "cuda") as search:
        require_current_search(env._lib)
        for lane in range(args.games):
            env.reset_at(lane, split_seed(args.seed, lane), lane % 2)
        view, result = env.numpy_views(), results_view(env)
        active = np.ones(args.games, bool)
        for tick in range(4096):
            env.observe()
            ids = np.flatnonzero(active)
            obs = view["observations"][ids].copy()
            snapshots = [np.frombuffer(env.snapshot(int(lane)), np.uint8).copy() for lane in ids]
            with torch.inference_mode():
                ending = EndFeatures()(torch.from_numpy(obs))[:, -1].numpy() > .5
            actions, _, _ = search.run(split_seed(args.seed ^ 0xA35F, tick), active)
            if search.metrics["simulations"] != len(ids) * 4096:
                raise RuntimeError("Incorrect generation budget")
            for index, lane in enumerate(ids):
                rows["snapshot"].append(snapshots[index])
                rows["observation"].append(obs[index])
                rows["tick"].append(tick); rows["lane"].append(int(lane))
                rows["round"].append(int(round(float(obs[index, 170]) * 10)))
                rows["legal_count"].append(int(view["masks"][lane].sum()))
                rows["ending"].append(bool(ending[index]))
                rows["action4096"].append(int(actions[lane]))
            view["actions"][:] = actions
            env.step()
            if result["invalid_action"][active].any():
                raise RuntimeError("Invalid generation action")
            done = active & result["terminated"].astype(bool)
            for lane in np.flatnonzero(done):
                finals[str(lane)] = dict(scores=result["scores"][lane].astype(int).tolist(), moves=tick+1)
            active[done] = False
            if time.perf_counter() - last >= 30:
                emit("generation_progress", completed_games=len(finals), tick=tick+1, positions=len(rows["lane"]))
                last = time.perf_counter()
            if not active.any():
                break
        if active.any():
            raise RuntimeError("Truncated games cannot be included")
    with data_path.open("wb") as stream:
        np.savez_compressed(stream, **{k: np.asarray(v) for k, v in rows.items()})
    meta = dict(contract=args.contract, games=args.games, positions=len(rows["lane"]),
                all_decisions=True, final_games=finals, seconds=time.perf_counter()-start)
    save_json(args.output / "dataset.json", meta)
    emit("dataset_complete", games=args.games, positions=meta["positions"], seconds=meta["seconds"])
    return meta


def evaluate(args, actor, payload, device, data, budget):
    path = args.output / f"choices-{budget}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["contract"] != args.contract or saved["budget"] != budget:
            raise ValueError("Choice result contract mismatch")
        return np.asarray(saved["actions"], np.int32)
    actions = np.full(len(data["lane"]), -1, np.int32)
    start_tick, previous_seconds = 0, 0.
    partial_path = path.with_suffix(".partial.json")
    if partial_path.exists():
        saved = json.loads(partial_path.read_text())
        if saved["contract"] != args.contract or saved["budget"] != budget:
            raise ValueError("Partial contract mismatch")
        actions[:] = saved["actions"]
        start_tick, previous_seconds = saved["next_tick"], saved["seconds"]
    cfg, tuning = config(payload, budget)
    start = time.perf_counter()
    with Batch(args.games, threads=args.threads, library=args.library) as env, PipelineSearch(
            env, actor, cfg, tuning, threads=args.threads, queues=1, graphs=device.type == "cuda") as search:
        view = env.numpy_views()
        for tick in range(start_tick, int(data["tick"].max()) + 1):
            indices = np.flatnonzero(data["tick"] == tick)
            lanes = data["lane"][indices].astype(int)
            active = np.zeros(args.games, bool); active[lanes] = True
            for index, lane in zip(indices, lanes):
                env.restore(int(lane), data["snapshot"][index].tobytes())
            env.observe()
            chosen, _, _ = search.run(split_seed(args.seed ^ 0xA35F, tick), active)
            if search.metrics["simulations"] != len(lanes) * budget:
                raise RuntimeError("Incorrect evaluation budget")
            if not view["masks"][lanes, chosen[lanes]].all():
                raise RuntimeError("Illegal evaluation choice")
            actions[indices] = chosen[lanes]
            save_json(partial_path, dict(contract=args.contract, budget=budget, actions=actions.tolist(),
                next_tick=tick+1, seconds=previous_seconds + time.perf_counter()-start))
            if tick % 8 == 0:
                emit("choice_progress", budget=budget, positions=int((actions >= 0).sum()), total=len(actions))
    if (actions < 0).any():
        raise RuntimeError("Missing evaluated positions")
    save_json(path, dict(contract=args.contract, budget=budget, actions=actions.tolist(),
                        seconds=previous_seconds + time.perf_counter()-start))
    return actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, default=ROOT / "runs/value_calibrated_from_1900/actor-002100.pt")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=21114096)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--library", type=Path, default=ROOT / "build-ranked-search/azul.dll")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/2100-search-choices")
    args = parser.parse_args()
    if args.games < 1 or args.threads < 1:
        parser.error("Positive game/thread counts required")
    args.output.mkdir(parents=True, exist_ok=True)
    device = device_setup(args.device, 2)
    with Batch(1, library=args.library) as env:
        revision = require_current_search(env._lib)
    payload = torch.load(args.actor, map_location="cpu", weights_only=False)
    cfg, tuning = config(payload, 4096)
    args.contract = dict(actor=str(args.actor.resolve()),
        actor_sha256=hashlib.sha256(args.actor.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(), revision=revision,
        games=args.games, threads=args.threads, seed=args.seed, device=str(device),
        search=asdict(cfg), tuning=asdict(tuning), budgets=[4096,10000,30000],
        protocol="all_4096_game_decisions_same_state_same_native_seed_fresh_single_tree")
    actor = FrozenActor(load_actor(args.actor, device), device)
    meta = generate(args, actor, payload, device)
    with np.load(args.output / "positions.npz", allow_pickle=False) as stored:
        data = {k: stored[k] for k in stored.files}
    actions = {4096: data["action4096"]}
    for budget in (10000, 30000):
        actions[budget] = evaluate(args, actor, payload, device, data, budget)
    groups = dict(all=np.ones(meta["positions"],bool),
        multiple_legal_actions=data["legal_count"]>1, committed_final_round=data["ending"].astype(bool))
    final_round = np.zeros(meta["positions"], bool)
    for lane in range(args.games):
        mask = data["lane"] == lane
        final_round[mask] = data["round"][mask] == data["round"][mask].max()
    groups["actual_final_round"] = final_round
    groups.update({f"round_{r}":data["round"]==r for r in np.unique(data["round"])})
    report = dict(contract=args.contract, dataset=meta, comparisons=rates(data["observation"],actions,groups),
        per_game={str(lane):rates(data["observation"],actions,{"all":data["lane"]==lane})
                  for lane in range(args.games)},
        interpretation="Choice differences are not win-rate or strength differences; all states follow the 4096 generating policy.")
    details = []
    semantic = {b: semantic_actions(data["observation"], a) for b, a in actions.items()}
    for i in range(meta["positions"]):
        details.append(dict(position=i, game=int(data["lane"][i]), move=int(data["tick"][i])+1,
            round=int(data["round"][i]), legal_count=int(data["legal_count"][i]),
            committed_ending=bool(data["ending"][i]), actual_final_round=bool(final_round[i]),
            actions={str(b):int(a[i]) for b,a in actions.items()},
            raw_difference={f"{a}_vs_{b}":bool(actions[a][i]!=actions[b][i]) for a,b in itertools.combinations(sorted(actions),2)},
            semantic_difference={f"{a}_vs_{b}":bool(semantic[a][i]!=semantic[b][i]) for a,b in itertools.combinations(sorted(actions),2)}))
    save_json(args.output / "per-position.json", dict(contract=args.contract, positions=details))
    save_json(args.output / "summary.json", report)
    emit("comparison_complete", positions=meta["positions"], comparisons=report["comparisons"])


if __name__ == "__main__":
    main()
