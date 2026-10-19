"""Paired-seat self matches with fixed per-tree budget and native-action voting."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
import numpy as np
import torch
from azul import Batch
from azul_ai.evaluate import load_actor, paired_interval
from azul_ai.optimized_search import FrozenActor, PipelineSearch, Tuning
from azul_ai.search import SearchConfig
from azul_ai.search_decision import rank_search_actions
from azul_ai.search_defaults import require_current_search
from azul_ai.selfplay import results_view, split_seed
from azul_ai.train import device_setup
from paired_seat_score import score_pairs


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def summarize(contract, records, seconds, final=False):
    scores = [r["score"] for r in records]
    if len(scores) % 2 or any(s not in (0., .5, 1.) for s in scores):
        raise ValueError("Only complete valid seed pairs may be recorded")
    report = dict(contract, games=len(scores), wins=scores.count(1.),
                  draws=scores.count(.5), losses=scores.count(0.), truncated=0,
                  seconds=seconds, provisional=not final, records=records)
    report["paired_score"] = score_pairs(report)
    report["score_rate"] = float(np.mean(scores)) if scores else None
    if final:
        pair_scores = np.asarray(scores).reshape(-1, 2).mean(1)
        report["ci95"] = paired_interval(pair_scores, contract["seed"])
    return report


def restore_records(path, contract):
    if not path.exists():
        return [], 0.
    saved = json.loads(path.read_text())
    for key, expected in contract.items():
        if saved.get(key) != expected:
            raise ValueError(f"Resume contract mismatch for {key} in {path}")
    records = saved["records"]
    if len(records) % 2 or len(records) > contract["pairs"] * 2:
        raise ValueError("Invalid saved pair count")
    for i, record in enumerate(records):
        if record["pair"] != i // 2 or record["seat"] != i % 2:
            raise ValueError("Saved pairs must form a complete contiguous prefix")
    summarize(contract, records, saved["seconds"])
    return records, saved["seconds"]


def vote_actions(actions, policies, lanes, trees):
    chosen = {}
    for lane in lanes:
        sl = slice(int(lane) * trees, (int(lane) + 1) * trees)
        vote = np.zeros(180, np.int64)
        action, _, _ = rank_search_actions(actions[sl], policies[sl].mean(0), vote, limit=1)
        chosen[int(lane)] = action
    return chosen


def match(args, actor, checkpoint, device, high, low, seed):
    final_path = args.output / f"t{high}-v-t{low}.json"
    partial_path = final_path.with_suffix(".partial.json")
    cfg = SearchConfig(**checkpoint["search"])
    cfg.simulations = args.simulations
    cfg.candidates = min(cfg.candidates, cfg.simulations)
    cfg.gumbel_scale = 0
    tuning = Tuning(**checkpoint["tuning"])
    tuning.subtree_reuse = 0
    contract = dict(candidate=str(args.actor.resolve()), opponent=str(args.actor.resolve()),
        actor_sha256=args.actor_hash, library_sha256=args.library_hash, search_revision=args.revision,
        simulations=[args.simulations, args.simulations], tree_counts=[high, low],
        pairs=args.pairs, batch_games=args.batch_games, threads=args.threads, queues=args.queues,
        max_steps=args.max_steps,
        seed=seed, search=asdict(cfg), tuning=asdict(tuning),
        cuda_graphs=device.type == "cuda", dtype=str(actor.dtype),
        protocol="fresh_independent_trees_native_final_action_vote_tie_by_action_id_v1")
    if final_path.exists():
        records, seconds = restore_records(final_path, contract)
        if len(records) != args.pairs * 2:
            raise ValueError("Final report is incomplete")
        return summarize(contract, records, seconds, final=True)
    records, saved_seconds = restore_records(partial_path, contract)
    start = time.perf_counter()
    last_progress = start
    games = args.batch_games
    emit("stage_start", tree_counts=[high, low], resumed_games=len(records), target_games=args.pairs * 2)
    with ExitStack() as stack:
        real = stack.enter_context(Batch(games, threads=min(args.threads, games), library=args.library))
        require_current_search(real._lib)
        copies, searches = [], []
        for trees in (high, low):
            env = stack.enter_context(Batch(games * trees, threads=args.threads, library=args.library))
            search = stack.enter_context(PipelineSearch(env, actor, cfg, tuning,
                threads=args.threads, queues=args.queues, graphs=device.type == "cuda"))
            copies.append(env)
            searches.append(search)
        views, result = real.numpy_views(), results_view(real)
        seats = np.arange(games) % 2
        for offset in range(len(records) // 2, args.pairs, games // 2):
            count = min(games // 2, args.pairs - offset) * 2
            active = np.arange(games) < count
            winners = np.full(games, -2, np.int8)
            final_scores = np.zeros((games, 2), np.int32)
            for lane in range(games):
                real.reset_at(lane, split_seed(seed, offset + lane // 2), 0)
            for tick in range(args.max_steps):
                real.observe()
                actions = np.full(games, 65535, np.uint16)
                high_turn = views["players"] == seats
                for side, (trees, env, search) in enumerate(zip((high, low), copies, searches)):
                    lane_mask = active & (high_turn if side == 0 else ~high_turn)
                    lanes = np.flatnonzero(lane_mask)
                    if not len(lanes):
                        continue
                    for lane in lanes:
                        snapshot = real.snapshot(int(lane))
                        for tree in range(trees):
                            env.restore(int(lane) * trees + tree, snapshot)
                    native, policies, _ = search.run(
                        split_seed(seed ^ (0x11AA if side == 0 else 0x22BB), offset * args.max_steps + tick),
                        np.repeat(lane_mask, trees))
                    # Confirm every active tree completed its full specified budget.
                    expected = len(lanes) * trees * args.simulations
                    if search.metrics["simulations"] != expected:
                        raise RuntimeError("Incorrect per-tree simulation count")
                    for lane, action in vote_actions(native, policies, lanes, trees).items():
                        actions[lane] = action
                views["actions"][:] = actions
                real.step()
                if result["invalid_action"][active].any():
                    raise RuntimeError("Illegal tournament action")
                done = active & result["terminated"].astype(bool)
                reward = result["reward"]
                absolute = np.where(reward > 0, result["actor"], 1 - result["actor"])
                winners[done] = np.where(reward[done] == 0, -1, absolute[done] == seats[done])
                final_scores[done] = result["scores"][done]
                active[done] = False
                now = time.perf_counter()
                if now - last_progress >= 30:
                    emit("search_progress", tree_counts=[high, low], saved_games=len(records),
                         batch_finished=int((winners[:count] != -2).sum()), tick=tick + 1,
                         seconds=saved_seconds + now - start)
                    last_progress = now
                if not active.any():
                    break
            if active.any():
                raise RuntimeError("Truncated batch; previous complete pairs remain saved")
            records.extend(dict(pair=offset + lane // 2, seat=int(seats[lane]),
                score=.5 if winners[lane] == -1 else float(winners[lane]),
                scores=final_scores[lane].tolist()) for lane in range(count))
            report = summarize(contract, records, saved_seconds + time.perf_counter() - start)
            save_json(partial_path, report)
            emit("batch_complete", tree_counts=[high, low], games=report["games"],
                 wins=report["wins"], draws=report["draws"], losses=report["losses"],
                 score_rate=report["score_rate"], seconds=report["seconds"],
                 filtered=[report["paired_score"]["a_points"], report["paired_score"]["b_points"]])
    report = summarize(contract, records, saved_seconds + time.perf_counter() - start, final=True)
    save_json(final_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, default=ROOT / "runs/value_calibrated_from_1900/actor-002100.pt")
    parser.add_argument("--simulations", type=int, default=1024)
    parser.add_argument("--tree-counts", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--seed", type=int, default=21100408)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--batch-games", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--queues", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/2100-tree-counts")
    parser.add_argument("--library", type=Path, default=ROOT / "build-ranked-search/azul.dll")
    args = parser.parse_args()
    if min(args.pairs, args.threads, args.queues, args.max_steps) < 1 or args.simulations < 2:
        parser.error("Positive counts and simulations >= 2 required")
    if args.batch_games < 2 or args.batch_games % 2:
        parser.error("batch-games must be even and >= 2")
    if len(args.tree_counts) < 2 or sorted(set(args.tree_counts)) != args.tree_counts or args.tree_counts[0] < 1:
        parser.error("tree-counts must be distinct positive counts in increasing order")
    args.output.mkdir(parents=True, exist_ok=True)
    args.actor_hash = hashlib.sha256(args.actor.read_bytes()).hexdigest()
    args.library_hash = hashlib.sha256(args.library.read_bytes()).hexdigest()
    device = device_setup(args.device, 2)
    with Batch(1, library=args.library) as probe:
        args.revision = require_current_search(probe._lib)
    checkpoint = torch.load(args.actor, map_location="cpu", weights_only=False)
    actor = FrozenActor(load_actor(args.actor, device), device)
    stages = []
    comparisons = [(high, low) for low, high in zip(args.tree_counts, args.tree_counts[1:])]
    for i, (high, low) in enumerate(comparisons):
        report = match(args, actor, checkpoint, device, high, low, args.seed + i * 100003)
        stages.append({k: v for k, v in report.items() if k != "records"})
        save_json(args.output / "summary.json", dict(status="complete" if i == len(comparisons)-1 else "running", stages=stages))
        emit("stage_complete", **stages[-1])


if __name__ == "__main__":
    main()
