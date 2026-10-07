import time
import numpy as np

RESULT_DTYPE = np.dtype([("reward", "<f4"), ("scores", "<u2", (2,)),
                         ("actor", "u1"), ("terminated", "u1"),
                         ("round_finished", "u1"), ("invalid_action", "u1")])


def results_view(batch):
    return np.frombuffer(batch.results, dtype=RESULT_DTYPE, count=batch.n)


def split_seed(base, episode):
    """Exact unsigned-64 counterpart of C++ episode_seed."""
    mask = (1 << 64) - 1
    z = (int(base) + 0x9E3779B97F4A7C15 * (int(episode) + 1)) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def collect(batch, search, evaluator, seed, max_steps=4096, progress=None):
    """One wave, one game per lane, weights frozen throughout the wave.

    Completed lanes keep terminal state; no partial-game labels enter replay.
    A wave boundary is also a coherent checkpoint / network publication point.
    """
    batch.reset(seed)
    n = batch.n
    views = batch.numpy_views()
    results = results_view(batch)
    active = np.ones(n, dtype=bool)
    winners = np.full(n, -2, dtype=np.int8)
    obs_rows, policy_rows, mask_rows, players_rows, lanes_rows = [], [], [], [], []
    stats = {k: 0 for k, _ in search.stats._fields_}
    decisions, start, last_log = 0, time.perf_counter(), time.perf_counter()
    for tick in range(max_steps):
        batch.observe()
        obs = views["observations"][active].copy()
        masks = views["masks"][active].copy()
        players = views["players"][active].copy()
        lane_ids = np.flatnonzero(active)
        actions, policies, _ = search.run(evaluator, split_seed(seed ^ 0xA0761D6478BD642F, tick))
        obs_rows.append(obs.astype(np.float16))
        policy_rows.append(policies[active].astype(np.float16))
        mask_rows.append(masks)
        players_rows.append(players)
        lanes_rows.append(lane_ids)
        np.copyto(views["actions"], actions)
        batch.step()
        if results["invalid_action"][active].any():
            raise RuntimeError("Search selected an illegal move")
        done = active & results["terminated"].astype(bool)
        reward, actor = results["reward"], results["actor"]
        winners[done & (reward == 0)] = -1
        wins = np.where(reward > 0, actor, 1 - actor)
        winners[done & (reward != 0)] = wins[done & (reward != 0)]
        decisions += int(active.sum())
        active[done] = False
        for key in stats:
            stats[key] += getattr(search.stats, key)
        now = time.perf_counter()
        if progress and now - last_log >= 30:
            progress({"event": "selfplay_progress", "batch_tick": tick+1,
                      "finished": int((~active).sum()), "decisions": decisions,
                      "decisions_per_s": decisions/(now-start)})
            last_log = now
        if not active.any():
            break
    obs = np.concatenate(obs_rows)
    policy = np.concatenate(policy_rows)
    masks = np.concatenate(mask_rows)
    actors = np.concatenate(players_rows)
    lanes = np.concatenate(lanes_rows)
    valid = winners[lanes] != -2
    outcome = np.where(winners[lanes] == -1, 1, np.where(winners[lanes] == actors, 0, 2)).astype(np.uint8)
    seconds = time.perf_counter() - start
    metrics = dict(event="selfplay", games=int((~active).sum()), truncated=int(active.sum()),
                   wins0=int((winners==0).sum()), wins1=int((winners==1).sum()), draws=int((winners==-1).sum()),
                   decisions=decisions, positions=int(valid.sum()), seconds=seconds,
                   decisions_per_s=decisions/seconds, **stats)
    return (obs[valid], policy[valid], masks[valid], outcome[valid]), metrics
