"""Run: python examples/selfplay.py (no third-party Python dependencies).
This illustrates the ABI; Python action selection is not a throughput benchmark.
"""
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from azul import Batch, ACTION_COUNT


def main():
    policy = random.Random(42)
    n = 64
    finished = [False] * n
    wins = [0, 0]
    draws = 0
    with Batch(n, threads=4, seed=42) as env:
        # Snapshot restores RNG too: the first transition replays exactly.
        env.observe()
        snapshots = [env.snapshot(i) for i in range(n)]
        actions = [next(a for a in range(ACTION_COUNT) if env.masks[i*ACTION_COUNT+a]) for i in range(n)]
        env.step(actions)
        after = [env.snapshot(i) for i in range(n)]
        for i in range(n):
            env.restore(i, snapshots[i])
        env.step(actions)
        assert all(env.snapshot(i) == after[i] for i in range(n))
        env.reset(seed=42)
        for tick in range(4096):
            env.observe()
            for i in range(n):
                legal = [a for a in range(ACTION_COUNT) if env.masks[i*ACTION_COUNT+a]]
                env.actions[i] = policy.choice(legal) if legal else 65535
            for i, result in enumerate(env.step()):
                if finished[i]:
                    assert result.terminated and result.invalid_action and result.reward == 0
                    continue
                assert not result.invalid_action
                if result.terminated:
                    finished[i] = True
                    if result.reward == 0:
                        draws += 1
                    else:
                        winner = result.actor if result.reward > 0 else 1-result.actor
                        wins[winner] += 1
            if all(finished):
                print(f"Completed {n} games; wins={wins}, draws={draws}, batch_steps={tick+1}; snapshot replay passed")
                return
        raise RuntimeError("4096-step training cutoff reached; this is truncation, not an Azul draw")


if __name__ == "__main__":
    main()
