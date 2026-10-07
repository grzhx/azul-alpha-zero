# Two-Player Historical Opponent Pool

Historical opponents are frozen checkpoints from the learner's own training.
They do not introduce human records or external-engine supervision.

## Behavior

- Current-model self-play is mixed with learner-versus-history games.
- `--history-fraction` controls the requested fraction of learner policy positions
  generated against historical opponents, not the raw fraction of all games.
- Only learner moves generate policy targets in historical games. Final outcomes
  and eligible afterstates provide value supervision.
- Learner seats and starting players are balanced. One historical opponent is
  selected per publication with a smooth weighted schedule to retain batch size.
- The pool records generation, architecture, engine version, and SHA256. Exact
  resume rejects changed or missing opponents.

## Enable a Pool

First retain the desired existing actors:

```sh
python scripts/checkpoints.py runs/new-run --pin 100 200
python scripts/train_optimized.py --resume runs/new-run/latest.pt --retune --iterations 300 --history-fraction 0.4 --history-iterations 100 200 --history-weights 1 2
```

Iteration numbers are examples. They must identify compatible actors actually
present in the run directory, and cannot be newer than the resumed learner.

## Resume and Roll

```sh
python scripts/train_optimized.py --resume runs/new-run/latest.pt --iterations 400
```

Pool settings restore automatically. Roll the pool with explicit `--retune`,
retaining selected actors before their cleanup window closes. Evaluate a new
actor against the current champion and older pool members before promoting it;
a training loss decrease alone does not establish stronger play.

## Recorded Generation-1,800 Pool

The final two-player phase used opponents 1,400/1,500/1,600/1,699 with weights
1:2:3:4. Approximately 40% of learner policy positions came from history games.
See the [README](../README.md#playing-strength) for the recorded model comparison.
These weights are a historical configuration, not a universally optimal schedule.

## Three-Player Difference

`scripts/train_3p.py --history` currently mixes complete frozen-model self-play
cohorts. It does not implement the two-player learner-versus-history mechanism.
See [Three-Player Limitations](TRAINING.md#three-player-limitations).
