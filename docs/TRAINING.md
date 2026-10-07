# Training and Evaluation

All training targets are generated from self-play. There are no human game
records, external engine labels, or manually designed move rewards. Original
Azul's fixed colored wall is the only supported rules variant.

## Prerequisites

Build the native library as described in the [README](../README.md#build). Install
NumPy and a compatible PyTorch build in an activated virtual environment. CPU is
available through `--device cpu`; CUDA runs additionally use BF16 and CUDA Graphs
where supported. Run commands from the repository root.

## Optimized Two-Player Training

`scripts/train_optimized.py` is the primary two-player training entry point.

```sh
python scripts/train_optimized.py --output runs/new-run --iterations 100
```

Defaults include 2,048 environment slots, 128 simulations per move, 16 root
candidates, width 256, four residual blocks, and 500,000 replay positions.
These are training defaults, not guaranteed optimum settings for every machine.
The final recorded generation-1,800 model used 256 simulations, 1,000,000 replay
positions, learning rate 1e-4, uniform replay sampling, and sample reuse 2.

For an inexpensive installation test use the small command in the README. The
larger defaults need substantially more memory and compute.

The optimized path includes a factory-equivariant policy, sampled chance nodes,
duplicate sample reuse, optional subtree reuse and output caching, leaf buckets,
CPU/GPU inference pipelines, device-resident replay, an afterstate value head,
and graph-captured learner updates. These mechanisms are in the two-player path;
the three-player path does not implement every optimization.

## Resume and Retune

```sh
python scripts/train_optimized.py --resume runs/new-run/latest.pt --iterations 200
```

`--iterations` is the total target publication count. Resuming iteration 100 with
target 200 adds 100 iterations. Exact resume restores weights, optimizer moments,
replay contents and metadata, RNG state, counters, and remaining sample budget.
It reads saved protocol settings automatically and rejects incompatible explicit
changes. Reproducibility assumes the same software/hardware protocol.

Supported tuning changes require `--retune` and are recorded in `phase_history`:

```sh
python scripts/train_optimized.py --resume runs/new-run/latest.pt --retune --iterations 200 --learning-rate 0.0001 --replay-capacity 1000000 --recent-fraction 0 --sample-reuse 2 --simulations 256
```

Replay expansion preserves previous samples. Replay reduction is rejected.
Learning rate is applied after restoring the optimizer. Do not reuse a run's
output directory for an unrelated random initialization.

## Historical Opponents

The two-player historical pool pairs the learner against frozen checkpoints of
its own earlier versions. Only learner decisions produce policy targets; frozen
opponent actions are not imitated. See [Historical Pool](HISTORICAL_POOL_TRAINING.md).

## Checkpoint Storage

The two-player optimized trainer stores one `latest.pt`, a compressed replay
snapshot with required deltas, the initial actor, a bounded recent/anchor actor
set, and optional pinned checkpoints.

```sh
python scripts/checkpoints.py runs/new-run --pin 100 200
python scripts/checkpoints.py runs/new-run --prune --keep-recent 2 --keep-anchors 4
python scripts/checkpoints.py runs/new-run --compact
```

Only existing actors can be pinned. Preserve the full run directory to resume;
`latest.pt` alone does not contain its replay chain. A lightweight actor is enough
for inference. Never prune or compact the same directory while training writes it.

## Two-Player Evaluation

Portable baseline evaluation (CPU or CUDA):

```sh
python scripts/evaluate.py --candidate runs/new-run/actor-000200.pt --opponent runs/new-run/actor-000100.pt --simulations 256 --pairs 500 --output runs/match.json
```

This evaluates paired seats with deterministic initial seeds, uses the baseline
search interface, and reports win/draw/loss and a paired bootstrap interval.
Keep both actor files pinned while comparing them.

For evaluation with the complete optimized search configuration and independently
specified budgets, use the CUDA audit entry point:

```sh
python scripts/audit_ai.py --candidate runs/new-run/actor-000200.pt --opponent runs/new-run/actor-000100.pt --budget-a 256 --budget-b 256 --pairs 500 --output runs/optimized-match.json
```

`audit_ai.py` uses each checkpoint's tuning and requires CUDA. Do not compare its
results interchangeably with baseline-search results. Initial seeds pair opening
deals; later tile draws can diverge after different decisions. Truncated games are
excluded rather than counted as draws.

## Three-Player Training

The three-player network reads 244 features and outputs 240 action logits plus
three rank utilities in current-player order. Native search maps the vector to
absolute seats and uses the acting player's component at each Max-N decision.
Chance returns are weighted component by component. Utility targets use final
score rank: unique first 1, second 0, third -1; score ties average the ranks.

Create a three-player model by transferring a compatible two-player equivariant
actor. The file need not be from generation 1,800:

```sh
python scripts/train_3p.py --base runs/demo/actor-000003.pt --output runs/three_player --iterations 100 --envs 128 --simulations 64 --candidates 16
```

The default base path refers to the author's local iteration-1,800 actor, which
is not included in Git; always pass an available actor when starting a new run.

```sh
python scripts/train_3p.py --resume runs/three_player/latest.pt --output runs/three_player --iterations 200 --history runs/three_player/actor-3p-000100.pt
```

This trainer restores weights and optimizer, reapplies the requested learning
rate, reads compressed FP16 chunks from `replay3p/`, and keeps fixed validation
samples in `validation3p.npz`. Defaults are learning rate 5e-5, 16 minibatch updates
per iteration, batch size 2,048, and four recent replay chunks. Preserve that run
directory and use the same output/replay paths when resuming.

Three-player continuation controller:

```sh
python scripts/train_3p_longrun.py --output runs/three_player --start-generation 100 --block 100 --max-iterations 10000
```

It requires an existing `latest.pt`, trains one block, evaluates against its
block-start anchor, and stops on a failed promotion gate. The current implementation
retains experimental anchor naming and is intended for an existing run.

```sh
python scripts/evaluate_3p.py --candidate runs/three_player/latest.pt --opponent runs/three_player/actor-3p-000100.pt --pairs 300 --simulations 128 --candidates 16 --output runs/three-player-match.json
```

## Three-Player Limitations

- `--history` samples a complete self-play cohort from one frozen model or the
  current model. It is trajectory mixing, not learner-versus-history opposition.
  It also trains on frozen policies. The two-player opponent pool is different.
- Resume does not persist full NumPy/Torch RNG state. Three-player resume is not
  guaranteed to match uninterrupted training bit for bit.
- Replay restores existing directory chunks rather than verifying exactly the
  checkpoint's manifest. Missing replay is not rejected consistently, and chunk
  cleanup is not transactional with checkpoint publication.
- Fixed validation samples are taken from data that also entered replay. These
  losses diagnose drift; they are not a strictly held-out generalization metric.
- Evaluation balances candidate seats, but members of a three-game group do not
  share identical initial deals. Bootstrap grouping is weaker than a controlled
  common-seed, seat-swapped comparison.
- The winner metric uses score only and awards 0.5 for a highest-score tie. It does
  not implement the native rule's complete-row tiebreak, nor 1/3 for a three-way tie.
  The training utility also ignores the complete-row tiebreak.
- Promotion requires the lower interval bound to exceed 0.5; the symmetric
  three-player baseline is approximately 1/3. Failure to promote is not evidence
  that the candidate is weaker. Earlier reports calling 0.42 or 0.367 regression
  on that basis were incorrect.
- Historical anchor copies can duplicate a generation, and the controller's
  retention is not as fully bounded or verified as two-player checkpoint storage.

These issues are documented rather than silently changing saved training results
or rerunning experiments during source publication. They should be addressed
before using three-player evaluations as formal strength claims.

## Legacy Two-Player Trainer

`scripts/train.py` retains the original MLP/standard search implementation for
older checkpoints. Its weights are not interchangeable with the optimized
equivariant model. CLI help is available for all train/evaluate entry points.
