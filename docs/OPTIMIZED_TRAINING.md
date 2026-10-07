# Optimized Two-Player Training

The primary entry point is `scripts/train_optimized.py`. See the consolidated
[Training and Evaluation](TRAINING.md) guide for setup, commands, exact resume,
retuning, and checkpoint retention.

## Implementation Map

| Module | Responsibility |
| --- | --- |
| `python/azul_ai/model.py` | Factory-equivariant network, policy/value and afterstate heads |
| `include/azul/search.hpp` | Gumbel Sequential Halving, completed-Q, chance samples, subtree reuse |
| `include/azul/evaluation_cache.hpp` | Versioned native evaluation cache |
| `python/azul_ai/optimized_search.py` | Frozen actors, leaf buckets, inference pipeline |
| `python/azul_ai/optimized_selfplay.py` | Complete-game collection and historical opponents |
| `python/azul_ai/optimized_learning.py` | Device replay, graph learner, checkpoint journal |
| `python/azul_ai/checkpoint_storage.py` | Replay compression, actor retention and pins |
| `python/azul_ai/optimized_train.py` | Training orchestration, sample budget, protocol validation |

## Performance Controls

- Choose `--envs`, `--threads`, and `--queues` for the available CPU/GPU rather
  than assuming a larger batch always improves throughput.
- `--positions-per-update` adjusts cohort size; episodes finish before producing
  terminal-supervised targets.
- CUDA inference normally uses BF16 and graph replay; `--fp32` and `--no-graphs`
  are available for debugging and compatibility.
- Replay remains on the learner device. Capture combines sampling, augmentation,
  forward/backward, gradient clipping, and fused AdamW.
- `--sample-reuse` budgets learner samples per new policy position. It changes
  optimization dynamics; it is not an equivalence-preserving speed switch.
- Cache and subtree reuse are versioned. Actor publication prevents stale model
  values from being reused after a weight update.
- Tail padding can cause unused GPU work. The framework does not guarantee full
  GPU utilization on every move.

## Benchmarks

```sh
python scripts/benchmark_search.py --output reports/search-benchmark.json
python scripts/benchmark_optimized.py --output reports/optimized-benchmark.json
python scripts/benchmark_learner.py --output reports/learner-benchmark.json
```

Benchmarks require CUDA; generated `reports/` files are ignored by Git. The learner
benchmark uses synthetic data only for throughput and does not train a game model.
Compare measured speed separately from fixed-budget search quality and playing
strength per wall-clock time.
