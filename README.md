# Azul AlphaZero

[中文说明](README.zh-CN.md)

A C++20 Azul engine and GPU-accelerated pure self-play learning framework for
two- and three-player games. This algorithm-only release consolidates the
training line through generation **2,300**: Full Gumbel AlphaZero, explicit
stochastic refills, factory-canonical chance merging, batched native search,
PyTorch policy/value networks, replay, checkpoint storage, and offline evaluation.

The current source tree excludes browser interfaces, BGA collectors, local web
servers, and browser automation. It implements original Azul's fixed colored
wall; four-player games, gray walls, and other Azul titles are not implemented.

## Playing Strength and Release Baseline

Generation **2,300** is the selected two-player release baseline. Generations
count model publication iterations, not games or dataset epochs. It uses a
factory-equivariant network, width 256, four shared residual blocks, 45 derived
ending features, and a separate two-block value tower: **1,031,427 parameters**.
The three-player experiment remains at generation 500 (639,490 parameters),
initialized from the earlier two-player generation 1,800 model and using Max-N.

The 2,201–2,300 phase completed **332,484 games**, **13,118,656 decision samples**,
and **25,623 learner updates** with zero completed-cohort truncations. It used:

| Setting | Generation-2,300 phase |
| --- | --- |
| Training search | 2,048 simulations, 16 candidates, depth 96 |
| Concurrent environments / target decision samples | 2,048 / 131,072 per publication |
| Chance pool cap | 256 raw draws, merged with empirical multiplicities |
| Replay / batch / sample reuse | 1,000,000 / 1,024 / 2 |
| Learning rate | 5e-5 |
| Historical opponents | 1,800 / 1,900 / 2,002 / 2,100, weights 1:2:3:4 |
| Historical policy samples | Approximately 40% |
| Policy / value / utility auxiliary weights | 1 / 1 / 0.1 |
| Value label smoothing / afterstate loss | 0.01 / 0.25 |
| Extra rollout/action-ranking supervision | Disabled in this phase |

No human records, external-engine labels, or manually selected move targets are
used. Historical opponents are frozen versions of the same training lineage.

At equal per-move search budgets, generation 2,300 versus 2,200 scored:

| Simulations each | Games | Wins / draws / losses | Score rate | Paired 95% interval |
| --- | ---: | --- | ---: | --- |
| 1,024 | 1,000 | 541 / 13 / 446 | 54.75% | 51.65%–57.85% |
| 2,048 | 1,000 | 519 / 17 / 464 | 52.75% | 49.65%–55.85% |
| 4,096 | 1,000 | 533 / 11 / 456 | 53.85% | 50.75%–56.95% |

Later generation-2,400 training did not show reliable improvement over 2,300
at any tested budget. This does not prove regression. Later structured-rule/CUDA
branches, adaptive chance allocation, balanced chance reuse, and the automatic
2,400 continuation controller are excluded from this release.
See [Release Changes](docs/RELEASE_2300.md) for the complete boundary and evidence.

Weights, replay buffers, full local logs, and binaries are not distributed in Git.
Training from source is supported; the smoke example does not reproduce this strength.

## Download Generation 2300 and Play Locally

The [generation-2300 release](https://github.com/grzhx/azul-alpha-zero/releases/tag/v2300.1)
distributes weights and an optional standalone player separately from algorithm source:

- [Inference weights](https://github.com/grzhx/azul-alpha-zero/releases/download/v2300.1/azul-generation-2300.pt)
  (about 4.15 MB; platform-independent PyTorch actor).
- [Windows x64 local-play ZIP](https://github.com/grzhx/azul-alpha-zero/releases/download/v2300.1/azul-2300-local-play-windows-x64.zip)
  (about 3.99 MB; includes the model, native DLL, and a separate local UI).
- [Model metadata](https://github.com/grzhx/azul-alpha-zero/releases/download/v2300.1/azul-generation-2300.json)
  and [SHA256 checksums](https://github.com/grzhx/azul-alpha-zero/releases/download/v2300.1/SHA256SUMS.txt).

For local human-versus-AI play, install **64-bit Python 3.11+**, extract the ZIP,
and run **Start-Windows.cmd**. First launch downloads CPU PyTorch and NumPy to
the package's `.venv`. Open **http://127.0.0.1:8765/**, select generation 2,300
and human-versus-AI mode, then start a game. CPU works without an NVIDIA GPU;
start with 128 or 512 simulations. The package README explains optional CUDA
installation and the Microsoft runtime if needed. The player interface is in English.

The main branch remains algorithm-only; the separate release ZIP includes its
own local UI. Weights contain no optimizer or replay and cannot restore the exact
historical generation-2,300 training state. macOS/Linux users can use the weights
with the source-built engine; the supplied precompiled player targets Windows x64.

## Build

Requirements: a C++20 compiler and CMake >= 3.20. Native tests do not require Python.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Windows with Visual Studio C++ tools can also use:

```powershell
./scripts/build-msvc.ps1 -Configuration Release
```

Python discovers libraries under `build-release/`, `build/Release/`, or `build/`.
Use `--native-library` to select an isolated training library. A revision-2 native
library is required for factory-canonical merging. CMake optionally supports
`AZUL_NATIVE`, `AZUL_SANITIZE`, `AZUL_SEARCH_MAX_SIMULATIONS`, and `AZUL_ANYTIME`.

## Python Setup

Use Python >= 3.11, NumPy, and a compatible PyTorch build. Python 3.11 is needed
by checkpoint hashing through `hashlib.file_digest`. Select CUDA packages using
the [official PyTorch installer](https://pytorch.org/get-started/locally/).

```sh
python -m venv .venv
# Activate .venv before continuing.
python -m pip install -r requirements.txt
python examples/selfplay.py
```

CPU is supported with `--device cpu`. CUDA paths provide BF16 inference,
pinned-memory transfers, graph replay, device-resident replay, and captured learning.

## Train and Evaluate

Small two-player installation check:

```sh
python scripts/train_optimized.py --output runs/demo --iterations 3 --envs 16 --threads 4 --queues 1 --positions-per-update 1000 --simulations 16 --candidates 8 --width 64 --blocks 2 --end-features --batch-size 64 --warmup 64 --replay-capacity 20000 --updates-per-iteration 4
```

Resume to a total of 100 publications:

```sh
python scripts/train_optimized.py --resume runs/demo/latest.pt --iterations 100
```

Evaluate compatible actors before they are pruned, or pin them first:

```sh
python scripts/evaluate.py --candidate runs/demo/actor-000003.pt --opponent runs/demo/actor-000002.pt --simulations 128 --pairs 500 --output runs/evaluation.json
```

The portable evaluator uses the baseline search interface. `scripts/audit_ai.py`
uses full optimized tuning and CUDA; `scripts/evaluate_tree_counts.py` compares
independent-tree voting. They are different evaluation protocols.
See [Training](docs/TRAINING.md), [Ending Features](docs/ENDING_FEATURES.md),
[Historical Pool](docs/HISTORICAL_POOL_TRAINING.md), and
[Factory Symmetry](docs/CHANCE_FACTORY_SYMMETRY.md).

## Source Layout

| Directory | Contents |
| --- | --- |
| `include/azul/`, `src/` | Rules, C ABI, Gumbel/chance search, factory symmetry, benchmarks |
| `python/` | Bindings, networks, self-play, learners, replay, checkpoint storage |
| `scripts/` | Build, training, offline evaluation, diagnostics, benchmarks |
| `tests/` | Native rules/search, learning, migration, exact resume, evaluation tests |
| `examples/` | Standard-library-only binding example |
| `docs/` | Rules, algorithm and training guides, release evidence |

Public observations contain every player's board, score, pattern lines, floor,
tile sources, and public-history color counts. Simulator RNG and future draws
are never exposed. See [Interfaces](docs/ARCHITECTURE.md),
[Two-Player Rules](docs/RULES.zh-CN.md), and [Three-Player Rules](docs/RULES_3P.zh-CN.md).

## Tests

After building the native library and installing Python requirements:

```sh
python tests/test_ai.py
python tests/test_optimized.py
python tests/test_resume.py
python tests/test_optimized_resume.py --end-features --release-cohort-arenas
python tests/test_checkpoint_storage.py
python tests/test_history_pool.py
python tests/test_retune.py
python tests/test_end_features.py
python tests/test_value_training.py
python tests/test_policy_balance.py
python tests/test_action_ranking.py
python tests/test_search_decision.py
python tests/test_tree_count_evaluation.py
```

CUDA tests skip when unavailable. For optional GPU verification use
`tests/test_optimized_resume.py --device cuda --end-features` and
`tests/test_action_ranking.py --device cuda`. Native continuation/partial-result
APIs remain as algorithm capabilities; this release has no browser or HTTP dependencies.
