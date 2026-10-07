# Azul AlphaZero

[中文说明](README.zh-CN.md)

A C++20 Azul engine and GPU-accelerated pure self-play learning framework for
two- and three-player games. It combines Full Gumbel AlphaZero search with
explicit stochastic tile refills, reusable chance samples, batched native
environments, and PyTorch policy/value networks.

The repository includes training, evaluation, benchmarks, tests, a local play
interface, and a read-only Board Game Arena (BGA) analysis panel. It implements
original Azul with the fixed colored wall. Four-player games, the gray-wall
variant, and other Azul titles are not implemented.

## Playing Strength

**Both the two-player and three-player models substantially outperform the human
players encountered in practical BGA testing.**

The strongest checkpoint validated by the two-player model tournament is
**generation 1,800**. Generations mean model publication iterations, not games or
complete passes over a fixed dataset.

| Item | Two-player model | Three-player experiment |
| --- | --- | --- |
| Training progress | 1,800 publication iterations | Continued to iteration 500 |
| Initialization | Random weights, pure self-play | Transfer from two-player iteration 1,800 |
| Network | Factory-equivariant, width 256, 4 residual blocks | 7-factory equivariant, width 256, 4 residual blocks |
| Parameters | 623,618 | 639,490 |
| Search | Full Gumbel with stochastic chance nodes | Full Gumbel with three-component Max-N backup |
| Endpoint training search | 256 simulations per move | 64 simulations per move |
| Learner updates | 447,533 total | 4 per iteration initially; 16 in iterations 401-500 |
| Completed games | 4,207,953 in the checkpoint counter | 128 concurrent games per iteration; older resumed blocks reset counters |
| Hardware used | Ryzen 7 9800X3D and RTX 5070 Ti | Same local CPU/GPU |

The two-player model used a rolling pool of its own historical checkpoints from
iteration 1,401 onward. In iterations 1,701-1,800 the opponents were 1,400, 1,500,
1,600, and 1,699, weighted 1:2:3:4, with approximately 40% of learner policy samples
coming from games against history. No human records or external-engine targets
were used for training.

At equal 256-simulation budgets, iteration 1,800 scored **54.45%** against iteration
1,699 over 2,000 seat-swapped games (1,068 wins, 42 draws, 890 losses; paired 95%
interval **52.375%-56.60%**). At 1,024 simulations, it scored 52.15% over 1,000
games with a 49.20%-55.10% interval; this larger-budget advantage was inconclusive.

Three-player training is experimental. Iteration 500 is the latest local model.
Its evaluator uses a
conservative 0.5 promotion threshold although the equal-model baseline is
approximately 1/3. Failure to promote therefore does not establish regression.
See [Training](docs/TRAINING.md) for other implementation limitations.

Weights, replay buffers, downloaded rulebooks, and local experiment logs are
excluded from Git. New models can be trained without pretrained files. The smoke
example below checks installation and does not reproduce the reported strength.

## Build

Requirements: a C++20 compiler and CMake 3.20 or newer. Native tests do not need Python.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Windows users with Visual Studio C++ tools can build and test without CMake:

```powershell
./scripts/build-msvc.ps1 -Configuration Release
```

Outputs are in `build-release/`. Python bindings find the library in
`build-release/`, `build/Release/`, or `build/`. GCC/Clang builds optionally support
`-DAZUL_NATIVE=ON` and `-DAZUL_SANITIZE=ON`.

## Python Setup

Training requires Python 3.10 or newer, NumPy, and a compatible PyTorch build.
Use the [official PyTorch selector](https://pytorch.org/get-started/locally/)
for your GPU's CUDA requirements.

```sh
python -m venv .venv
# Activate .venv before the following commands.
python -m pip install -r requirements.txt
python examples/selfplay.py
```

The ctypes-only example uses the standard library. Training can run on CPU with
`--device cpu`. GPU paths provide pinned memory, BF16 inference, and CUDA Graphs
where supported. The optimized two-player trainer additionally uses pipelined
inference, device-resident replay, and captured learner updates.

## Train and Evaluate

Start a small two-player installation check:

```sh
python scripts/train_optimized.py --output runs/demo --iterations 3 --envs 16 --threads 4 --queues 1 --positions-per-update 1000 --simulations 16 --candidates 8 --width 64 --blocks 2 --batch-size 64 --warmup 64 --replay-capacity 20000 --updates-per-iteration 4
```

Continue to a total of 100 publication iterations:

```sh
python scripts/train_optimized.py --resume runs/demo/latest.pt --iterations 100
```

Evaluate compatible actors with equal search budgets:

```sh
python scripts/evaluate.py --candidate runs/demo/actor-000003.pt --opponent runs/demo/actor-000002.pt --simulations 128 --pairs 500 --output runs/evaluation.json
```

Run the evaluation before resuming while both files still exist, or pin them first.
See [Training](docs/TRAINING.md) for full-scale settings, three-player initialization,
resume constraints, evaluation caveats, and storage. The legacy two-player MLP
entry point is `scripts/train.py`.

## Local Play and BGA Analysis

```sh
python scripts/play.py --port 8765
python scripts/monitor_bga.py --port 8770
```

Open `http://127.0.0.1:8765/` for two-player local play or
`http://127.0.0.1:8770/` for the two-/three-player BGA panel. Models are read from
`runs/`; create or provide compatible checkpoints before requesting AI analysis.
The BGA collector reads public state and does not submit game actions.

Detailed UI guides are available in Chinese:
[Local Play](docs/LOCAL_PLAY.md), [Manual Dealing](docs/MANUAL_DEAL.md),
and [BGA Panel](docs/BGA_MONITOR.zh-CN.md).

## Source Layout

| Directory | Contents |
| --- | --- |
| `include/azul/` | Rules, Gumbel search, cache, C ABI |
| `src/` | Batched implementation and native benchmarks |
| `python/` | Bindings, networks, learners, storage, UI servers |
| `scripts/` | Build, train, evaluate, benchmark, and diagnostic commands |
| `tests/` | Rules, differential reference, search, resume, history, and UI tests |
| `examples/` | Standard-library-only binding example |
| `web/`, `bga_web/` | Browser UI assets |
| `docs/` | Rules, interfaces, user guides |

Observations include every player's public board, score, pattern lines, floor,
tile sources, and color-count bookkeeping. They exclude simulator RNG and future
draws. Bag/discard counts assume complete public history; external-board analysis
estimates missing history. See [Interfaces](docs/ARCHITECTURE.md),
[Two-Player Rules](docs/RULES.zh-CN.md), and [Three-Player Rules](docs/RULES_3P.zh-CN.md).

## Tests

After building and installing Python dependencies:

```sh
python tests/test_ai.py
python tests/test_optimized.py
python tests/test_resume.py
python tests/test_optimized_resume.py
python tests/test_checkpoint_storage.py
python tests/test_history_pool.py
python tests/test_retune.py
python tests/test_public_origin.py
node tests/test_bga_collector.js
```

Node.js is only required for the collector test. Additional local-play/BGA
integration tests require checkpoints and isolated DLL builds as explained in
their guides. CUDA tests skip automatically when CUDA is unavailable.
