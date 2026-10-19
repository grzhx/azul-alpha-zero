# Two-Player Ending Features

The optional `end_features` architecture adds 45 rule-derived input features to
the existing policy/WDL network. The native ABI and replay remain 172 floats per
observation. Features are derived on the model device for every forward pass,
including search leaves and afterstates; existing replay requires no conversion.
There is no score-margin head, new reward, endgame solver, or algorithm switch.

## Feature Layout

The first 22 features belong to the observation's current-player perspective;
the next 22 belong to its opponent. Each block contains:

| Offset | Count | Meaning | Scale |
| --- | --- | --- | --- |
| 0 | 5 | Empty wall cells in each horizontal row | /5 |
| 5 | 5 | Pattern line already full | 0 or 1 |
| 10 | 5 | Empty wall cells after committed full pattern lines transfer | /5 |
| 15 | 5 | A full pattern line can complete its wall row | 0 or 1 |
| 20 | 1 | End bonus on the projected wall (rows, columns, colors) | /95 |
| 21 | 1 | Penalty for currently occupied floor spaces | /14 |

Feature 44 is one if either projected wall has a complete horizontal row. During
drafting, this means the game will certainly end after the current round settles:
full pattern lines cannot be emptied before settlement. It is not a fixed round
limit. An empty line can take any missing color; a nonempty line is color-locked.
The projected wall and its bonus are conditional on no additional placements;
they are not predictions of the final score. In afterstates, the observations
already contain the settled wall and cleared pattern lines/floors.

## Compatibility and Training

New input columns are appended after existing inputs and initialized to zero.
`load_compatible_state_dict` explicitly pads old input weights; ordinary loading
remains strict. AdamW input-weight moment tensors receive matching zero columns.
Existing weights, old moments, replay, RNG and progress are retained. Old frozen
historical actors can load into the augmented architecture with zero new columns;
their files and hashes remain unchanged.

Enable on the optimized training entry point:

```powershell
.\.venv\Scripts\python.exe scripts/train_optimized.py --resume runs/optimized/latest.pt --retune --end-features --iterations 1900
```

This command runs training; implementation alone does not mutate checkpoints or
start training. After the first new checkpoint, ordinary resume inherits the flag.
New runs can use `--end-features` directly. Disabling trained features on resume is
rejected. Actor architecture metadata ensures evaluation and inference create
the correct model automatically. Old checkpoints remain unchanged until migrated.

The WDL objective still treats all wins equally. These features help learn when
future rounds cannot happen; they do not establish a preference for winning by a
larger margin, and playing-strength improvement requires training and evaluation.

## Validation

`tests/test_end_features.py` checks pending row completion with nontrivial color
mapping, incomplete and incompatible lines, end bonuses, floor penalties, real
native-game observations, zero-column migration, feature gradients, factory
permutation symmetry, CPU checkpoint migration with old history/replay, subsequent
resume, and CUDA Graph inference/learning. Feature extraction uses batched tensor
operations and fixed registered lookup buffers, without CPU transfers or dynamic
per-position loops.

Local 2026-10-08 timing with the generation-1800 actor, BF16 frozen inference,
batch 1024 and CUDA Graph replay (median of three 200-forward blocks) measured
0.156 ms without features and 0.246 ms with features. This is about 58% extra
network-forward time, not a measurement of total search or training slowdown.
No playing-strength gain has been measured, and existing actor files have not
been migrated or trained by this implementation pass.
