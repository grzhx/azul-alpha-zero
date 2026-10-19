# Generation 2300 Algorithm Release

Prepared 2026-10-19. This consolidates previously uncommitted work after the
generation-1,800 public release and selects generation 2,300 as the publication
baseline. Historical training did not create Git commits at each generation;
this is a curated reconstruction, not a claimed exact historical source checkout.
Checkpoint source hashes and compact results are in [the evidence file](release-2300-evidence.json).

## Included

- Two-/three-player rules and C ABI, fixed colored walls, independent episode RNG.
- Full Gumbel Sequential Halving; native final action remains distinct from
  the improved training-policy target, including offline independent-tree voting.
- Factory-canonical root/child states and external action mapping for 5/7 factories.
  Chance merging retains raw draw multiplicities and true deal distributions.
- Chance cap 256 with the historical progressive-widening schedule and random
  reuse proportional to sample multiplicity. The later adaptive allocator is removed.
- Optional 45-dimensional ending features and zero-column migration; separate
  residual value tower, terminal-utility auxiliary MSE and WDL label smoothing.
- Complete-game holdout validation, round/ending calibration and policy diagnostics;
  explicit actor restarts with fresh optimizer/replay versus full-state resume.
- GPU replay, BF16 inference, CUDA Graph inference/learning, versioned caches,
  historical opponents, lossless replay journaling and bounded actor retention.
- Smaller initial tree reservations, lazy inactive roots, obsolete-parent scratch
  release after re-rooting, and complete-cohort arena reclamation. No GPU-buffer
  reclamation or loss of within-trajectory search evidence is required.
- Optional continuation/action-ranking module implemented before this baseline.
  It is disabled in the 2,201–2,300 run; its availability does not establish a gain.
- Native progress, partial-result and continuing-search APIs and their algorithm
  tests, without HTTP/browser integrations.

## Excluded

All local-play/BGA/ranked web assets, collectors, browser automation, service
entry points, UI tests and UI documents are removed from the current source tree.
Git history still contains the earlier release; its history has not been rewritten.
The live local development directory and its frontend/experiment files are intact.

Post-baseline structured rule/action encoders, structured value residuals, fused
CUDA rule kernels, common-random-number chance scenarios, adaptive breadth/depth,
balanced chance reuse, and the automatic 2,400 continuation controller are excluded.
Neither training state nor heavyweight experimental results are uploaded.

## Why Stop at 2300

The 2,300 versus 2,200 paired-seat comparisons completed 1,000 games per budget:
54.75% at 1,024 simulations (95% 51.65%–57.85%), 52.75% at 2,048
(49.65%–55.85%), and 53.85% at 4,096 (50.75%–56.95%). Against 2,100 at
1,024 it scored 52.50% (49.45%–55.60%). This is budget-dependent evidence,
not proof of dominance over every older model or every budget.

Later standard 2,400 versus 2,300 scored 51.70%, 47.55%, and 48.75% at
1,024/2,048/4,096. All intervals included 50%; no reliable improvement was
demonstrated. Selecting 2,300 is a release choice and does not assert that 2,400
is statistically weaker. Later rule-structure experiments are a separate lineage.

## Reproduction Limits

Only actor weights survive at the exact 2,300 endpoint; its original full replay
and optimizer were superseded by the local 2,400 continuation. An actor-weight
restart is supported but must not be described as full-state exact resume.
Full-state checkpoints with their replay chains still support exact resume.
The historical run's cumulative update counter includes a restart at 1,900;
it is not a lifetime count from the initial random model.

The release is verified by native/Python regressions and comparison with the
retained historical revision-2 DLL, not by rerunning long training or tournaments.
The [verification record](release-2300-verification.json) covers 100 batches of
eight roots (539 active decisions, 2,555 chance draws, 4,484 chance reuses), plus
12 batches with the actual 2,300 network and optimized subtree/cache search.
Actions, policies, values and applicable native counters matched the retained DLL.
