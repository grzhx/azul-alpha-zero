# Factory-equivalent chance search

The 2-player and 3-player native trees use factory-canonical states. Only the
factory slots are sorted; center, colors, boards, player identities and bag/
discard counts remain intact. A stable insertion sort over packed five-color
factory count keys avoids heap allocation (5 or 7 slots).

Chance outcomes are drawn by the unchanged engine deal routine using the real
bag/refill rules BEFORE sorting. Existing canonical children share the subtree
and increment `ChanceEdge.multiplicity`; the parent sample count increments for
every draw. X, permutation(X), Y produce multiplicities 2 and 1. Both later
sampling from the retained pool and value expectation use these masses; there
is no uniform sampling over distinct canonical boards. `chance_cap` limits raw
draw count, not unique child count. The optional afterstate prior keeps its
existing weight and is separate from sampled multiplicity.

Canonicalization applies at roots and decision transitions too. Each begin/reset
records canonical-to-external factory slots. Native request/logits and subtree
edges use canonical indices; result policies and selected actions map back to
the actual game's factory indices. Center is unchanged. Re-rooting canonicalizes
the real state before exact semantic comparison, preserving reuse after real
deals or factory reorderings. State equality still includes other game fields.
All results of real self-play are stored in external observation/action order.

The chance cap is 256 raw samples; progressive widening controls draws actually
made. Old checkpoints retain their saved protocol on resume unless explicitly
retuned with `--retune --chance-cap 256`.

Native `azul_search_revision()` returns 2. Optimized training rejects an older
library to prevent silently changing chance semantics. Standard builds support
up to 4,096 simulations; use CMake `AZUL_SEARCH_MAX_SIMULATIONS` to build a larger
offline search budget.

Tests cover 2:1 empirical mass, permutation-equivalent deals, analytic expectation,
external action mapping, re-rooting and three-player Max-N permutation handling.
Sorting does not enumerate all outcomes. No end-to-end speedup or strength gain
is inferred from correctness tests alone.
