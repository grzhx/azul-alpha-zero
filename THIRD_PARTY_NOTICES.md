# Third-party notices

## DeepMind Mctx

Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.

The Sequential Halving visit schedule and completed-Q / deterministic action
selection implementations in `include/azul/search.hpp` and
`include/azul/search3.hpp` are adapted into C++ from
the algorithmic implementation in DeepMind Mctx:

- https://github.com/google-deepmind/mctx/blob/main/mctx/_src/seq_halving.py
- https://github.com/google-deepmind/mctx/blob/main/mctx/_src/qtransforms.py
- https://github.com/google-deepmind/mctx/blob/main/mctx/_src/action_selection.py

Changes include the C++ indexed memory arenas, native batched request/submit
interface, legal-action-only representation, player-0 value storage and Azul's
sampled chance-node integration. Mctx itself is not a runtime dependency.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
the covered material except in compliance with the License. You may obtain a
copy of the License at https://www.apache.org/licenses/LICENSE-2.0 . A local copy
is in `licenses/MCTX-LICENSE.txt`.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Azul is a game designed by Michael Kiesling. Its name, rulebook, and original
game materials belong to their respective owners. This repository contains an
independent software implementation and external rules references. Downloaded
rulebooks and reference web pages are not distributed with the source.
