# Dataset provenance and Stage 2 source investigation

Recorded on 17 October 2026. **The student explicitly approved the
specification-linked 128-feature file. This records Stage 2's data validation;
subsequent task-model runs are documented in the README.**
Generated downloads stay in ignored `data/` directories.

## Original prescribed PyG artifact

- Loader: `torch_geometric.datasets.FacebookPagePage`, pinned PyG 2.6.1.
- Source: <https://graphmining.ai/datasets/ptg/facebook.npz>.
- Expected numeric keys from the loader: `features`, `target`, `edges`.
- Expected sizes: 22,470 nodes, 128 features, four classes, 342,004 edge entries.
- Direct PyG download and the project's downloader both failed with DNS error
  `getaddrinfo failed` (Windows error 11001).
- Upstream reports: <https://github.com/pyg-team/pytorch_geometric/issues/10525>
  and <https://github.com/pyg-team/pytorch_geometric/pull/10415>.
- The PR's proposed raw-SNAP feature conversion is not evidence of equivalence
  to the original 128-feature matrix; it is not used here.

## Sources investigated

- Course repository `shakes76/PatternAnalysis-2026`: both main and
  topic-recognition trees contain no dataset artifacts. No relevant public
  dataset issue was found in its issues listing.
- Appendix A lists `/home/groups/comp3710/` but no Facebook-specific path. The
  linked compute instructions are
  <https://student.eait.uq.edu.au/infrastructure/compute/>. No configured Rangpur
  connection was found, and no Rangpur filesystem was inspected. A local SSH
  alias for another UQ server does not establish Rangpur access.
- Original MUSAE repository and scientific repositories SAMGPT, PULL,
  GraphFramEx and Lipschitz-GNN did not provide an original Facebook artifact
  in the public trees/releases inspected.
- An archive index did not yield the original Facebook NPZ. Guessed PyG-hosted
  dataset mirror paths returned 404. No unverified pickle collection was loaded.

## Approved specification-linked artifact

**Important correction:** the embedded hyperlink on Appendix A item 6's
`PyG FacebookPagePage` text points to
<https://github.com/gayanku/facebook_page-page_dataset>. Plain-text extraction
showed the link text without its destination. The repository is therefore
explicitly linked by the supplied specification, although its artifact is a
reconstruction rather than a verified copy of GraphMining's original features.

Approved artifact inspected without enabling NumPy pickle:

- Immutable source revision: `ec01d02c39da4385052023ed28e36f0794f61dc8`.
- File: `facebook.npz`, 11,989,878 bytes.
- URL: <https://raw.githubusercontent.com/gayanku/facebook_page-page_dataset/ec01d02c39da4385052023ed28e36f0794f61dc8/facebook.npz>.
- SHA256: `7a1b8a0687515e9859d1ca649dc276825f5db44568894dfbb6b55ebc1cf6a9c1`.
- Numeric keys: `X`, `edges_x`, `edges_y`, `y`. `page_name` is an object array
  and was deliberately not deserialised.
- Features: `(22470, 128)`, float32, finite, observed range approximately
  -2.037593 to 3.002028.
- The supplied preprocessing notebook constructs description-token indicators
  and applies `TruncatedSVD(n_components=128, random_state=42)` without passing
  labels to the feature transform. The SVD was not rerun here. Its features are
  not demonstrated to equal GraphMining's original representation.
- Edge entries: 342,004; unique unordered pairs including loops: 171,002.
  Duplicate ordered entries: 179; self-loop entries: 358; isolated nodes: zero.
  The entries are reciprocal. No graph repairs were applied.
- Every unique directed edge matches the raw UCI graph after adding reverse
  edges, and all 22,470 labels match UCI by node ID.

**The source README's class-ID description is wrong.** Direct comparison shows:

| ID | Observed class | Nodes |
| --- | --- | ---: |
| 0 | company | 6,495 |
| 1 | government | 6,880 |
| 2 | politician | 5,768 |
| 3 | tvshow | 3,327 |

This matches the notebook's actual alphabetical category encoding. Following
the student's explicit source approval, the exact pinned file was installed in
the ignored project data directory. A numeric adapter constructs PyG Data
directly because the built-in FacebookPagePage loader expects different keys.
The adapter preserves supplied numeric values and does not fit new features.
Checksums are verified on each load; optional object metadata is not unpickled.
An audit JSON and provenance manifest are generated locally and excluded from
Git. The final suite passed: 22 synthetic tests plus 6 real-data/operator tests, including
the UCI edge/label comparison. These are not validation of the unavailable
original feature representation, and they are not model performance results.
The added normalization check confirms one effective self-loop per node and
equivalence to deduplicated input under PyG 2.6.1's default unweighted COO GCN
normalization. It does not establish the same behavior for other operators.

## Raw UCI source used only for comparison

- Source: <https://archive.ics.uci.edu/dataset/527/facebook+large+page+page+network>.
- Download: <https://archive.ics.uci.edu/static/public/527/facebook+large+page+page+network.zip>.
- Archive SHA256: `fa2f5c4bcb806695873acaba2769c318aa41085417074ae0406fe7c450a4e0d8`.
- Observed: 22,470 contiguous node IDs; 4,714 description-word feature IDs;
  171,002 raw edge rows, including 179 self-loops; no duplicate ordered rows;
  all nodes incident to an edge. Class counts match the table above.
- The raw release is not substituted for the prescribed 128-feature input.
  Its labels/edges serve only as an independent source comparison.

## Information to request from teaching staff if original data is required

Request the original 128-feature `facebook.npz` used by PyG's
`FacebookPagePage`, its authoritative download link or exact Rangpur path and
access instructions, a SHA256 checksum if available, and the integer-class-name
mapping. Explain that GraphMining's host fails DNS resolution.

The current project uses the existing gayanku artifact linked by Appendix A with
explicit student approval. This is not a claim of a separate staff approval or
of original-PyG feature equivalence. If staff later require a different artifact,
record that change and regenerate all splits/experiments under a documented
protocol; do not mix results between representations.
