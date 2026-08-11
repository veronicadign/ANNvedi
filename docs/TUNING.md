# Scenario × dataset tuning matrix

Working document for the competition strategy: one row per dataset family, one column per
scenario. Fill cells as they get tuned; every claim should point at its evidence
(sweep file, verify script, or a `results.db` run).

## What is already known

### Dataset properties (from `notebooks/dataset_analysis.ipynb`)

| dataset (stem `<name>-public`) | dims | notes |
|---|---|---|
| agnews-mxbai        | 1024 | **NOT unit-normalized** → needs true L2, dot-product shortcuts invalid |
| celeba-resnet       | 2048 | **NOT unit-normalized** → same caveat; widest after simplewiki → watch memory |
| gooaq-distilroberta | 768  | unit-norm; ⚠️ currently missing from `orthogonal-competition/datasets.txt` |
| imagenet-clip       | 512  | unit-norm |
| landmark-nomic      | 768  | unit-norm; contrast ratio 0.268 → **easiest for HNSW** (low ef suffices) |
| simplewiki-openai   | 3072 | unit-norm; ~3 GB raw train → **memory-scenario stress case**, SQ8/PQ mandatory; high contrast → hardest for HNSW (needs higher M / ef_construction) |
| yahoo-minilm        | 384  | unit-norm; the only dataset tuned in depth so far |

### Chosen configs so far (all measured on yahoo-minilm unless noted)

`scenarios.yaml` now carries measured per-dataset blocks for all 7 datasets (see
the matrix below). Cross-dataset defaults (what an unknown stem would hit):
`high_recall` M16/efC100/heurON/ef120, `fast` M8/efC64/heurON/ef120,
`memory` M8/efC64/heurON/ef300 — all sq8.

Still open on `memory`: recall clears the bar everywhere, but the ≤2×-faiss speed
gate (~0.44ms on this machine) is exceeded by every ≥0.95 config measured
(0.46-1.35ms) — closing it needs kernel speed work, and actual RSS ranking beyond
the analytic estimate needs a verified tuner run on the competition machine.

Evidence base:
- `experiments/results/dense_sweep_results.json` — 27,600 measured IVF-LSH configs
  (params + recall + query ms + fit s), yahoo-minilm 100k. Selector: `experiments/print_sweep_results.py`.
- `archive/HNSW/scenarios.yaml` — the only record of tuned params for HNSW's experimental
  hybrid modes (lsh / lsh_sq8 / lsh_float).
- `orthogonal-competition/results.db` — 104 historical harness runs (through Jul 8) incl.
  faiss-hnsw baseline numbers.
- HNSW mode profiling (float / sq8 / lsh at ef 40/120/250): `experiments/profile_hnsw.py`.

### Measured per-dataset picks (2026-08-11, focused sweep on FULL datasets)

All cells: HNSW sq8 + **heuristic ON**; only M/efC and ef vary. Every cell clears
its bar (high_recall/memory ≥ 0.955 = bar+margin, fast ≥ 0.855). Sweep:
`--grid focused --backends hnsw --no-verify --queries 1000`, all 1000 file-GT
queries, k=100, single-thread, local machine. Full measurements:
`experiments/results/tuning_20260811_210929.json` (re-select via `--from-json`).

| dataset ↓ / scenario → | high_recall | fast | memory (recall; RSS unverified) |
|---|---|---|---|
| agnews-mxbai        | M16/ef120: .9684 @ .68ms | M8/ef120: .9231 @ .50ms | M8/ef250: .9636 @ .90ms |
| celeba-resnet       | M16/ef120: .9629 @ .84ms | M8/ef120: .8867 @ .58ms | M8/ef300: .9569 @ 1.28ms |
| gooaq-distilroberta | M16/ef100: .9617 @ .84ms | M8/ef100: .9125 @ .73ms | M8/ef200: .9554 @ .99ms |
| imagenet-clip       | M8/ef170:  .9603 @ .69ms | M8/ef120: .9389 @ .57ms | M8/ef170: .9603 @ .69ms |
| landmark-nomic      | M8/ef120:  .9643 @ .46ms | M8/ef120: .9643 @ .46ms | M8/ef120: .9643 @ .46ms |
| simplewiki-openai   | M16/ef100: .9568 @ .99ms | M8/ef120: .9149 @ .73ms | M8/ef250: .9563 @ 1.35ms |
| yahoo-minilm        | M16/ef120: .9615 @ .58ms | M8/ef100: .8785 @ .41ms | M8/ef300: .9578 @ .83ms |

Notes: M16 builds are 2.5-3x slower than M8 (e.g. gooaq 1041s vs 387s) — where M8
passes high_recall (landmark, imagenet), it wins on build time too. The memory
column picks the smaller M8 graph via the analytic estimate; actual RSS was not
re-verified locally (`--no-verify`) — run the tuner with verification on the
competition machine for the final word. The old ivf_lsh memory hypotheses are dead
at full scale (recall ~0.6, see full-scale benchmark).

## The k=100 regime (why the old low-ef configs are dead)

The evaluator queries with **k=100**, and `hnsw_cpp` floors `ef` at `k` — so every
query runs a ≥100-wide beam no matter what `ef` the scenario asks for. Consequences:

- The `fast` scenario's `ef: 30` is silently `ef: 100`; low-ef tuning knowledge from
  the k∈{5,10} era does not transfer. The remaining HNSW speed levers are **M**
  (graph degree), build mode, and the backend choice itself.
- Recall saturates **at 100k subsets only**: M=16/ef=100 scores recall 1.0000 on
  every dataset tried there, but this DOES NOT transfer — on full yahoo (676k) the
  same config w/o heuristic measures 0.899. Never pick configs from subset recall;
  sweep full datasets (the full-scale benchmark below is the cautionary tale).

**Neighbor-selection heuristic (graph sparsity/diversity, `heuristic: true`)**:
implemented in `src/hnsw.cpp` (`fit(..., heuristic=)`), measured 2026-08-10 at 100k
subsets, sq8, k=100:

| cell | heuristic OFF | heuristic ON |
|---|---|---|
| yahoo × high_recall  | 3666 qps, build 16s | 3172 qps, build 31s |
| yahoo × fast         | 5121 qps, build 6s  | 4376 qps, build 10s |
| simplewiki × high_recall | 1333 qps, build 61s | 835 qps, build 224s |
| simplewiki × fast    | 2097 qps, build 23s | 1228 qps, build 76s |

With recall already at ceiling, diversity edges only widen the beam's exploration
(more distance evals) and slow the build — so the default was set to OFF.

**CORRECTION (2026-08-11, full-scale benchmark):** that conclusion was a 100k-subset
artifact — at 100k recall sits at 1.0 either way, so the heuristic could only show
its costs. On the FULL dataset it is worth **~+0.05 recall at equal ef** (0.899 →
0.947 at M16/ef100) and is the difference between failing and passing the 0.95 bar.
`scenarios.yaml` defaults now set `heuristic: true`; the module-level default stays
`false` (backward-compatible for 4-arg callers / frozen baselines). At small M the
heuristic even builds FASTER (M8: 83s vs 110s). OFF remains worth sweeping only
where recall saturates at full scale too.

## Full-scale benchmark vs faiss (2026-08-11, local machine)

Full yahoo-minilm (676,305 × 384), 1000 queries, k=100, single-thread queries,
multithread builds, evaluator-style recall against file GT. faiss-cpu 1.15.0,
`IndexHNSWFlat` exactly as the official baseline competitor runs it.

| config | recall@100 | mean ms | qps | build |
|---|---|---|---|---|
| faiss M16/efC100, efSearch=50 (baseline cfg) | 0.8484 | 0.221 | 4532 | 148s |
| faiss efSearch=100 | 0.9336 | 0.348 | 2872 | " |
| faiss efSearch=200 | 0.9746 | 0.621 | 1609 | " |
| ours sq8 M16/efC100 heurOFF, ef→100 | 0.8988 | 0.558 | 1793 | 202s |
| ours sq8 M16/efC100 heurOFF, ef=200 | 0.9476 | 0.839 | 1192 | " |
| ours sq8 M8/efC64 heurOFF, ef→100 | 0.7598 | 0.520 | 1924 | 110s |
| ours sq8 M16/efC100 **heurON**, ef→100 | 0.9467 | 0.781 | 1280 | 411s* |
| ours sq8 M16/efC100 **heurON**, ef=120 | 0.9614 | 0.579 | 1727 | 274s* |
| ours sq8 M16/efC100 **heurON**, ef=200 | 0.9811 | 1.150 | 870 | " |
| ours sq8 M8/efC64 **heurON**, ef→100 | 0.8755 | 0.368 | 2717 | 83s |
| ours ivf_lsh 20/8/512, probes 8–20, r≤300 | 0.58–0.60 | 0.36–0.41 | — | 137s |

\* same build config; 411s vs 274s is machine-load variance across runs.

Takeaways:
- faiss does NOT floor efSearch at k (50 vs 100 differ) — our `ef = max(ef, k)`
  floor removes the fast/low-recall end of our curve entirely.
- With heuristic ON our recall-per-ef BEATS faiss (0.947 vs 0.934 at ef=100); the
  remaining gap is per-query speed (~2× slower at matched recall — distance-kernel
  throughput, not graph quality) and build time.
- IVF-LSH collapses at full scale under k=100 (recall ~0.6 where the 30k smoke
  measured 1.0) — its old `memory` pick was never revalidated in this regime.

### Open work on the `memory` cell
- Speed gate is ≤2× faiss ef50 ≈ 0.44ms/query (this machine); our only ≥0.95
  config runs 0.58ms. Closing it needs kernel speed (SIMD in `simd.h`), not params.
- Big memory lever: in sq8 mode with `refine_r=-1` the retained float `data_`
  (~993MB on yahoo) is unused at query time — dropping it post-build would roughly
  halve index RSS. Needs a `fit` option in `src/hnsw.cpp` (heuristic uses `data_`
  during build, so free it after construction, and forbid refine at query time).

## Offline tuning (final-machine procedure)

Fit-time auto-tuning was removed (build time is scored). On the competition machine:

```bash
python3 setup.py build_ext --inplace && mv -f *.so build/
python3 scripts/tune_parameters.py                 # hours; sweeps hnsw + ivf_lsh per dataset
# review scenarios.tuned.yaml + the tuning report, then merge into the
# submission's scenarios.yaml and rebuild the Docker image
```

The tuner fits each build config once and scans all query configs on it, verifies
finalists in fresh processes (clean memory numbers), picks per-scenario winners per
dataset, emits `-private`-twin blocks, and derives a robust cross-dataset `default`.
`--quick` smoke-tests the pipeline; `--from-json` re-selects without re-measuring.

Safety rails (added after review, 2026-08-11): memory-scenario finalists are ranked
by an analytic index-size estimate (measured RSS still decides among them); if every
finalist misses its recall bar on fresh-process re-measurement, the pool widens to
the next feasible configs instead of shipping a below-bar block; configs that ever
return fewer than k ids are disqualified (the harness raises on short results);
sweep children save incrementally so a timeout keeps completed builds, and a dataset
whose sweep fails is excluded from the `default` computation rather than erasing it.

## How to fill a cell

0. Measure the current state of the whole matrix:
   `python3 scripts/run_matrix.py` (full) or `--subset 100000 --queries 200` (quick pass).
   Single cell: `python3 scripts/run_matrix.py --datasets <stem> --scenario <name>`.
1. Local sweep on a subset (`experiments/dense_sweep.py` pattern, or
   `experiments/compare_all_datasets.py` for a 20k-subset cross-dataset pass).
2. Verify the shortlist on the **full** dataset, single-threaded query loop
   (the harness times queries one by one — see `experiments/verify_single_threaded.py`).
3. Encode the winner as a per-dataset block in the bundle's `scenarios.yaml`, keyed by the
   stem, e.g.:
   ```yaml
   scenarios:
     memory:
       default: { ... }
       simplewiki-openai-public:
         index_params: { backend: lsh, n_tables: ..., n_bits: ..., n_clusters: ... }
         query_params: { n_probes: ..., n_probe_clusters: ..., refine_r: ... }
   ```
4. Confirm under the real harness (`orthogonal-competition/run_all.sh` or a single
   `evaluator.py evaluate` call) — recall there is measured with k=100 and the memory
   scenario is scored on **whole-container** peak RSS, not just the index.

## Open items

- `fast` threshold: harness README says ≥0.80, competition brief said 85% — tune for 0.85
  so both readings are safe.
- Only yahoo-minilm has full-dataset measurements; the other 6 rows are unmeasured.
- The `memory` scenario also has a speed gate (≤2× faiss-hnsw efC100/M16/ef50) — check it
  whenever memory params get more aggressive.
- `orthogonal-competition/gp_tuner.py` needs its sklearn imports fixed before use.
