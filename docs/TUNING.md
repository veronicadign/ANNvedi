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

| scenario | current default (root `scenarios.yaml`) | verified alternatives / evidence |
|---|---|---|
| `high_recall` | HNSW sq8, M=16, efC=100, ef=80 | IVF-LSH (20,10,256, probe_cl=16, probes=10) full-dataset verified in `experiments/verify_single_threaded.py`; shortlist in `verify_final_95_sweep.py` |
| `fast` | HNSW sq8, M=8, efC=64, ef=30 | IVF-LSH (9,7,256, probe_cl=6, probes=6) verified in `verify_single_threaded.py`; shortlist in `verify_sweep_85.py` |
| `memory` | IVF-LSH backend=lsh (25,10,256, probes=12, probe_cl=24, refine_r=100) | `competitors/ivf_lsh` auto-tunes at fit time (GP, reads `SCENARIO_NAME`); SQ8 storage is the memory lever |

Evidence base:
- `experiments/results/dense_sweep_results.json` — 27,600 measured IVF-LSH configs
  (params + recall + query ms + fit s), yahoo-minilm 100k. Selector: `experiments/print_sweep_results.py`.
- `archive/HNSW/scenarios.yaml` — the only record of tuned params for HNSW's experimental
  hybrid modes (lsh / lsh_sq8 / lsh_float).
- `orthogonal-competition/results.db` — 104 historical harness runs (through Jul 8) incl.
  faiss-hnsw baseline numbers.
- HNSW mode profiling (float / sq8 / lsh at ef 40/120/250): `experiments/profile_hnsw.py`.

### Backend picks per cell (hypotheses to validate)

| dataset ↓ / scenario → | high_recall (≥.95, speed) | fast (≥.80–.85, speed) | memory (≥.95, min RAM) |
|---|---|---|---|
| agnews-mxbai        | HNSW sq8 (check recall: non-normalized!) | HNSW sq8 low-ef | ivf_lsh SQ8 |
| celeba-resnet       | HNSW sq8 (same caveat) | HNSW sq8 low-ef | ivf_lsh SQ8 |
| gooaq-distilroberta | HNSW sq8 | HNSW sq8 low-ef | ivf_lsh SQ8 |
| imagenet-clip       | HNSW sq8 | HNSW sq8 low-ef | ivf_lsh SQ8 |
| landmark-nomic      | HNSW sq8, low ef should hit .95 | HNSW very low ef | ivf_lsh SQ8 |
| simplewiki-openai   | HNSW needs higher M/efC here | HNSW sq8 | **critical cell**: 3072-dim → SQ8 at minimum, consider stronger compression |
| yahoo-minilm        | HNSW sq8 M16/ef80 ✓ or IVF-LSH (20,10,256,16,10) ✓ | IVF-LSH (9,7,256,6,6) ✓ or HNSW M8/ef30 | IVF-LSH (25,10,256,12/24,r100) |

✓ = measured on the full dataset. Everything else is extrapolation — measure before trusting.

## The k=100 regime (why the old low-ef configs are dead)

The evaluator queries with **k=100**, and `hnsw_cpp` floors `ef` at `k` — so every
query runs a ≥100-wide beam no matter what `ef` the scenario asks for. Consequences:

- The `fast` scenario's `ef: 30` is silently `ef: 100`; low-ef tuning knowledge from
  the k∈{5,10} era does not transfer. The remaining HNSW speed levers are **M**
  (graph degree), build mode, and the backend choice itself.
- Recall saturates: at 100k subsets, M=16/ef=100 scores recall 1.0000 on every
  dataset tried. The bar (0.95) is likely reachable even at full scale with modest
  params — measure, don't overspend.

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
(more distance evals) and slow the build — so the **default is OFF**; it can pay
off only where recall misses the bar at ef=100 (the offline tuner sweeps both).

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
