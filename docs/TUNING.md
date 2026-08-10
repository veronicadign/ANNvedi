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

## How to fill a cell

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
