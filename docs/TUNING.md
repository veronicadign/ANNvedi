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
- IVF-LSH loses to HNSW on every dataset at full scale (complete sweep 2026-08-12,
  `experiments/results/ivf_lsh_full_scale_20260812.json`). Root cause of the earlier
  "recall ~0.6" reading: the C++ caps candidates at max(2000, 4×refine_r)
  (`lsh_index_optimized.cpp:358`), so small refine_r values silently scan ≤2000
  points. With refine_r up to 2500 and 1024 clusters the recall ceiling lifts to
  0.83–0.96, but the price is 1.8–13ms/query: the ≥0.95 bar is reached only on
  landmark (0.9612 @ 3.17ms — HNSW: 0.9643 @ 0.46ms), the fast bar costs 3–9× HNSW's
  latency everywhere, gooaq misses both bars outright (best 0.8285), and builds run
  214–2139s (vs 83–387s for HNSW M8). No scenario/prize dimension favors it.

### Per-dimension SQ8 (`mode: sq8pd`, added 2026-08-12)

Ported the IVF-LSH fork's per-dimension quantization into `hnsw_cpp` (each dim
gets its own [min,max]; query pre-shifted once per search; kernel in
`src/simd.h`). Query/build cost identical to `sq8`. Measured ef-for-ef on full
datasets (M16 and M8 heurON builds): **a per-dataset lever, not a universal
win** — although per-element quantization error is strictly smaller, the graph
is also *built* with these distances, and graph-quality shifts dominate:

- gooaq: better on both measured cells (fast M8/ef100 0.9153 vs 0.9125, memory
  M8/ef200 0.9579 vs 0.9554 — was the thinnest margin in the matrix) → adopted
  in `scenarios.yaml` for those two cells.
- yahoo / celeba / agnews: neutral (±0.001-0.005, largest gains only at low ef).
- simplewiki: WORSE (M8/ef250 0.9531 vs 0.9563; M8/ef120 −0.006) → kept on sq8.

Sweep it alongside sq8 in future tuner runs (add to the focused grid when
retuning on the competition machine).

### Locality reorder & SQ4 (measured 2026-08-12, full yahoo, interleaved A/B)

- **BFS locality reorder — ADOPTED** (`reorder: true` in every hnsw block;
  implemented in submission/src/hnsw.cpp): after the build, nodes are
  renumbered in BFS order over layer 0 and all payloads permuted, so the
  query beam's hops land on nearby cache lines/pages instead of uniformly
  random ones. **+12.7% qps** (0.523 → 0.457ms, faster in every interleaved
  round), recall identical by construction, build-time cost ≈ 0 (244s vs
  245s once ordering bias was controlled). Query() maps results back to
  original ids. NOTE: not yet ported to the dev src/hnsw.cpp — dev-side
  runs slightly underestimate the submission until it is.
- **SQ4 per-dim mode (`sq4pd`) — implemented, NOT enabled**: half the cache
  lines per eval and ~18% cheaper per-eval as predicted, but the coarser
  beam needs ~1.5x the ef to clear the bar: at the 0.955 operating point it
  measures 0.584ms vs sq8's 0.474ms. Kept as a tuner lever; the promising
  target is simplewiki (3072d = 48 lines/eval, bandwidth-dominated), worth
  sweeping on the competition machine.
- Context for both: perf profiling showed the query is memory-latency-bound
  (IPC 0.72, 70% LLC miss rate, ~6,900 DRAM misses/query; branch mispredicts
  only ~4% of cycles; AVX-512 16-wide confirmed; software prefetch already
  saturates line-fill buffers at +2 lookahead).

### Final-round hardware: AWS g7.2xlarge (confirmed 2026-08-13)

8 vCPU (4 cores + HT) on custom 6th-gen Intel Xeon Scalable (Granite Rapids
family → native full-width AVX-512: our kernels run at full width, no AVX2
fallback needed), 32 GiB RAM, 600 GB NVMe. GPU: 1x NVIDIA RTX PRO 4500
Blackwell, 32 GB GDDR7, ~800 GB/s. Organizers confirmed one-query-at-a-time
(no batching), so any GPU brute-force rival is VRAM-bandwidth-bound:
qps <= 800 GB/s / dataset_bytes per query. With int8 codes that beats our
HNSW on the small datasets (yahoo/celeba), roughly ties the mid ones, loses
the biggest (gooaq) and loses Paperone outright (max distance count) while
trivially winning Marie Kondo (near-zero build). Whether GPU use is legal
and whether the evaluator passes --gpus to containers are open questions
for the organizers — both must be yes before it matters. Race-day checks:
lscpu (confirm AVX-512), rebuild image on the box (-march=native), rerun
the tuner, one reorder A/B.

### Rust port experiment (2026-08-13): C++ stays

Full port of the submission index to Rust (rust/hnsw_rs: pyo3 + std::arch
AVX-512 kernels, same algorithm/locking/reorder). Functionally equivalent
(recall and distance counts match within build nondeterminism on all three
modes). Performance, interleaved full-yahoo A/B at the shipping config:
queries C++ 0.500ms vs Rust 0.524ms (~4.5% slower, consistent every round —
plausibly bounds-checking in the beam hot loop); builds 327s vs 398s
(sequential measurement, thermal ordering bias makes the true gap <= ~18%).
Closing the gap would need get_unchecked-style unsafe throughout the hot
path, i.e. the same code with fewer guarantees. No reason to switch.

### Measured on the real finals hardware (g7e.2xlarge, 2026-08-19)

Box: Xeon Platinum 8559C 8 vCPU (AVX-512 ✓), 64 GiB, RTX PRO 6000 Blackwell
96 GB (driver 610.57, CUDA 13.2), 1.7 TB instance NVMe. Full yahoo, 1000
queries, k=100, one query at a time:

| approach | recall | ms | qps | build |
|---|---|---|---|---|
| GPU filter R=150 | 1.0000 | 0.368 | 2718 | ~1s upload (+~2s quantize) |
| GPU filter R=500 | 1.0000 | 0.526 | 1902 | " |
| ours hnsw ef=120 (sq8+heur+reorder) | 0.9612 | 0.535 | 1868 | 92s @ 7.8x |
| faiss ef=200 / 100 / 50 | .975/.935/.850 | .500/.291/.196 | 2001/3431/5115 | 36s |

GPU stage split (R=150): scan 0.209ms (247 MB @ ~1.2 TB/s effective), CUB
top-R 0.063ms, CPU rerank 0.078ms.

All-datasets GPU filter (R=150, measured on the g7e; ours = local pick
latency x1.15 box factor, yahoo measured directly):

| dataset | GPU qps (recall) | ours hnsw qps (est) | high_recall verdict |
|---|---|---|---|
| yahoo | 2718 (1.0000) | 1868 (measured) | GPU +45% |
| celeba | 1637 (0.9984) | ~1040 | GPU +57% |
| imagenet | 1278 (1.0000) | ~1260 | tie |
| landmark | 1259 (1.0000) | ~1875 | ours +49% |
| agnews | 920 (0.9971) | ~1290 | ours +40% |
| simplewiki | 823 (0.9998) | ~880 | ~tie |
| gooaq | 723 (1.0000) | ~1035 | ours +43% |

Scan efficiency 850-1180 GB/s across sizes. GPU recall is 0.997-1.0
everywhere (zero private-query margin risk), build is seconds on every
dataset (Marie Kondo sweep), and distance count is 150/query (Paperone
sweep, ~10-16x below the field). The strongest final-form submission if
GPUs are ruled legal: ONE image with both backends, chosen per dataset in
scenarios.yaml (GPU: yahoo/celeba/imagenet; hnsw: landmark/agnews/gooaq/
simplewiki), pending the two organizer gates. On yahoo the GPU filter beats every
≥0.95 entrant at recall 1.0 with a seconds-long build and 150 full-Euclidean
distances/query — best Sherlock+Marie Kondo+Paperone entry IF GPUs are
ruled legal and the evaluator passes --gpus. CPU notes: our build scales to
7.8x on the 8 vCPUs (92s vs 245-327s local); our query is ~15% slower than
local (server DRAM latency vs our random-access reads) while faiss gains
~15% (bandwidth-bound streaming) — rerun the CPU tuner on this box.

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
