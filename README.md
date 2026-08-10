# ANNvedi — ANN indices for the Orthogonal k-NN competition

C++ (pybind11) approximate-nearest-neighbor indices — HNSW, IVF+multiprobe LSH (SQ8),
simple LSH, multiprobe LSH, exact linear scan — plus the competition harness, submission
bundles, and the tuning experiments behind them.

## The competition in one paragraph

A submission is a **Docker image** (`FROM ann-orthogonal/base:latest`) containing
`/app/algorithm.py` (a class `Algorithm` with `fit(train, **index_params)`,
`query(q, k, **query_params) -> int array of shape (k,)`, `get_n_distances()`) and
`/app/scenarios.yaml`. The evaluator runs **one fresh container per scenario** (network
disabled, 1800 s timeout, k=100) and scores recall against the dataset ground truth.
Scenarios:

| scenario key  | requirement            | what wins                                              |
|---------------|------------------------|--------------------------------------------------------|
| `high_recall` | recall ≥ 0.95          | fastest QPS (also: fastest build, fewest distance comps) |
| `fast`        | recall ≥ 0.80 (per the harness README; plan for 0.85 to be safe) | fastest QPS |
| `memory`      | recall ≥ 0.95          | least container memory, within 2× faiss-hnsw (efC=100, M=16, ef=50) query time |

**Per-scenario × per-dataset approaches** plug in through `scenarios.yaml`: every scenario
has a `default` block plus optional per-dataset override blocks **keyed by the HDF5
filename stem including the suffix** (e.g. `yahoo-minilm-public`). Since `backend` is just
an index_param, a single image can route each (scenario, dataset) cell to a different
algorithm — that is exactly what the root `algorithm.py` facade does.

> ⚠️ Private test sets will be named `*-private`, so overrides keyed `...-public` will NOT
> match them and silently fall back to `default`. Keep `default` strong, or key overrides
> for both stems if the organizers keep train distributions identical.

## Layout

```
algorithm.py, scenarios.yaml   The root submission: multi-backend facade (linear/lsh/hnsw/multiprobe)
Dockerfile, .dockerignore      Root submission image (context trimmed by .dockerignore)
setup.py, src/                 Canonical C++: hnsw, lsh_index_optimized, lsh_index_simple,
                               linear_index, multiprobe_lsh (+ simd.h, thread_pool.h)
linear_ann/, lsh_ann/          Thin Python wrappers used by the facade and experiments
dataset/                       The 7 public HDF5 datasets (gitignored)
competitors/                   Self-contained submission bundles, one dir per approach:
                               ivf_lsh (newest: per-dim SQ8 C++ fork + GP auto-tuning),
                               ANNvedi-demo (HNSW sq8, as submitted Jul 5), lsh_optimized,
                               lsh_simple, linear (exact baseline)
orthogonal-competition/        Official harness (upstream snapshot) + our drivers:
                               run_all.sh, competitors.txt, datasets.txt, gp_tuner.py
tests/                         Test suite: python3 tests/run_tests.py   (run from repo root)
experiments/                   Kept tuning/benchmark scripts + results/dense_sweep_results.json
                               (27,600 measured IVF-LSH configs on yahoo-minilm 100k)
notebooks/                     dataset_analysis (per-dataset stats → tuning implications),
                               competitor_evaluation (results.db dashboard)
scripts/                       data_download.py, update_on_aws.sh
docs/                          TUNING.md (scenario × dataset decision matrix), walkthrough.md
archive/                       Frozen history: superseded bundles (HNSW, LSH_Cpp,
                               MultiProbe_LSH), one-off scratch scripts, old logs. See its README.
build/                         Compiled .so files (gitignored; tests import from here)
```

## Workflows

### Build & test locally
```bash
pip install -r requirements.txt
python3 setup.py build_ext --inplace && mv -f *.so build/
python3 tests/run_tests.py          # from repo root; 16 tests
```
Conda note: if imports fail with `undefined symbol: __cxa_call_terminate`, the conda env's
libstdc++ (gcc-11 era) is older than the system compiler. Either prefix commands with
`LD_PRELOAD=/usr/lib64/libstdc++.so.6` or upgrade the runtime once:
`conda install -c conda-forge "libstdcxx-ng>=13"`.

### Evaluate bundles under the real harness (Docker)
```bash
cd orthogonal-competition
just build-all-containers          # builds ann-orthogonal/base FIRST (required by all bundles)
./run_all.sh                       # competitors.txt × datasets.txt → results.db
```
Inspect results with `notebooks/competitor_evaluation.ipynb`.
Note: `datasets.txt` currently lists 6 of the 7 datasets — `gooaq-distilroberta` is missing.

### Tune a (scenario, dataset) cell
1. Read `docs/TUNING.md` for what is already known (chosen configs, dataset properties).
2. Sweep locally (`experiments/dense_sweep.py`, `experiments/compare_all_datasets.py`) or
   GP-tune (`orthogonal-competition/gp_tuner.py`; `competitors/ivf_lsh/algorithm.py` does it
   at fit time inside the container).
3. Record the winner as a per-dataset override block in the bundle's `scenarios.yaml`.
4. Re-run the harness on that dataset and check `results.db`.

Convention: **all Python scripts run from the repo root** (`python3 experiments/xxx.py`);
notebooks run from `notebooks/`; `run_all.sh` runs from `orthogonal-competition/`.

## Traps (read before touching the C++)

- **Five `simd.h` variants** share one filename with *incompatible*
  `compute_l2_distance_quantized` signatures (root `src/`, `competitors/ivf_lsh/src/`
  per-dimension scale, `competitors/lsh_optimized|linear/src/` 5-arg shifted,
  `competitors/ANNvedi-demo/src/` AVX-512-only, `archive/HNSW/src/`). **Never deduplicate
  bundle copies by filename.** In `src/simd.h` the shifted variant is a separate function,
  `compute_l2_distance_quantized_shifted` — see the comment at `src/lsh_index_optimized.cpp:109`.
- **`competitors/ivf_lsh/src/` is not a copy of `src/`** — it is a newer divergent fork
  (per-dimension SQ8 scales, mean-centered projections, module `ivf_lsh_cpp`). The best LSH
  code lives there, not in `src/`.
- **Module-name collisions**: root `setup.py` and the bundles build the same module names
  (`linear_ann_cpp`, `lsh_cpp_optimized`, …). Don't `pip install` the root package and a
  bundle into the same environment; bundles are meant to be built inside their own Docker image.
- `orthogonal-competition/gp_tuner.py` has a known bug: it uses
  `GaussianProcessRegressor/Matern/WhiteKernel` without importing them from
  `sklearn.gaussian_process` (NameError once GP fitting starts).
- Bundle Dockerfiles `COPY . /app/source` — keep bundle dirs free of large files.
