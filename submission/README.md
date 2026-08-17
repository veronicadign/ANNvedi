# ANNvedi — competition submission

Self-contained submission for the Orthogonal k-NN competition: an HNSW index
(C++/pybind11) with SQ8 or per-dimension SQ8 quantization and optional
diversity-heuristic neighbor selection. All parameters are resolved by the
harness from `scenarios.yaml`; nothing is tuned at `fit()` time.

## Build the image

Requires the competition base image (`ann-orthogonal/base:latest`) to be
built first, then:

```bash
docker build -t ann-orthogonal/annvedi submission/   # from the repo root
# or, from inside this directory:
docker build -t ann-orthogonal/annvedi .
```

The build compiles `hnsw_cpp` with `-march=native` — build the image on the
machine that will run it.

## Contents

| file | role |
|---|---|
| `src/hnsw.cpp`, `src/simd.h` | the index: HNSW, sq8 / sq8pd quantization, SIMD L2 kernels |
| `algorithm.py` | harness wrapper (`fit` / `query` / `get_n_distances`), copied to `/app/algorithm.py` |
| `scenarios.yaml` | measured per-dataset parameters (+ `-private` twins and defaults), copied to `/app/scenarios.yaml` |
| `setup.py`, `pyproject.toml` | builds and installs the `hnsw_cpp` module (`pip install .`) |

## Where the parameters come from

Every `scenarios.yaml` block was measured on the full public datasets
(k=100, evaluator-identical recall, all 1000 development queries) with
`scripts/tune_parameters.py` in the development repo; see
`docs/TUNING.md` there for the full measurement matrix and evidence files.
To re-tune (e.g. on new hardware): run the tuner in the development repo and
copy the emitted blocks into this folder's `scenarios.yaml`, then rebuild
the image.
