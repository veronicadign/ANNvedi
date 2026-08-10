# Archive — frozen history, nothing here is on the live path

- `HNSW/`, `LSH_Cpp/`, `MultiProbe_LSH/` — superseded submission bundles (Jul 4 era).
  Unique bits worth knowing about:
  - `HNSW/scenarios.yaml` — only record of tuned params for the experimental HNSW hybrid
    modes (lsh / lsh_sq8 / lsh_float).
  - `MultiProbe_LSH/src/multiprobe_lsh.cpp` — original of the standalone multiprobe index;
    a copy now lives in `src/` and is built by the root `setup.py` as `multiprobe_lsh_cpp`.
  - `LSH_Cpp/` — ancestor of `src/lsh_index_simple.cpp` (renamed class/module only).
- `scratch/` — one-off sweep/verify scripts superseded by what was kept in `experiments/`.
  Most import the pre-rename `lsh_ann`/`linear_ann` wrappers and hardcode old paths; they
  are kept for provenance, not for running.
- `logs/` — `res1.txt` (results.db dump, Jul 9), `test_out.txt` / `test_out_ec2_2.txt`
  (old local/EC2 test runs).
- `run_hnsw.py`, `test_run.py` — root-era demo/smoke scripts for deleted module layouts.
- `colab_notebook.ipynb` — early Colab bootstrap (self-contained, /content paths).
- `ANNvedi-demo.zip` — byte-exact copy of the bundle submitted on Jul 5 ("send demo").
  Do not regenerate; it is the record of what was actually sent.
