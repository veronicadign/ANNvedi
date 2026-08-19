# ANNvedi-competitor — self-calibrating final contestant

One image, zero per-dataset tuning. At `fit()` the index itself (rule 3
allows parameters "impostati al volo dall'indice stesso"):

1. **Selects the backend**: GPU quantized-filter + CPU exact rerank when a
   CUDA device is visible (`--gpus`); otherwise the tuned CPU HNSW
   (sq8 + diversity heuristic + BFS locality reorder). Exactly one backend
   is built — build time is a scored metric.
2. **Calibrates itself**: computes exact ground truth for a 128-point
   sample of train pseudo-queries (chunked BLAS scan, a few seconds), then
   walks a ladder — `r` ∈ {150…1200} on GPU, `ef` ∈ {100…500} on HNSW —
   until sample recall reaches the scenario bar **+ 1.5% safety margin**
   (high_recall/memory → 0.965, fast → 0.865). Scenario and k come from the
   harness env (`SCENARIO_NAME`, `QUERY_K`).

Measured context (g7e.2xlarge): the GPU path lands at r=150 with recall
0.997–1.0 on all seven public datasets (2,718 qps on yahoo), builds in
seconds, and reports r full-Euclidean distances per query. The HNSW
fallback matches the tuned submission.

## Build

```bash
docker build -t ann-orthogonal/annvedi-competitor ANNvedi-competitor/
```

Builds without a GPU (kernel compiled to compute_80 PTX, JIT on target).

## Notes

- Calibration adds ~5–15 s to fit (sample GT dominates) — bounded and flat
  per dataset, unlike parameter mis-tuning which costs recall or a prize.
- Pseudo-queries are train points, slightly optimistic vs unseen queries —
  covered by the +1.5% margin (race queries share the train distribution).
- Requires the evaluator to pass `--gpus` for the GPU path; degrades to the
  CPU path cleanly otherwise (message in the harness log).
