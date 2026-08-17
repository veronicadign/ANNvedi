# gpu_filter — GPU quantized filter + CPU exact rerank

Per query (one at a time, per the rules): the GPU brute-force scans
per-dimension SQ8 codes (1/4 the float bytes, dim-major for coalescing),
keeps the top-`r` candidates, and the CPU verifies exactly those `r` with
full-precision float distances from host RAM. Only those `r` computations
are full Euclidean distances, so the reported distance count is `r` per
query — honest per the Paperone rule and an order of magnitude below any
graph traversal.

Validated on full yahoo-minilm (identical math, CPU simulation): the
quantized top-150 always contains the true top-100 → recall 1.0000.

## Build

```bash
docker build -t ann-orthogonal/gpu-filter competitors/gpu_filter/
```

Builds on a GPU-less machine: the kernel is compiled to `compute_80` PTX and
JIT-compiled by the driver on first run (works on Blackwell).

## Run requirements

The evaluator must start the container with `--gpus` (NVIDIA container
toolkit). Without a visible device, `fit()` raises
`no CUDA device visible ... was the container started with --gpus?`.

## Open items

- GPU legality + `--gpus` at the evaluator: organizer questions, unanswered.
- Top-R uses a full CUB radix sort (simple v1, ~0.2-0.3 ms); a block-local
  selector can roughly halve total query time if this contestant matters.
- `r` values in scenarios.yaml are yahoo-validated with wide margins;
  re-validate per dataset (experiments/cuda + numpy simulation in the dev
  repo) before racing them.
