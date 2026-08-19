#!/usr/bin/env python3
"""Build-time race (Marie Kondo selector).

Per dataset: build the GPU filter and a ladder of ever-lighter HNSW build
configs, check each still qualifies (recall >= BAR at some ef on the file
ground truth), and pick the fastest-building qualifier. Query latency is
reported alongside every entry because the same run feeds Sherlock (qps):
the "better build" is the fastest QUALIFYING build — the printout shows what
that choice costs in query speed.

Run on the g7e (needs submission/hnsw_cpp built and
competitors/gpu_filter/libgpufilter.so compiled with nvcc -arch=native):
    python3 experiments/build_race.py [dataset-stem ...]
Writes experiments/results/build_race.json
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
import hnsw_cpp  # noqa: E402

sys.path.insert(0, str(ROOT / "competitors" / "gpu_filter"))
try:
    import algorithm as gpu_algo  # loads libgpufilter.so; needs a CUDA device
    GPU_OK = True
except OSError as e:
    print(f"GPU backend unavailable ({e}) — racing HNSW configs only")
    GPU_OK = False

K = 100
BAR = 0.955                 # high_recall bar + safety margin
GPU_R = 150                 # validated: recall 0.997-1.0 on all datasets
HNSW_BUILDS = [(8, 64), (8, 48), (6, 48), (6, 40)]   # lighter and lighter
EF_LADDER = [100, 120, 170, 250, 300, 400]

ALL = ["yahoo-minilm", "celeba-resnet", "landmark-nomic", "imagenet-clip",
       "agnews-mxbai", "simplewiki-openai", "gooaq-distilroberta"]


def measure(query_fn, train, queries, gt):
    """Evaluator-style recall + mean query ms (all 1000 queries)."""
    recs, t_total = [], 0.0
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        r = np.asarray(query_fn(q, K))
        t_total += time.perf_counter() - t0
        d = np.sort(np.linalg.norm(train[r] - q, axis=1))[:K]
        recs.append(np.mean(d <= gt[i][K - 1] + 1e-6))
    return float(np.mean(recs)), t_total * 1000 / len(queries)


def main():
    datasets = sys.argv[1:] or ALL
    out = {}
    for ds in datasets:
        with h5py.File(ROOT / "dataset" / f"{ds}-public.hdf5", "r") as f:
            train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
            queries = np.asarray(f["test"][:], dtype=np.float32)
            gt = np.asarray(f["distances"][:, :K], dtype=np.float32)
        print(f"\n=== {ds} ({train.shape[0]}x{train.shape[1]}) ===", flush=True)
        entries = []

        if GPU_OK:
            a = gpu_algo.Algorithm()
            t0 = time.perf_counter()
            a.fit(train)
            build = time.perf_counter() - t0
            rec, ms = measure(lambda q, k: a.query(q, k, r=GPU_R), train, queries, gt)
            ok = rec >= BAR
            entries.append(dict(name=f"gpu_filter r={GPU_R}", build_s=round(build, 1),
                                recall=round(rec, 4), query_ms=round(ms, 3), ef=None, ok=ok))
            print(f"  gpu_filter      build={build:6.1f}s recall={rec:.4f} "
                  f"query={ms:.3f}ms {'OK' if ok else 'BELOW BAR'}", flush=True)
            del a

        for M, efc in HNSW_BUILDS:
            idx = hnsw_cpp.HNSWIndex()
            t0 = time.perf_counter()
            idx.fit(train, M, efc, "sq8", True, True)
            build = time.perf_counter() - t0
            chosen = None
            for ef in EF_LADDER:
                rec, ms = measure(lambda q, k: idx.query(q, k, ef), train, queries, gt)
                if rec >= BAR:
                    chosen = (ef, rec, ms)
                    break
            if chosen:
                ef, rec, ms = chosen
                entries.append(dict(name=f"hnsw M{M}/efC{efc}", build_s=round(build, 1),
                                    recall=round(rec, 4), query_ms=round(ms, 3), ef=ef, ok=True))
                print(f"  hnsw M{M}/efC{efc:3d}  build={build:6.1f}s recall={rec:.4f} "
                      f"query={ms:.3f}ms (ef={ef}) OK", flush=True)
            else:
                entries.append(dict(name=f"hnsw M{M}/efC{efc}", build_s=round(build, 1),
                                    recall=round(rec, 4), query_ms=None, ef=None, ok=False))
                print(f"  hnsw M{M}/efC{efc:3d}  build={build:6.1f}s recall={rec:.4f} "
                      f"@ef{EF_LADDER[-1]} BELOW BAR — disqualified", flush=True)
            del idx

        ok_entries = [e for e in entries if e["ok"]]
        winner = min(ok_entries, key=lambda e: e["build_s"]) if ok_entries else None
        out[ds] = dict(entries=entries, winner=winner["name"] if winner else None)
        if winner:
            print(f"  >>> fastest qualifying build: {winner['name']} "
                  f"({winner['build_s']}s, query {winner['query_ms']}ms)", flush=True)

    res = ROOT / "experiments" / "results" / "build_race.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, indent=1))
    print(f"\nsummary -> {res}")
    print("\n=== WINNERS (fastest build meeting recall >= %.3f) ===" % BAR)
    for ds, r in out.items():
        print(f"  {ds:22s} {r['winner']}")


if __name__ == "__main__":
    main()
