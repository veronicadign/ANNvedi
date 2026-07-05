#!/usr/bin/env python3
"""
NNS Competition – Evaluation Harness
======================================
This script is provided by the organizers and baked into the base Docker image.
Students do NOT modify this file.

Runs exactly ONE scenario per container invocation.  The evaluator spawns a
fresh container for every (scenario, dataset) pair, so Docker's cgroup memory
accounting gives a clean peak-RSS measurement for each one independently.

Environment variables (set by the evaluator):
  DATASET_PATH   – path to the read-only HDF5 dataset
  RESULTS_PATH   – path where this script writes results.hdf5
  SCENARIO_NAME  – name of the scenario to run (must exist in scenarios.yaml)
  DATASET_NAME   – stem of the dataset filename, used to resolve per-dataset
                   params in scenarios.yaml (e.g. "sift-128-euclidean")
  QUERY_K        – number of neighbors to retrieve (default: 10)

scenarios.yaml format
---------------------
  scenarios:
    <scenario_name>:
      default:                    # fallback for any unlisted dataset
        index_params: {...}
        query_params: {...}
      <dataset_name>:             # dataset-specific override (optional)
        index_params: {...}
        query_params: {...}

Output HDF5 layout (flat – one file per container run):
  /neighbors       int32   (n_queries, k)  – predicted neighbor indices
  /build_time      float64 scalar          – seconds for fit()
  /query_times     float64 (n_queries,)    – per-query wall-clock seconds
  /n_dist_queries  int64   scalar          – distances computed during all queries
  /index_mem_mb    float64 scalar          – index memory in megabytes
"""

import importlib.util
import os
import time
import traceback
import resource

import h5py
import numpy as np
import yaml

SCENARIOS_FILE = "/app/scenarios.yaml"
ALGORITHM_FILE = "/app/algorithm.py"


def load_algorithm_class():
    spec = importlib.util.spec_from_file_location("algorithm", ALGORITHM_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Algorithm"):
        raise ImportError(
            "algorithm.py must define a class named 'Algorithm'. "
            "See the submission template for the required interface."
        )
    cls = module.Algorithm
    for method in ("fit", "query", "get_n_distances"):
        if not callable(getattr(cls, method, None)):
            raise ImportError(
                f"Algorithm is missing a callable method '{method}'. "
                "See the submission template for the required interface."
            )
    return cls


def load_scenario(scenario_name: str, dataset_name: str) -> dict:
    """
    Return the resolved index_params / query_params for the requested
    scenario and dataset.

    Resolution order:
      1. scenarios.<scenario_name>.<dataset_name>  – dataset-specific block
      2. scenarios.<scenario_name>.default          – fallback
      3. Raises ValueError with a descriptive message if neither exists.
    """
    with open(SCENARIOS_FILE) as f:
        raw = yaml.safe_load(f)

    if "scenarios" not in raw or not isinstance(raw["scenarios"], dict):
        raise ValueError("scenarios.yaml must contain a top-level 'scenarios' mapping.")

    if scenario_name not in raw["scenarios"]:
        available = ", ".join(raw["scenarios"])
        raise ValueError(
            f"Scenario {scenario_name!r} not found in scenarios.yaml. "
            f"Available: {available}"
        )

    scenario_cfg = raw["scenarios"][scenario_name] or {}

    if dataset_name in scenario_cfg:
        block  = scenario_cfg[dataset_name]
        source = f"dataset-specific block '{dataset_name}'"
    elif "default" in scenario_cfg:
        block  = scenario_cfg["default"]
        source = "default block"
    else:
        raise ValueError(
            f"Scenario {scenario_name!r} has no entry for dataset "
            f"{dataset_name!r} and no 'default' fallback."
        )

    block = block or {}
    print(f"[harness] Params resolved from {source}")
    return {
        "index_params": dict(block.get("index_params") or {}),
        "query_params": dict(block.get("query_params") or {}),
    }


def main():
    dataset_path  = os.environ["DATASET_PATH"]
    results_path  = os.environ["RESULTS_PATH"]
    scenario_name = os.environ["SCENARIO_NAME"]
    dataset_name  = os.environ["DATASET_NAME"]
    k             = int(os.environ.get("QUERY_K", 10))

    # ------------------------------------------------------------------
    # Load data (untimed)
    # ------------------------------------------------------------------
    print(f"[harness] Loading dataset from {dataset_path} ...")
    with h5py.File(dataset_path, "r") as f:
        train   = f["train"][:]
        queries = f["test"][:]

    n_train, dim = train.shape
    n_queries    = queries.shape[0]
    print(f"[harness] {n_train} train | {n_queries} queries | dim={dim} | k={k}")
    print(f"[harness] Scenario: {scenario_name}  Dataset: {dataset_name}")

    cfg = load_scenario(scenario_name, dataset_name)
    print(f"[harness] index_params: {cfg['index_params']}")
    print(f"[harness] query_params: {cfg['query_params']}")

    AlgorithmClass = load_algorithm_class()
    algo = AlgorithmClass()

    # ------------------------------------------------------------------
    # Phase 1: index build
    # ------------------------------------------------------------------
    start_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("[harness] Starting fit() ...")
    t0 = time.perf_counter()
    algo.fit(train, **cfg["index_params"])
    build_time   = time.perf_counter() - t0
    end_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    index_mem_mb = (end_mem_kb - start_mem_kb) / 1024
    print(f"[harness] fit() done: {build_time:.4f}s memory_mb={index_mem_mb:.4f}Mb")

    # ------------------------------------------------------------------
    # Phase 2: queries, timed individually
    # ------------------------------------------------------------------
    print("[harness] Starting query loop ...")
    neighbors   = np.empty((n_queries, k), dtype=np.int32)
    query_times = np.empty(n_queries,      dtype=np.float64)

    for i in range(n_queries):
        t0 = time.perf_counter()
        result = algo.query(queries[i], k=k, **cfg["query_params"])
        query_times[i] = time.perf_counter() - t0

        result = np.asarray(result, dtype=np.int32)
        if result.shape != (k,):
            raise ValueError(
                f"algo.query() must return a 1-D array of length k={k}, "
                f"got shape {result.shape} on query index {i}."
            )
        neighbors[i] = result

    n_dist_queries = int(algo.get_n_distances())
    total = query_times.sum()
    print(
        f"[harness] queries done: total={total:.4f}s  QPS={n_queries/total:.1f}  "
        f"median={np.median(query_times)*1e3:.3f}ms  "
        f"p99={np.percentile(query_times, 99)*1e3:.3f}ms  "
        f"n_dist_queries={n_dist_queries}"
    )

    # ------------------------------------------------------------------
    # Write results (flat layout – one scenario per file)
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    with h5py.File(results_path, "w") as f:
        f.create_dataset("neighbors",      data=neighbors)
        f.create_dataset("build_time",     data=float(build_time))
        f.create_dataset("query_times",    data=query_times)
        f.create_dataset("n_dist_queries", data=n_dist_queries)
        f.create_dataset("index_mem_mb", data=index_mem_mb)

    print(f"[harness] Results written to {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
