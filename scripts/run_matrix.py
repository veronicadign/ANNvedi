#!/usr/bin/env python3
"""
Run the full competition matrix locally: every dataset x every scenario.

Mirrors the official harness (orthogonal-competition/harness.py) and evaluator
(orthogonal-competition/evaluator.py) semantics without Docker:
  - params resolved from scenarios.yaml: dataset-stem block, else 'default'
  - a FRESH process per (dataset, scenario) cell (like one container per
    scenario): clean ru_maxrss for index memory, and a crashing cell does not
    kill the rest of the matrix
  - fit() timed once; queries timed individually, single-threaded, k=100
  - recall computed with the evaluator's distance-threshold definition
  - per-cell timeout (default 1800 s, same as the evaluator)

Usage (from anywhere; paths are repo-root anchored):
  python3 scripts/run_matrix.py                          # all 7 datasets x all scenarios
  python3 scripts/run_matrix.py --subset 100000 --queries 200   # fast iteration
  python3 scripts/run_matrix.py --datasets yahoo-minilm-public,agnews-mxbai-public
  python3 scripts/run_matrix.py --scenario memory
  python3 scripts/run_matrix.py --algorithm competitors/ivf_lsh/algorithm.py \
          --scenarios competitors/ivf_lsh/scenarios.yaml
  python3 scripts/run_matrix.py --list                   # show the resolved matrix and exit

Notes:
  - With --subset N the file's ground truth is invalid, so exact ground truth
    is recomputed with the linear index (train[:N] only).
  - index_mem_mb is the harness's ru_maxrss delta around fit(). The official
    memory-scenario score is the container's peak RSS (whole process),
    measured by the evaluator — validate winners with run_all.sh.
  - The 'fast' recall bar is set to 0.85 (competition brief); the harness
    README says 0.80 — both are reported.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = REPO_ROOT / "dataset"
RESULTS_DIR = REPO_ROOT / "experiments" / "results"

# recall bar per scenario: (target, harness_minimum)
RECALL_BARS = {
    "high_recall": (0.95, 0.95),
    "fast": (0.85, 0.80),
    "memory": (0.95, 0.95),
}


def _ensure_native_modules_importable():
    """Make build/ importable and self-heal the conda libstdc++ mismatch.

    On this machine the conda env's libstdc++ (gcc-11 era) is older than the
    system compiler, so freshly built extensions fail with
    'undefined symbol: __cxa_call_terminate' unless the system libstdc++ is
    preloaded. Re-exec once with LD_PRELOAD set if we hit exactly that error.
    """
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "build"))
    try:
        import linear_ann_cpp  # noqa: F401  (probe import)
    except ImportError as e:
        sys_libstdcxx = "/usr/lib64/libstdc++.so.6"
        if (
            "__cxa_call_terminate" in str(e)
            and os.path.exists(sys_libstdcxx)
            and os.environ.get("_ANN_MATRIX_REEXEC") != "1"
        ):
            env = dict(os.environ)
            env["LD_PRELOAD"] = f"{sys_libstdcxx}:{env.get('LD_PRELOAD', '')}".rstrip(":")
            env["_ANN_MATRIX_REEXEC"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, env)
        raise


# ---------------------------------------------------------------------------
# Evaluator-identical recall (orthogonal-competition/evaluator.py)
# ---------------------------------------------------------------------------

def recalls(true_distances, predicted_distances, k):
    import numpy as np

    def compute_recall(td, pd):
        pd = pd[:k]
        threshold = td[k - 1]
        return np.mean(pd <= threshold)

    return np.array(
        [compute_recall(true_distances[i], predicted_distances[i]) for i in range(true_distances.shape[0])]
    )


# ---------------------------------------------------------------------------
# scenarios.yaml resolution (harness-identical, incl. the stem-key semantics)
# ---------------------------------------------------------------------------

def load_scenarios_config(scenarios_path):
    import yaml

    with open(scenarios_path) as f:
        raw = yaml.safe_load(f)
    if "scenarios" not in raw or not isinstance(raw["scenarios"], dict):
        raise ValueError(f"{scenarios_path}: must contain a top-level 'scenarios' mapping.")
    return raw["scenarios"]


def resolve_params(scenario_cfg, dataset_stem):
    scenario_cfg = scenario_cfg or {}
    if dataset_stem in scenario_cfg:
        block, source = scenario_cfg[dataset_stem], f"override '{dataset_stem}'"
    elif "default" in scenario_cfg:
        block, source = scenario_cfg["default"], "default"
    else:
        raise ValueError(f"no block for dataset {dataset_stem!r} and no 'default'")
    block = block or {}
    return (
        dict(block.get("index_params") or {}),
        dict(block.get("query_params") or {}),
        source,
    )


def warn_near_miss_keys(scenarios, dataset_stems):
    """Per-dataset keys only match the full HDF5 stem (e.g. 'yahoo-minilm-public').
    Warn about blocks that look like dataset overrides but match no stem."""
    for scen, cfg in scenarios.items():
        for key in (cfg or {}):
            if key != "default" and key not in dataset_stems:
                print(
                    f"  WARNING: scenarios.{scen}.{key!r} matches no dataset stem "
                    f"{sorted(dataset_stems)} - it will never be used (keys must "
                    f"include the '-public' suffix)."
                )


# ---------------------------------------------------------------------------
# One cell = one (dataset, scenario), executed in a fresh subprocess
# ---------------------------------------------------------------------------

def run_cell_child(spec_path):
    """Child-process entry: run one cell, write result JSON to spec['out']."""
    _ensure_native_modules_importable()
    import resource

    import h5py
    import importlib.util
    import numpy as np

    with open(spec_path) as f:
        spec = json.load(f)
    out = {"dataset": spec["dataset_stem"], "scenario": spec["scenario"], "status": "error"}

    def finish():
        with open(spec["out"], "w") as f:
            json.dump(out, f)

    try:
        k = spec["k"]
        with h5py.File(spec["dataset_path"], "r") as f:
            train = f["train"][:]
            queries = f["test"][:]
            file_true_dist = f["distances"][:] if "distances" in f else None

        if spec["subset"]:
            train = train[: spec["subset"]]
            file_true_dist = None  # invalidated by subsetting
        if spec["n_queries"]:
            queries = queries[: spec["n_queries"]]

        # Ground truth: file distances when valid, else exact recompute.
        if file_true_dist is not None and file_true_dist.shape[1] >= k:
            true_dist = file_true_dist[: queries.shape[0], :k]
        else:
            from linear_ann.wrapper import LinearANN

            gt_index = LinearANN()
            gt_index.fit(train)
            idx = np.array([gt_index.query(q, k) for q in queries])
            true_dist = np.array(
                [np.sort(np.linalg.norm(train[idx[i]] - queries[i].astype(np.float32), axis=1)) for i in range(len(queries))]
            )

        scenarios = load_scenarios_config(spec["scenarios_path"])
        index_params, query_params, source = resolve_params(scenarios[spec["scenario"]], spec["dataset_stem"])

        # Load the Algorithm class exactly like the harness does.
        spec_mod = importlib.util.spec_from_file_location("algorithm_under_test", spec["algorithm_path"])
        module = importlib.util.module_from_spec(spec_mod)
        # env vars some algorithms use (competitors/ivf_lsh reads SCENARIO_NAME)
        os.environ["SCENARIO_NAME"] = spec["scenario"]
        os.environ["DATASET_NAME"] = spec["dataset_stem"]
        os.environ["QUERY_K"] = str(k)
        spec_mod.loader.exec_module(module)
        algo = module.Algorithm()

        start_mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        algo.fit(train, **index_params)
        build_time = time.perf_counter() - t0
        index_mem_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - start_mem_kb) / 1024

        n_q = queries.shape[0]
        neighbors = np.empty((n_q, k), dtype=np.int32)
        qtimes = np.empty(n_q, dtype=np.float64)
        for i in range(n_q):
            t0 = time.perf_counter()
            res = algo.query(queries[i], k=k, **query_params)
            qtimes[i] = time.perf_counter() - t0
            res = np.asarray(res, dtype=np.int32)
            if res.shape != (k,):
                raise ValueError(f"query() must return shape ({k},), got {res.shape} at query {i}")
            neighbors[i] = res

        pred_dist = np.array(
            [np.linalg.norm(train[neighbors[i]] - queries[i].astype(np.float32), axis=1) for i in range(n_q)]
        )
        all_recalls = recalls(true_dist, pred_dist, k)
        total = float(qtimes.sum())

        out.update(
            status="ok",
            params_source=source,
            index_params=index_params,
            query_params=query_params,
            n_train=int(train.shape[0]),
            n_queries=int(n_q),
            k=k,
            avg_recall=float(all_recalls.mean()),
            build_time_s=round(build_time, 4),
            total_query_time_s=round(total, 4),
            qps=round(n_q / total, 1) if total > 0 else None,
            median_query_ms=round(float(np.median(qtimes)) * 1e3, 3),
            p99_query_ms=round(float(np.percentile(qtimes, 99)) * 1e3, 3),
            index_mem_mb=round(index_mem_mb, 1),
            n_dist_queries=int(algo.get_n_distances()),
        )
    except Exception as e:  # noqa: BLE001 - report, parent continues the matrix
        import traceback

        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()
    finish()


def run_cell_parent(spec, timeout):
    with tempfile.TemporaryDirectory(prefix="ann-matrix-") as td:
        spec = dict(spec, out=os.path.join(td, "result.json"))
        spec_file = os.path.join(td, "spec.json")
        with open(spec_file, "w") as f:
            json.dump(spec, f)
        cmd = [sys.executable, os.path.abspath(__file__), "--_run-cell", spec_file]
        try:
            proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, cwd=REPO_ROOT)
        except subprocess.TimeoutExpired:
            return {
                "dataset": spec["dataset_stem"],
                "scenario": spec["scenario"],
                "status": "timeout",
                "error": f"exceeded {timeout}s (evaluator limit is 1800s per scenario)",
            }
        if os.path.exists(spec["out"]):
            with open(spec["out"]) as f:
                return json.load(f)
        return {
            "dataset": spec["dataset_stem"],
            "scenario": spec["scenario"],
            "status": "crashed",
            "error": f"child died (exit {proc.returncode}); stderr tail: {proc.stderr[-800:]}",
        }


# ---------------------------------------------------------------------------
# Matrix orchestration
# ---------------------------------------------------------------------------

def cell_summary(r):
    if r["status"] != "ok":
        return f"{r['status'].upper()}: {r.get('error', '')[:120]}"
    target, harness_min = RECALL_BARS.get(r["scenario"], (0.95, 0.95))
    rec = r["avg_recall"]
    mark = "PASS" if rec >= target else ("pass@harness-0.80" if rec >= harness_min else "FAIL")
    return (
        f"recall={rec:.4f} [{mark}]  qps={r['qps']}  build={r['build_time_s']}s  "
        f"mem={r['index_mem_mb']}MB  med={r['median_query_ms']}ms p99={r['p99_query_ms']}ms  "
        f"(params: {r['params_source']})"
    )


def print_table(results, scenarios):
    by = {(r["dataset"], r["scenario"]): r for r in results}
    datasets = sorted({r["dataset"] for r in results})
    col_w = 34
    header = "dataset".ljust(28) + "".join(s.ljust(col_w) for s in scenarios)
    print("\n" + header)
    print("-" * len(header))
    for d in datasets:
        row = d.ljust(28)
        for s in scenarios:
            r = by.get((d, s))
            if r is None:
                cell = "-"
            elif r["status"] != "ok":
                cell = r["status"].upper()
            else:
                target, _ = RECALL_BARS.get(s, (0.95, 0.95))
                flag = "ok" if r["avg_recall"] >= target else "LOW"
                cell = f"R={r['avg_recall']:.3f}({flag}) {r['qps']}qps {r['index_mem_mb']:.0f}MB"
            row += cell.ljust(col_w)
        print(row)
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", default=str(REPO_ROOT / "algorithm.py"), help="algorithm.py to evaluate")
    ap.add_argument("--scenarios", default=None, help="scenarios.yaml (default: next to --algorithm)")
    ap.add_argument("--datasets", default=None, help="comma list of dataset stems or .hdf5 paths (default: all in dataset/)")
    ap.add_argument("--scenario", default=None, help="comma list of scenario names (default: all in scenarios.yaml)")
    ap.add_argument("--k", type=int, default=100, help="neighbors per query (competition: 100)")
    ap.add_argument("--queries", type=int, default=None, help="limit number of test queries")
    ap.add_argument("--subset", type=int, default=None, help="use only the first N train vectors (ground truth recomputed)")
    ap.add_argument("--timeout", type=int, default=1800, help="per-cell timeout in seconds")
    ap.add_argument("--out", default=None, help="output JSON path (default: experiments/results/matrix_<ts>.json)")
    ap.add_argument("--list", action="store_true", help="print the resolved matrix and exit")
    ap.add_argument("--_run-cell", dest="run_cell", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.run_cell:
        run_cell_child(args.run_cell)
        return

    _ensure_native_modules_importable()

    algorithm_path = str(Path(args.algorithm).resolve())
    scenarios_path = str(Path(args.scenarios).resolve()) if args.scenarios else str(Path(algorithm_path).parent / "scenarios.yaml")

    if args.datasets:
        dataset_paths = []
        for tok in args.datasets.split(","):
            tok = tok.strip()
            p = Path(tok)
            if not p.suffix:
                p = DATASET_DIR / f"{tok}.hdf5"
            if not p.exists():
                sys.exit(f"dataset not found: {p}")
            dataset_paths.append(p.resolve())
    else:
        dataset_paths = sorted(DATASET_DIR.glob("*.hdf5"))
        if not dataset_paths:
            sys.exit(f"no .hdf5 datasets in {DATASET_DIR}")

    scenarios_cfg = load_scenarios_config(scenarios_path)
    scenario_names = [s.strip() for s in args.scenario.split(",")] if args.scenario else list(scenarios_cfg)
    for s in scenario_names:
        if s not in scenarios_cfg:
            sys.exit(f"scenario {s!r} not in {scenarios_path} (available: {list(scenarios_cfg)})")

    stems = {p.stem for p in dataset_paths}
    print(f"algorithm : {algorithm_path}")
    print(f"scenarios : {scenarios_path}  -> {scenario_names}")
    print(f"datasets  : {sorted(stems)}")
    print(f"k={args.k}  queries={args.queries or 'all'}  subset={args.subset or 'full train'}  timeout={args.timeout}s")
    warn_near_miss_keys({s: scenarios_cfg[s] for s in scenario_names}, stems)

    if args.list:
        for p in dataset_paths:
            for s in scenario_names:
                ip, qp, source = resolve_params(scenarios_cfg[s], p.stem)
                print(f"  {p.stem:32s} {s:14s} <- {source}: index={ip} query={qp}")
        return

    results = []
    n_cells = len(dataset_paths) * len(scenario_names)
    i = 0
    t_start = time.perf_counter()
    for p in dataset_paths:
        for s in scenario_names:
            i += 1
            print(f"[{i}/{n_cells}] {p.stem} x {s} ...", flush=True)
            r = run_cell_parent(
                {
                    "dataset_path": str(p),
                    "dataset_stem": p.stem,
                    "scenario": s,
                    "algorithm_path": algorithm_path,
                    "scenarios_path": scenarios_path,
                    "k": args.k,
                    "n_queries": args.queries,
                    "subset": args.subset,
                },
                timeout=args.timeout,
            )
            results.append(r)
            print(f"    {cell_summary(r)}", flush=True)

    print_table(results, scenario_names)
    print(f"total wall time: {time.perf_counter() - t_start:.0f}s")

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"matrix_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "algorithm": algorithm_path,
                "scenarios_file": scenarios_path,
                "k": args.k,
                "queries": args.queries,
                "subset": args.subset,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "note": "local run via scripts/run_matrix.py; official scores come from orthogonal-competition/evaluator.py",
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"results written to {out_path}")

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print(f"{len(failed)} cell(s) did not complete: " + ", ".join(f"{r['dataset']}x{r['scenario']}({r['status']})" for r in failed))
        sys.exit(2)


if __name__ == "__main__":
    main()
