#!/usr/bin/env python3
"""
One-shot offline parameter tuner for the competition.

Run this ONCE on the target/competition machine. For every dataset x scenario
it searches (backend, index_params, query_params), keeps only configurations
meeting the scenario's recall bar, optimizes the scenario objective, and
writes the result as scenarios.yaml blocks (per-dataset overrides + a robust
'default'). Fit-time auto-tuning was removed from the algorithms on purpose:
build time is a scored metric, so all tuning happens here, offline.

Scenario objectives (recall measured like the evaluator, k=100):
  high_recall : recall >= 0.95  -> minimize mean query time
  fast        : recall >= 0.85  -> minimize mean query time   (--fast-bar to change;
                                    harness README says 0.80, brief said 0.85)
  memory      : recall >= 0.95  -> minimize measured index memory (fresh-process
                                    ru_maxrss around fit, same as the harness)

Efficiency: one child process per (dataset, backend) loads the data once,
fits each BUILD config once, and scans all QUERY configs on that index —
query params (ef / n_probes / ...) don't need a refit. Finalists are then
re-measured in fresh processes (clean peak-RSS) before the final pick.

Safety rails: memory-scenario finalists are ranked by an analytic size
estimate (measured RSS decides among them); if every finalist misses its bar
on re-measurement, the next-best feasible configs are verified until one
passes; configs that ever return fewer than k ids are disqualified (the
harness raises on short results); sweep children save incrementally, so a
timeout keeps the builds that completed.

Usage:
  python3 scripts/tune_parameters.py                    # full tuning, all datasets
  python3 scripts/tune_parameters.py --quick            # tiny grids (smoke test)
  python3 scripts/tune_parameters.py --datasets yahoo-minilm-public --subset 100000
  python3 scripts/tune_parameters.py --style ivf_lsh    # emit bundle-style yaml (no backend field)
  python3 scripts/tune_parameters.py --from-json experiments/results/tuning_x.json
                                                        # re-select from saved measurements

Output:
  scenarios.tuned.yaml (or --write-to PATH)  – ready to drop into the submission
  experiments/results/tuning_<ts>.json       – every measurement (re-selectable)

Each dataset block is emitted twice: keyed '<stem>' and with '-public'
swapped for '-private', because the private test sets share the train data
but have a different filename stem (overrides are matched by stem).
"""

import argparse
import itertools
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

SCENARIOS = ("high_recall", "fast", "memory")


# ---------------------------------------------------------------------------
# Search grids
# ---------------------------------------------------------------------------

def build_grids(k, grid="full"):
    """Return {backend: (build_configs, query_configs)}.
    Build config keys go to index_params, query config keys to query_params."""
    if grid == "focused":
        # The slice of the space that full-scale benchmarking showed matters
        # (docs/TUNING.md "Full-scale benchmark"): sq8 + heuristic ON, two
        # build shapes, ef as the recall lever (hnsw floors ef at k).
        return {
            "hnsw": (
                [
                    {"M": 16, "ef_construction": 100, "mode": "sq8", "heuristic": True},
                    {"M": 8, "ef_construction": 64, "mode": "sq8", "heuristic": True},
                ],
                [{"ef": ef} for ef in (100, 120, 140, 170, 200, 250, 300)],
            ),
            "ivf_lsh": (
                [{"n_tables": 20, "n_bits": 8, "n_clusters": 512}],
                [
                    {"n_probes": p, "n_probe_clusters": pc, "refine_r": 2 * k}
                    for p in (8, 14)
                    for pc in (16, 24)
                ],
            ),
        }
    if grid == "quick":
        return {
            "hnsw": (
                [{"M": 16, "ef_construction": 100, "mode": "sq8", "heuristic": True}],
                [{"ef": 30}, {"ef": 80}],
            ),
            "ivf_lsh": (
                [{"n_tables": 15, "n_bits": 8, "n_clusters": 256}],
                [{"n_probes": 8, "n_probe_clusters": 16, "refine_r": 2 * k}],
            ),
        }
    hnsw_builds = [
        {"M": M, "ef_construction": efc, "mode": "sq8", "heuristic": h}
        for M in (8, 16, 24)
        for efc in (64, 100, 200)
        for h in (False, True)
    ]
    hnsw_queries = [{"ef": ef} for ef in (20, 30, 45, 60, 80, 110, 150, 200, 300)]
    ivf_builds = [
        {"n_tables": t, "n_bits": b, "n_clusters": c}
        for t in (10, 15, 20, 25)
        for b in (7, 8, 10)
        for c in (256, 512)
    ]
    ivf_queries = [
        {"n_probes": p, "n_probe_clusters": pc, "refine_r": r}
        for p in (4, 8, 14, 20)
        for pc in (8, 16, 24)
        for r in (2 * k, 3 * k)
    ]
    return {"hnsw": (hnsw_builds, hnsw_queries), "ivf_lsh": (ivf_builds, ivf_queries)}


def est_index_mb(backend, index_params, npts, dim):
    """Coarse analytic size (MB) of the structures fit() adds on top of the
    raw float train data (which both backends retain). Only used to RANK
    configs for the memory scenario — the decision among verified finalists
    still uses fresh-process ru_maxrss."""
    if backend == "hnsw":
        M = int(index_params["M"])
        codes = dim if str(index_params.get("mode", "sq8")) == "sq8" else 0
        # layer0: 2M int32 edges/node; upper layers hold ~npts/M nodes with M
        # edges; ~48B vector bookkeeping per node
        return npts * (8.0 * M + 4.0 + codes + 48.0) / 2**20
    if backend == "ivf_lsh":
        t = int(index_params["n_tables"])
        b = int(index_params["n_bits"])
        c = int(index_params["n_clusters"])
        ids = 4.0 * npts * t                 # every point sits in one bucket per table
        buckets = 24.0 * t * c * (2.0 ** b)  # nested empty-vector headers in tables_
        codes = 1.0 * npts * dim             # per-dimension sq8 codes
        centroids = 4.0 * c * dim
        return (ids + buckets + codes + centroids) / 2**20
    return float("inf")


# ---------------------------------------------------------------------------
# Shared helpers (evaluator-identical recall; see orthogonal-competition/)
# ---------------------------------------------------------------------------

def recalls(true_distances, predicted_distances, k):
    import numpy as np

    def one(td, pd):
        pd = pd[:k]
        return np.mean(pd <= td[k - 1])

    return np.array([one(true_distances[i], predicted_distances[i]) for i in range(true_distances.shape[0])])


def _ensure_native_modules_importable():
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "build"))
    try:
        import linear_ann_cpp  # noqa: F401
    except ImportError as e:
        sys_libstdcxx = "/usr/lib64/libstdc++.so.6"
        if (
            "__cxa_call_terminate" in str(e)
            and os.path.exists(sys_libstdcxx)
            and os.environ.get("_ANN_TUNE_REEXEC") != "1"
        ):
            env = dict(os.environ)
            env["LD_PRELOAD"] = f"{sys_libstdcxx}:{env.get('LD_PRELOAD', '')}".rstrip(":")
            env["_ANN_TUNE_REEXEC"] = "1"
            os.execve(sys.executable, [sys.executable] + sys.argv, env)
        raise


def make_index(backend, index_params):
    """Instantiate + return (index, fit_fn, query_fn) for a backend."""
    import numpy as np

    if backend == "hnsw":
        import hnsw_cpp

        idx = hnsw_cpp.HNSWIndex()

        def fit(train, p=index_params):
            idx.fit(
                np.asarray(train, dtype=np.float32),
                int(p["M"]),
                int(p["ef_construction"]),
                str(p.get("mode", "sq8")),
                # fallback must MATCH the facade's default (False), or an old
                # --from-json block lacking the key gets verified on a
                # different index than the one that ships
                bool(p.get("heuristic", False)),
            )

        def query(q, k, qp):
            return idx.query(q, k, int(qp["ef"]), int(qp.get("refine_r", -1)))

        return idx, fit, query

    if backend == "ivf_lsh":
        import ivf_lsh_cpp

        idx = ivf_lsh_cpp.LSHIndex()

        def fit(train, p=index_params):
            idx.fit(np.asarray(train, dtype=np.float32), int(p["n_tables"]), int(p["n_bits"]), int(p["n_clusters"]))

        def query(q, k, qp):
            return idx.query(q, k, int(qp["n_probes"]), int(qp["n_probe_clusters"]), int(qp.get("refine_r", -1)))

        return idx, fit, query

    raise ValueError(f"unknown backend {backend!r}")


def load_raw(dataset_path, subset, n_queries):
    """Return (train, queries, file_dist or None). file_dist is dropped when
    subsetting (the file's ground truth refers to the full train set)."""
    import h5py

    with h5py.File(dataset_path, "r") as f:
        train = f["train"][:]
        queries = f["test"][:]
        file_dist = f["distances"][:] if "distances" in f else None
    if subset:
        train = train[:subset]
        file_dist = None
    if n_queries:
        queries = queries[:n_queries]
    return train, queries, file_dist


def true_distances(train, queries, file_dist, k):
    import numpy as np

    if file_dist is not None and file_dist.shape[1] >= k:
        return file_dist[: queries.shape[0], :k]
    from linear_ann.wrapper import LinearANN

    gt = LinearANN()
    gt.fit(train)
    idx = np.array([gt.query(q, k) for q in queries])
    return np.array(
        [np.sort(np.linalg.norm(train[idx[i]] - queries[i].astype(np.float32), axis=1)) for i in range(len(queries))]
    )


def load_dataset(dataset_path, subset, n_queries, k):
    """Return (train, queries, true_dist[:, :k])."""
    train, queries, file_dist = load_raw(dataset_path, subset, n_queries)
    return train, queries, true_distances(train, queries, file_dist, k)


def measure_queries(train, queries, true_dist, k, query_fn, qp):
    """Run all queries with query params qp.
    Returns (recall, mean_ms, median_ms, n_short). n_short counts queries that
    returned fewer than k ids — the real harness raises ValueError on those,
    so callers must disqualify any config with n_short > 0."""
    import numpy as np

    n_q = queries.shape[0]
    qtimes = np.empty(n_q)
    neigh = np.empty((n_q, k), dtype=np.int64)
    n_short = 0
    for i in range(n_q):
        t0 = time.perf_counter()
        res = query_fn(queries[i], k, qp)
        qtimes[i] = time.perf_counter() - t0
        res = np.asarray(res)
        if res.shape[0] < k:  # pad so the numbers still compute; config is disqualified
            n_short += 1
            res = np.pad(res, (0, k - res.shape[0]), constant_values=res[-1] if res.shape[0] else 0)
        neigh[i] = res[:k]
    pred = np.array([np.linalg.norm(train[neigh[i]] - queries[i].astype(np.float32), axis=1) for i in range(n_q)])
    rec = float(recalls(true_dist, pred, k).mean())
    return rec, float(qtimes.mean() * 1e3), float(np.median(qtimes) * 1e3), n_short


# ---------------------------------------------------------------------------
# Child mode A: sweep one (dataset, backend) — one fit per build config,
# scan all query configs on the fitted index.
# ---------------------------------------------------------------------------

def _dump_json(obj, path):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def run_sweep_child(spec_path):
    _ensure_native_modules_importable()
    with open(spec_path) as f:
        spec = json.load(f)
    out = {"measurements": [], "errors": []}
    try:
        k = spec["k"]
        train, queries, true_dist = load_dataset(spec["dataset_path"], spec["subset"], spec["n_queries"], k)
        npts, dim = train.shape
        for bi, bp in enumerate(spec["build_configs"]):
            try:
                _, fit_fn, query_fn = make_index(spec["backend"], bp)
                t0 = time.perf_counter()
                fit_fn(train)
                build_s = time.perf_counter() - t0
                for qp in spec["query_configs"]:
                    rec, mean_ms, med_ms, n_short = measure_queries(train, queries, true_dist, k, query_fn, qp)
                    out["measurements"].append(
                        {
                            "backend": spec["backend"],
                            "index_params": bp,
                            "query_params": qp,
                            "recall": round(rec, 4),
                            "mean_query_ms": round(mean_ms, 4),
                            "median_query_ms": round(med_ms, 4),
                            "build_time_s": round(build_s, 2),
                            "est_index_mb": round(est_index_mb(spec["backend"], bp, npts, dim), 1),
                            "short_results": n_short,
                            "n_train": int(npts),
                            "dim": int(dim),
                        }
                    )
                print(f"    build {bi + 1}/{len(spec['build_configs'])} {bp} done ({build_s:.1f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                out["errors"].append(f"build {bp}: {type(e).__name__}: {e}")
            _dump_json(out, spec["out"])  # incremental: a timeout keeps completed builds
    except Exception as e:  # noqa: BLE001
        import traceback

        out["errors"].append(traceback.format_exc())
    _dump_json(out, spec["out"])


# ---------------------------------------------------------------------------
# Child mode B: measure ONE config in a fresh process (clean ru_maxrss).
# ---------------------------------------------------------------------------

def run_measure_child(spec_path):
    _ensure_native_modules_importable()
    import resource

    with open(spec_path) as f:
        spec = json.load(f)
    out = {"status": "error"}
    try:
        k = spec["k"]
        # fit BEFORE computing any missing ground truth: ru_maxrss is a
        # high-water mark, and a GT brute-force beforehand (--subset path)
        # would swallow the fit's memory delta.
        train, queries, file_dist = load_raw(spec["dataset_path"], spec["subset"], spec["n_queries"])
        _, fit_fn, query_fn = make_index(spec["backend"], spec["index_params"])
        mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        t0 = time.perf_counter()
        fit_fn(train)
        build_s = time.perf_counter() - t0
        index_mem_mb = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - mem0) / 1024
        true_dist = true_distances(train, queries, file_dist, k)
        rec, mean_ms, med_ms, n_short = measure_queries(train, queries, true_dist, k, query_fn, spec["query_params"])
        out = {
            "status": "ok",
            "recall": round(rec, 4),
            "mean_query_ms": round(mean_ms, 4),
            "median_query_ms": round(med_ms, 4),
            "build_time_s": round(build_s, 2),
            "index_mem_mb": round(index_mem_mb, 1),
            "short_results": n_short,
        }
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    _dump_json(out, spec["out"])


def spawn_child(mode_flag, spec, timeout):
    with tempfile.TemporaryDirectory(prefix="ann-tune-") as td:
        spec = dict(spec, out=os.path.join(td, "out.json"))
        sp = os.path.join(td, "spec.json")
        with open(sp, "w") as f:
            json.dump(spec, f)
        try:
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), mode_flag, sp],
                timeout=timeout,
                cwd=REPO_ROOT,
            )
        except subprocess.TimeoutExpired:
            if os.path.exists(spec["out"]):  # sweep children save incrementally
                with open(spec["out"]) as f:
                    out = json.load(f)
                out.setdefault("errors", []).append(f"exceeded {timeout}s — kept completed builds")
                out["status"] = "timeout"
                return out
            return {"status": "timeout", "errors": [f"exceeded {timeout}s"], "measurements": []}
        if os.path.exists(spec["out"]):
            with open(spec["out"]) as f:
                return json.load(f)
        return {"status": "crashed", "errors": ["child died"], "measurements": []}


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def scenario_bar(scenario, fast_bar):
    return {"high_recall": 0.95, "fast": fast_bar, "memory": 0.95}[scenario]


def config_key(m):
    return (m["backend"], json.dumps(m["index_params"], sort_keys=True), json.dumps(m["query_params"], sort_keys=True))


def objective_key(scenario):
    """Sort key for the scenario objective. memory ranks by the analytic size
    estimate (fresh-process RSS decides among the verified finalists); older
    tuning jsons without est_index_mb fall back to query time."""
    if scenario == "memory":
        return lambda m: (
            m["est_index_mb"] if m.get("est_index_mb") is not None else float("inf"),
            m["mean_query_ms"],
        )
    return lambda m: m["mean_query_ms"]


def usable(m):
    """The harness raises when a query returns fewer than k ids — a config
    that ever produced short results is disqualified outright."""
    return not m.get("short_results")


def select_for_scenario(measurements, scenario, fast_bar, margin):
    """Feasible = recall >= bar + margin (and no short results); objective:
    query time, or the analytic memory estimate for the memory scenario."""
    bar = scenario_bar(scenario, fast_bar) + margin
    feasible = [m for m in measurements if usable(m) and m["recall"] >= bar]
    if not feasible:
        # nothing meets the bar: return best-recall configs so the report shows how close we got
        return sorted(measurements, key=lambda m: (not usable(m), -m["recall"]))[:3], False
    return sorted(feasible, key=objective_key(scenario))[:3], True


def select_default(all_meas_by_dataset, scenario, fast_bar, margin):
    """Cross-dataset default: config feasible on EVERY dataset with the best
    worst-case objective. Falls back to the config feasible on most datasets.
    Datasets whose sweep produced nothing are excluded — one failed sweep must
    not erase the 'default' block (the evaluator refuses a scenarios.yaml
    without it)."""
    all_meas_by_dataset = {ds: ms for ds, ms in all_meas_by_dataset.items() if ms}
    if not all_meas_by_dataset:
        return None
    bar = scenario_bar(scenario, fast_bar) + margin
    obj = objective_key(scenario)
    per_config = {}
    for ds, ms in all_meas_by_dataset.items():
        for m in ms:
            if usable(m):
                per_config.setdefault(config_key(m), {})[ds] = m
    n_ds = len(all_meas_by_dataset)
    best, best_cover, best_worst = None, None, None
    for per_ds in per_config.values():
        if len(per_ds) < n_ds:
            continue  # not measured everywhere (shouldn't happen with fixed grids)
        cover = sum(1 for m in per_ds.values() if m["recall"] >= bar)
        worst_obj = max(obj(m) for m in per_ds.values())
        # maximize coverage, then minimize the worst-case objective
        if best is None or cover > best_cover or (cover == best_cover and worst_obj < best_worst):
            best, best_cover, best_worst = next(iter(per_ds.values())), cover, worst_obj
    return best


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------

def emit_yaml(selection, style, path):
    import yaml

    scenarios_block = {}
    for scen, per_ds in selection.items():
        block = {}
        for ds_key, chosen in per_ds.items():
            if chosen is None:
                continue
            ip = dict(chosen["index_params"])
            qp = dict(chosen["query_params"])
            if style == "facade":
                ip = {"backend": chosen["backend"], **ip}
            entry = {"index_params": ip, "query_params": qp}
            block[ds_key] = entry
            if ds_key.endswith("-public"):
                # private test sets share train data but have a different stem;
                # without this twin block they would fall back to 'default'
                block[ds_key.replace("-public", "-private")] = entry
        scenarios_block[scen] = block
    with open(path, "w") as f:
        f.write("# Generated by scripts/tune_parameters.py — do not hand-edit measured blocks.\n")
        f.write(f"# style={style}  generated={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        yaml.safe_dump({"scenarios": scenarios_block}, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default=None, help="comma list of dataset stems (default: all in dataset/)")
    ap.add_argument("--backends", default="hnsw,ivf_lsh", help="backends to search")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--queries", type=int, default=200, help="queries used during the sweep")
    ap.add_argument("--verify-queries", type=int, default=None, help="queries for finalist verification (default: all)")
    ap.add_argument("--subset", type=int, default=None, help="train subset for the sweep (default: full train)")
    ap.add_argument("--fast-bar", type=float, default=0.85)
    ap.add_argument("--margin", type=float, default=0.005, help="safety margin added to every recall bar")
    ap.add_argument("--quick", action="store_true", help="tiny grids — smoke test only (alias for --grid quick)")
    ap.add_argument("--grid", choices=("full", "quick", "focused"), default="full",
                    help="search-grid preset; 'focused' = the benchmark-informed hnsw slice")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip fresh-process verification; pick winners straight from sweep numbers "
                         "(use when the sweep already ran on the full dataset)")
    ap.add_argument("--timeout", type=int, default=7200, help="per (dataset,backend) sweep timeout")
    ap.add_argument("--style", choices=("facade", "ivf_lsh"), default="facade",
                    help="facade: blocks include backend (root algorithm.py); ivf_lsh: params only")
    ap.add_argument("--write-to", default=str(REPO_ROOT / "scenarios.tuned.yaml"))
    ap.add_argument("--from-json", default=None, help="skip measuring; re-select from a saved tuning json")
    ap.add_argument("--_sweep", dest="sweep_spec", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--_measure", dest="measure_spec", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.sweep_spec:
        run_sweep_child(args.sweep_spec)
        return
    if args.measure_spec:
        run_measure_child(args.measure_spec)
        return

    _ensure_native_modules_importable()

    if args.datasets:
        dataset_paths = [DATASET_DIR / f"{s.strip()}.hdf5" for s in args.datasets.split(",")]
        for p in dataset_paths:
            if not p.exists():
                sys.exit(f"dataset not found: {p}")
    else:
        dataset_paths = sorted(DATASET_DIR.glob("*.hdf5"))
    backends = [b.strip() for b in args.backends.split(",")]
    if args.style == "ivf_lsh" and backends != ["ivf_lsh"]:
        # bundle-style yaml has no backend field; the ivf_lsh bundle reads only
        # its own params, so hnsw configs would be emitted and silently ignored
        print("--style ivf_lsh: restricting backends to ivf_lsh (bundle can't run anything else)")
        backends = ["ivf_lsh"]
    grid = "quick" if args.quick else args.grid
    grids = build_grids(args.k, grid=grid)

    # ---------- Phase A: sweep ----------
    if args.from_json:
        with open(args.from_json) as f:
            saved = json.load(f)
        meas_by_dataset = saved["measurements_by_dataset"]
        print(f"loaded {sum(len(v) for v in meas_by_dataset.values())} measurements from {args.from_json}")
    else:
        meas_by_dataset = {}
        for p in dataset_paths:
            meas_by_dataset[p.stem] = []
            for backend in backends:
                if backend not in grids:
                    sys.exit(f"no grid for backend {backend!r}")
                builds, qcfgs = grids[backend]
                print(f"[sweep] {p.stem} x {backend}: {len(builds)} builds x {len(qcfgs)} query configs", flush=True)
                res = spawn_child(
                    "--_sweep",
                    {
                        "dataset_path": str(p),
                        "backend": backend,
                        "build_configs": builds,
                        "query_configs": qcfgs,
                        "k": args.k,
                        "n_queries": args.queries,
                        "subset": args.subset,
                    },
                    timeout=args.timeout,
                )
                for err in res.get("errors", []):
                    print(f"  ERROR: {err}")
                meas_by_dataset[p.stem].extend(res.get("measurements", []))
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        sweep_json = RESULTS_DIR / f"tuning_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(sweep_json, "w") as f:
            json.dump(
                {
                    "k": args.k,
                    "queries": args.queries,
                    "subset": args.subset,
                    "grid": grid,
                    "measurements_by_dataset": meas_by_dataset,
                },
                f,
                indent=1,
            )
        print(f"sweep measurements saved to {sweep_json}")

    if args.style == "ivf_lsh":
        # also covers --from-json files that contain hnsw measurements
        meas_by_dataset = {ds: [m for m in ms if m["backend"] == "ivf_lsh"] for ds, ms in meas_by_dataset.items()}

    for ds, ms in meas_by_dataset.items():
        if not ms:
            print(
                f"WARNING: {ds}: sweep produced no measurements — it gets no per-dataset "
                f"block (falls back to 'default') and is excluded from the default computation"
            )

    def verify_one(ds, cand):
        return spawn_child(
            "--_measure",
            {
                "dataset_path": str(DATASET_DIR / f"{ds}.hdf5"),
                "backend": cand["backend"],
                "index_params": cand["index_params"],
                "query_params": cand["query_params"],
                "k": args.k,
                "n_queries": args.verify_queries,
                "subset": args.subset,
            },
            timeout=args.timeout,
        )

    # ---------- Phase B: select + verify finalists in fresh processes ----------
    selection = {}
    report_lines = []
    for scen in SCENARIOS:
        selection[scen] = {}
        for ds, ms in meas_by_dataset.items():
            if not ms:
                continue
            finalists, feasible = select_for_scenario(ms, scen, args.fast_bar, args.margin)
            if args.no_verify:
                verified = list(finalists)  # trust the sweep numbers (full-dataset sweeps)
            else:
                verified = []
                for cand in finalists:
                    res = verify_one(ds, cand)
                    if res.get("status") == "ok":
                        verified.append({**cand, **res})
            bar = scenario_bar(scen, args.fast_bar) + args.margin
            ok = [m for m in verified if usable(m) and m["recall"] >= bar]
            if feasible and not ok:
                # every finalist missed its bar on re-measurement (sweep recall is
                # a ~200-query sample) — widen to the next-best feasible configs
                # instead of shipping a below-bar block
                tried = {config_key(c) for c in finalists}
                rest = sorted(
                    (m for m in ms if usable(m) and m["recall"] >= bar and config_key(m) not in tried),
                    key=objective_key(scen),
                )
                for cand in rest[:5]:
                    print(
                        f"  [{scen} x {ds}] finalists failed verification — widening to "
                        f"{cand['index_params']} {cand['query_params']}",
                        flush=True,
                    )
                    res = verify_one(ds, cand)
                    if res.get("status") != "ok":
                        continue
                    verified.append({**cand, **res})
                    if usable(verified[-1]) and verified[-1]["recall"] >= bar:
                        ok = [verified[-1]]
                        break
            pool = verified or finalists
            if scen == "memory":
                chosen = min(ok, key=lambda m: m.get("index_mem_mb", 1e18)) if ok else max(pool, key=lambda m: m["recall"])
            else:
                chosen = min(ok, key=lambda m: m["mean_query_ms"]) if ok else max(pool, key=lambda m: m["recall"])
            selection[scen][ds] = chosen
            status = "OK " if (feasible and ok) else "BELOW-BAR"
            report_lines.append(
                f"{scen:12s} {ds:30s} {status} {chosen['backend']:8s} "
                f"recall={chosen['recall']:.4f} mean={chosen['mean_query_ms']:.3f}ms "
                f"mem={chosen.get('index_mem_mb', float('nan'))}MB build={chosen['build_time_s']}s "
                f"index={chosen['index_params']} query={chosen['query_params']}"
            )
        # cross-dataset default (what private sets hit if the -private twin blocks are removed)
        default = select_default(meas_by_dataset, scen, args.fast_bar, args.margin)
        if default is not None:
            selection[scen]["default"] = default
            report_lines.append(
                f"{scen:12s} {'default':30s}     {default['backend']:8s} "
                f"recall(min across ds)>=bar? see json  index={default['index_params']} query={default['query_params']}"
            )

    print("\n==== TUNING REPORT " + "=" * 60)
    for line in report_lines:
        print(line)

    # 'default' must come first for readability; per-dataset blocks after
    ordered = {scen: {"default": sel.pop("default"), **sel} if "default" in sel else sel for scen, sel in selection.items()}
    emit_yaml(ordered, args.style, args.write_to)
    print(f"\ntuned scenarios written to {args.write_to}")
    print("Review it, then copy/merge into the submission's scenarios.yaml.")


if __name__ == "__main__":
    main()
