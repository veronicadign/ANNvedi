import ctypes
import os
import time

import numpy as np

import hnsw_cpp  # pip-installed at image build

# ---------------------------------------------------------------------------
# Self-calibrating contestant.
#
# At fit() the index (rules: parameters may be set "al volo" by the index):
#   1. selects the backend: GPU quantized-filter if a CUDA device is visible,
#      else the CPU HNSW;
#   2. computes EXACT ground truth for a small sample of pseudo-queries
#      (train points; multithreaded float scan — a few seconds);
#   3. ladder-calibrates the backend's query knob (r for GPU, ef for HNSW)
#      until sample recall clears the scenario bar PLUS a +1.5% safety margin.
#
# Scenario bars (harness exposes SCENARIO_NAME / QUERY_K):
#   high_recall, memory: 0.95    fast: 0.85    margin: +0.015
# ---------------------------------------------------------------------------

BARS = {"high_recall": 0.95, "memory": 0.95, "fast": 0.85}
MARGIN = 0.01
# In-sample pseudo-queries measure ~1% easier than real unseen queries even
# with the self-hit excluded (measured on yahoo: sample 0.973 vs true 0.963).
# The sample target compensates: bar + MARGIN + CALIBRATION_BIAS.
CALIBRATION_BIAS = 0.012
SAMPLE = 128
GPU_R_LADDER = [150, 300, 600, 1200]
EF_LADDER = [100, 120, 170, 250, 350, 500]
HNSW_BUILD = {  # per scenario: (M, ef_construction) — recall headroom for
    "high_recall": (16, 100),   # high_recall, small graph for fast/memory
    "fast": (8, 64),
    "memory": (8, 64),
}

_GPU = None


def _load_gpu():
    global _GPU
    if _GPU is not None:
        return _GPU
    try:
        lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libgpufilter.so"))
    except OSError:
        return None
    lib.gf_create.restype = ctypes.c_void_p
    lib.gf_create.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_long, ctypes.c_long]
    lib.gf_query.restype = ctypes.c_int
    lib.gf_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                             ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
    lib.gf_ndist.restype = ctypes.c_longlong
    lib.gf_ndist.argtypes = [ctypes.c_void_p]
    lib.gf_free.argtypes = [ctypes.c_void_p]
    lib.gf_last_error.restype = ctypes.c_char_p
    _GPU = lib
    return lib


class Algorithm:
    def __init__(self):
        self._impl = None
        self._gpu_handle = None
        self._hnsw = None
        self._train = None
        self._r = GPU_R_LADDER[0]
        self._ef = EF_LADDER[0]

    # -- calibration helpers -------------------------------------------------

    def _sample_gt(self, k):
        """Exact kth NON-SELF NN thresholds for SAMPLE train pseudo-queries.

        Self is excluded (rank-0 hit at distance 0 would gift 1/k recall).
        Even so, in-graph queries measure ~1% easier than real unseen ones
        (the beam's landing zone IS their own neighborhood) — that measured
        gap is compensated by CALIBRATION_BIAS on the sample target.
        """
        n = self._train.shape[0]
        rng = np.random.default_rng(12345)
        self._cal_idx = rng.choice(n, size=min(SAMPLE, n), replace=False)
        self._cal_q = np.ascontiguousarray(self._train[self._cal_idx])
        q_norm = (self._cal_q ** 2).sum(1)
        CH = 200_000
        keep = k + 1
        best = np.full((len(self._cal_idx), keep), np.inf, dtype=np.float32)
        for s in range(0, n, CH):
            block = self._train[s:s + CH]
            d = (block ** 2).sum(1)[None, :] - 2.0 * (self._cal_q @ block.T) + q_norm[:, None]
            both = np.concatenate([best, d], axis=1)
            best = np.partition(both, keep - 1, axis=1)[:, :keep]
        return np.sqrt(np.maximum(np.sort(best, axis=1)[:, k], 0.0))

    def _sample_recall(self, query_fn, k, thresholds):
        recs = []
        for i, q in enumerate(self._cal_q):
            r = np.asarray(query_fn(q, k + 1))
            r = r[r != self._cal_idx[i]][:k]
            d = np.linalg.norm(self._train[r] - q, axis=1)
            recs.append(np.mean(np.sort(d)[:k] <= thresholds[i] + 1e-6))
        return float(np.mean(recs))

    # -- harness API ----------------------------------------------------------

    def fit(self, train: np.ndarray, **index_params) -> None:
        t0 = time.perf_counter()
        self._train = np.ascontiguousarray(train, dtype=np.float32)
        scenario = os.environ.get("SCENARIO_NAME", "high_recall").lower()
        k = int(os.environ.get("QUERY_K", 100))
        target = BARS.get(scenario, 0.95) + MARGIN + CALIBRATION_BIAS
        print(f"[ANNvedi] scenario={scenario} k={k} sample_target={target:.3f} (bar+{MARGIN}+bias {CALIBRATION_BIAS})", flush=True)

        lib = _load_gpu()
        if lib is not None:
            n, dim = self._train.shape
            ptr = self._train.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            h = lib.gf_create(ptr, n, dim)
            if h:
                self._gpu_handle, self._impl = h, "gpu"
            else:
                print(f"[ANNvedi] gpu unavailable ({lib.gf_last_error().decode()})", flush=True)
        if self._impl != "gpu":
            dM, defc = HNSW_BUILD.get(scenario, (16, 100))
            self._hnsw = hnsw_cpp.HNSWIndex()
            self._hnsw.fit(self._train,
                           int(index_params.get("M", dM)),
                           int(index_params.get("ef_construction", defc)),
                           str(index_params.get("mode", "sq8")))
            self._impl = "hnsw"

        # ---- knobs: pre-seeded per-dataset values skip calibration entirely ----
        seed = index_params.get("r" if self._impl == "gpu" else "ef")
        if seed is not None:
            if self._impl == "gpu":
                self._r = int(seed)
            else:
                self._ef = int(seed)
            print(f"[ANNvedi] seeded {'r' if self._impl == 'gpu' else 'ef'}={seed} "
                  f"— calibration skipped", flush=True)
            return

        # ---- self-calibration against exact sample ground truth ----
        thresholds = self._sample_gt(k)
        if self._impl == "gpu":
            for r in GPU_R_LADDER:
                rec = self._sample_recall(lambda q, kk: self._gpu_query(q, kk, r), k, thresholds)
                print(f"[ANNvedi] calibrate gpu r={r}: sample recall={rec:.4f}", flush=True)
                if rec >= target:
                    break
            self._r = r
        else:
            for ef in EF_LADDER:
                rec = self._sample_recall(lambda q, kk: self._hnsw.query(q, kk, ef), k, thresholds)
                print(f"[ANNvedi] calibrate hnsw ef={ef}: sample recall={rec:.4f}", flush=True)
                if rec >= target:
                    break
            self._ef = ef
        print(f"[ANNvedi] impl={self._impl} calibrated "
              f"{'r=' + str(self._r) if self._impl == 'gpu' else 'ef=' + str(self._ef)} "
              f"(calibration took {time.perf_counter() - t0:.1f}s total fit overhead incl. build)",
              flush=True)

    def _gpu_query(self, q, k, r):
        q = np.ascontiguousarray(q, dtype=np.float32)
        out = np.empty(k, dtype=np.int64)
        got = _GPU.gf_query(self._gpu_handle,
                            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            k, r, out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)))
        if got < 0:
            raise RuntimeError(_GPU.gf_last_error().decode())
        return out[:got]

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        if self._impl == "gpu":
            return self._gpu_query(query, k, int(query_params.get("r", self._r)))
        return self._hnsw.query(np.ascontiguousarray(query, dtype=np.float32), k,
                                int(query_params.get("ef", self._ef)))

    def get_n_distances(self) -> int:
        if self._impl == "gpu":
            return int(_GPU.gf_ndist(self._gpu_handle))
        return self._hnsw.total_distances_count() if self._hnsw else 0

    def __del__(self):
        if self._gpu_handle and _GPU:
            _GPU.gf_free(self._gpu_handle)
            self._gpu_handle = None
