import ctypes
import os

import numpy as np

import hnsw_cpp  # pip-installed at image build

_GPU_LIB = None


def _load_gpu():
    """Load the GPU library lazily; returns None when unusable."""
    global _GPU_LIB
    if _GPU_LIB is not None:
        return _GPU_LIB
    path = os.path.join(os.path.dirname(__file__), "libgpufilter.so")
    try:
        lib = ctypes.CDLL(path)
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
    _GPU_LIB = lib
    return lib


class Algorithm:
    """Hybrid contestant: the index itself selects the backend at fit() time
    (allowed by the rules: parameters may be set 'al volo' by the index).

    backend: 'gpu'  -> GPU quantized filter + CPU exact rerank (build =
                       quantize + upload, seconds; recall 0.997-1.0 measured
                       on all 7 datasets at r=150); if no CUDA device is
                       visible it FALLS BACK to hnsw with this block's params.
             'hnsw' -> the tuned CPU HNSW (sq8/sq8pd + heuristic + reorder).

    No double-building: exactly one backend is constructed per fit — build
    time is a scored metric.
    """

    def __init__(self):
        self._impl = None
        self._gpu_handle = None
        self._hnsw = None
        self._train = None

    def fit(self, train: np.ndarray, **p) -> None:
        self._train = np.ascontiguousarray(train, dtype=np.float32)
        want = str(p.get("backend", "gpu")).lower()

        if want == "gpu":
            lib = _load_gpu()
            if lib is not None:
                n, dim = self._train.shape
                ptr = self._train.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                h = lib.gf_create(ptr, n, dim)
                if h:
                    self._gpu_handle, self._impl = h, "gpu"
                    return
                print(f"[hybrid] gpu unavailable ({lib.gf_last_error().decode()}) "
                      f"-> hnsw fallback", flush=True)
            else:
                print("[hybrid] libgpufilter.so not loadable -> hnsw fallback", flush=True)

        self._hnsw = hnsw_cpp.HNSWIndex()
        self._hnsw.fit(
            self._train,
            int(p.get("M", 16)),
            int(p.get("ef_construction", 100)),
            str(p.get("mode", "sq8")),
            bool(p.get("heuristic", True)),
            bool(p.get("reorder", True)),
        )
        self._impl = "hnsw"

    def query(self, query: np.ndarray, k: int, **qp) -> np.ndarray:
        q = np.ascontiguousarray(query, dtype=np.float32)
        if self._impl == "gpu":
            lib = _GPU_LIB
            out = np.empty(k, dtype=np.int64)
            got = lib.gf_query(self._gpu_handle,
                               q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               k, int(qp.get("r", 150)),
                               out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)))
            if got < 0:
                raise RuntimeError(lib.gf_last_error().decode())
            return out[:got]
        return self._hnsw.query(q, k, int(qp.get("ef", 120)))

    def get_n_distances(self) -> int:
        if self._impl == "gpu":
            return int(_GPU_LIB.gf_ndist(self._gpu_handle))
        return self._hnsw.total_distances_count() if self._hnsw else 0

    def __del__(self):
        if self._gpu_handle and _GPU_LIB:
            _GPU_LIB.gf_free(self._gpu_handle)
            self._gpu_handle = None
