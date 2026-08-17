import ctypes
import os

import numpy as np

_LIB = ctypes.CDLL(os.path.join(os.path.dirname(__file__), "libgpufilter.so"))
_LIB.gf_create.restype = ctypes.c_void_p
_LIB.gf_create.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_long, ctypes.c_long]
_LIB.gf_query.restype = ctypes.c_int
_LIB.gf_query.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
                          ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
_LIB.gf_ndist.restype = ctypes.c_longlong
_LIB.gf_ndist.argtypes = [ctypes.c_void_p]
_LIB.gf_free.argtypes = [ctypes.c_void_p]
_LIB.gf_last_error.restype = ctypes.c_char_p


class Algorithm:
    """GPU quantized-filter / CPU exact-rerank contestant.

    The GPU holds only per-dimension SQ8 codes (1/4 the float bytes) and
    brute-force scans them per query; the top-R candidate ids come back and
    the exact float verification happens on the CPU from host RAM. Validated
    on full yahoo: the quantized top-150 always contains the true top-100
    (recall 1.0000), so R in scenarios.yaml carries large safety margins.

    Requires a visible CUDA device (container started with --gpus).
    """

    def __init__(self):
        self._handle = None
        self._train = None  # keeps the buffer gf_create borrows alive

    def fit(self, train: np.ndarray, **index_params) -> None:
        self._train = np.ascontiguousarray(train, dtype=np.float32)
        n, dim = self._train.shape
        ptr = self._train.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._handle = _LIB.gf_create(ptr, n, dim)
        if not self._handle:
            raise RuntimeError(_LIB.gf_last_error().decode())

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        r = int(query_params.get("r", 500))
        q = np.ascontiguousarray(query, dtype=np.float32)
        out = np.empty(k, dtype=np.int64)
        got = _LIB.gf_query(self._handle,
                            q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            k, r, out.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)))
        if got < 0:
            raise RuntimeError(_LIB.gf_last_error().decode())
        return out[:got]

    def get_n_distances(self) -> int:
        return int(_LIB.gf_ndist(self._handle)) if self._handle else 0

    def __del__(self):
        if self._handle:
            _LIB.gf_free(self._handle)
            self._handle = None
