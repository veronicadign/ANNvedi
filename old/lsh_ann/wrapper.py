import numpy as np
import lsh_cpp_module


class LSHANN:
    def __init__(self, n_tables=10, n_bits=8, n_clusters=256):
        self._n_tables = int(n_tables)
        self._n_bits = int(n_bits)
        self._n_clusters = int(n_clusters)
        self._index = lsh_cpp_module.LSHIndex()
        self._data = None

    def fit(self, data: np.ndarray) -> None:
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2:
            raise ValueError("data must be 2D")
        self._data = data
        self._index.fit(data, self._n_tables, self._n_bits, self._n_clusters)

    def query(self, q: np.ndarray, k: int, n_probes: int = 0, n_probe_clusters: int = 8, refine_r: int = -1) -> np.ndarray:
        if self._data is None:
            raise RuntimeError("index not trained")
        q = np.asarray(q, dtype=np.float32)
        if q.ndim != 1 or q.shape[0] != self._data.shape[1]:
            raise ValueError("query shape mismatch")
        
        return self._index.query(q, k, int(n_probes), int(n_probe_clusters), int(refine_r))

    def total_distances_count(self) -> int:
        return self._index.total_distances_count()

    def reset_distances_count(self) -> None:
        self._index.reset_distances_count()

    def get_profile_results(self) -> dict:
        return self._index.get_profile_results()

    def reset_profile_results(self) -> None:
        self._index.reset_profile_results()
