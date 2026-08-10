import numpy as np
import multiprobe_lsh_cpp

class Algorithm:

    def __init__(self):
        self._index = multiprobe_lsh_cpp.MultiProbeLSH()

    def fit(self, train: np.ndarray, **index_params) -> None:
        ds_size = index_params.get('ds_size', None)
        if ds_size is not None:
            train = train[:int(ds_size)]
        n_tables    = int(index_params.get('n_tables',    10))
        n_bits      = int(index_params.get('n_bits',       8))
        train = np.asarray(train, dtype=np.float32)
        self._index.fit(train, n_tables, n_bits)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        probe_depth = int(query_params.get('probe_depth', 1))
        query = np.asarray(query, dtype=np.float32)
        return self._index.query(query, k, probe_depth)

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
