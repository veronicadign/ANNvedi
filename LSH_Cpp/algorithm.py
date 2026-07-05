import numpy as np
import lsh_cpp_module

class Algorithm:

    def __init__(self):
        self._index = lsh_cpp_module.LSHIndex()

    def fit(self, train: np.ndarray, **index_params) -> None:
        ds_size = index_params.get('ds_size', None)
        if ds_size is not None:
            train = train[:int(ds_size)]
        n_tables = index_params.get('n_tables', 10)
        n_bits = index_params.get('n_bits', 8)
        n_clusters = index_params.get('n_clusters', 256)
        
        # Cast to float32 to ensure exact match with C++ float buffer
        train = np.asarray(train, dtype=np.float32)
        self._index.fit(train, n_tables, n_bits, n_clusters)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        n_probes = int(query_params.get('n_probes', 0))
        n_probe_clusters = int(query_params.get('n_probe_clusters', 8))
        refine_r = int(query_params.get('refine_r', -1))
        # Cast to float32 to ensure exact match with C++ float buffer
        query = np.asarray(query, dtype=np.float32)
        neighbors = self._index.query(query, k, n_probes, n_probe_clusters, refine_r)
        return neighbors

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
