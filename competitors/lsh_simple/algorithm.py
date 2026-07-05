import numpy as np
import sys
import os

# Ensure the local directory is on python path for importing the compiled binary module
sys.path.insert(0, os.path.dirname(__file__))

import lsh_cpp_simple

class Algorithm:
    def __init__(self):
        self._index = lsh_cpp_simple.LSHIndex()

    def fit(self, train: np.ndarray, **index_params) -> None:
        ds_size = index_params.get('ds_size', None)
        if ds_size is not None:
            train = train[:int(ds_size)]
        
        n_tables = int(index_params.get('n_tables', 10))
        n_bits = int(index_params.get('n_bits', 8))
        
        train = np.asarray(train, dtype=np.float32)
        self._index.fit(train, n_tables, n_bits)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        n_probes = int(query_params.get('n_probes', 0))
        n_probe_clusters = int(query_params.get('n_probe_clusters', 8))
        refine_r = int(query_params.get('refine_r', -1))
        
        query = np.asarray(query, dtype=np.float32)
        return self._index.query(query, k, n_probes, n_probe_clusters, refine_r)

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
