import numpy as np
import sys
import os

# Ensure the local directory is on python path for importing the compiled binary module
sys.path.insert(0, os.path.dirname(__file__))

import hnsw_cpp

class Algorithm:
    def __init__(self):
        self._index = hnsw_cpp.HNSWIndex()

    def fit(self, train: np.ndarray, **index_params) -> None:
        ds_size = index_params.get('ds_size', None)
        if ds_size is not None:
            train = train[:int(ds_size)]
        
        M = int(index_params.get('M', 16))
        ef_construction = int(index_params.get('ef_construction', 100))
        mode = str(index_params.get('mode', 'float'))
        
        train = np.asarray(train, dtype=np.float32)
        self._index.fit(train, M, ef_construction, mode)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        ef = int(query_params.get('ef', 50))
        refine_r = int(query_params.get('refine_r', -1))
        
        query = np.asarray(query, dtype=np.float32)
        return self._index.query(query, k, ef, refine_r)

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
