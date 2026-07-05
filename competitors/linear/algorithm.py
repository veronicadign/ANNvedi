import numpy as np
import sys
import os

# Ensure the local directory is on python path for importing the compiled binary module
sys.path.insert(0, os.path.dirname(__file__))

import linear_ann_cpp

class Algorithm:
    def __init__(self):
        self._index = linear_ann_cpp.LinearIndex()

    def fit(self, train: np.ndarray, **index_params) -> None:
        ds_size = index_params.get('ds_size', None)
        if ds_size is not None:
            train = train[:int(ds_size)]
        train = np.asarray(train, dtype=np.float32)
        self._index.fit(train)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        return self._index.query(query, k)

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
