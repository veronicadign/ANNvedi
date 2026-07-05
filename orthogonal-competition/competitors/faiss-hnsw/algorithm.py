"""
Example: FAISS competitor
=========================

This is an example submission that solves queries using the
popular faiss HNSW implementation.
"""

import numpy as np
import faiss


class Algorithm:

    def __init__(self):
        self._n_distances = 0   # cumulative distance counter – update in query()

    def fit(self, train: np.ndarray, **index_params) -> None:
        self._index = faiss.IndexHNSWFlat(train.shape[1], index_params["M"])
        self._index.hnsw.efConstruction = index_params["efConstruction"]
        self._index.add(train)
        faiss.omp_set_num_threads(1)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        self._index.hnsw.efSearch = query_params["ef"]
        _, neighbors = self._index.search(np.expand_dims(query, axis=0).astype(np.float32), k)
        return neighbors[0]

    def get_n_distances(self) -> int:
        return faiss.cvar.hnsw_stats.ndis
