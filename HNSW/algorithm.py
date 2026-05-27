import numpy as np
import hnsw_cpp

class Algorithm:

    def __init__(self):
        self._index = hnsw_cpp.HNSWIndex()

    def fit(self, train: np.ndarray, **index_params) -> None:
        M               = int(index_params.get('M',               16))
        ef_construction = int(index_params.get('ef_construction', 100))
        train = np.asarray(train, dtype=np.float32)
        self._index.fit(train, M, ef_construction)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        ef    = int(query_params.get('ef', 50))
        query = np.asarray(query, dtype=np.float32)
        return self._index.query(query, k, ef)

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
