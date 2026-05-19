import numpy as np
import lsh_cpp_module

class Algorithm:

    def __init__(self):
        self._index = lsh_cpp_module.LSHIndex()

    def fit(self, train: np.ndarray, **index_params) -> None:
        n_tables = index_params.get('n_tables', 10)
        n_bits = index_params.get('n_bits', 8)
        
        # Cast to float32 to ensure exact match with C++ float buffer
        train = np.asarray(train, dtype=np.float32)
        self._index.fit(train, n_tables, n_bits)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        # Cast to float32 to ensure exact match with C++ float buffer
        query = np.asarray(query, dtype=np.float32)
        neighbors = self._index.query(query, k)
        return neighbors

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
