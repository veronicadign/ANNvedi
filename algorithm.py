
import numpy as np
from python.wrapper import LinearANN

class Algorithm:

    def __init__(self):
        self._n_distances = 0   # cumulative distance counter – update in query()

    def fit(self, train: np.ndarray, **index_params) -> None:
        self._index = LinearANN()
        self._index.fit(train)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        neighbors = self._index.query(query, k)
        return neighbors

    def get_n_distances(self) -> int:
        return self._index.total_distances_count()
