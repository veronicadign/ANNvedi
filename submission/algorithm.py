import numpy as np

import hnsw_cpp


class Algorithm:
    """ANNvedi submission: HNSW with SQ8 / per-dimension SQ8 quantization and
    optional diversity-heuristic neighbor selection (Malkov & Yashunin Alg. 4).

    Parameters come EXCLUSIVELY from scenarios.yaml (index_params /
    query_params, resolved per scenario and per dataset by the harness).
    Every block was measured on the full public datasets; nothing is tuned
    at fit() time — build time is a scored metric.
    """

    def __init__(self):
        self._index = None

    def fit(self, train: np.ndarray, **index_params) -> None:
        backend = str(index_params.pop('backend', 'hnsw')).lower()
        if backend != 'hnsw':
            raise ValueError(f"this submission ships only the hnsw backend, got {backend!r}")
        self._index = hnsw_cpp.HNSWIndex()
        # Fallbacks mirror the cross-dataset 'default' block; every shipped
        # scenarios.yaml block sets all of these explicitly.
        self._index.fit(
            np.asarray(train, dtype=np.float32),
            int(index_params.get('M', 16)),
            int(index_params.get('ef_construction', 100)),
            str(index_params.get('mode', 'sq8')),
            bool(index_params.get('heuristic', True)),
            bool(index_params.get('reorder', True)),
        )

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        ef = int(query_params.get('ef', 120))
        return self._index.query(np.asarray(query, dtype=np.float32), k, ef)

    def get_n_distances(self) -> int:
        return self._index.total_distances_count() if self._index is not None else 0
